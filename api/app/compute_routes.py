"""A bounded CUDA-only feasibility job, governed by the same scheduler and budget."""

from fastapi import APIRouter, Depends, Request

from .common import APIError, admin, current_user, get_store, ok
from .gpu.colab import GoogleColabProvider
from .gpu.compute import GpuConfig
from .gpu.google import GoogleService
from .gpu.kaggle import KaggleProvider
from .gpu.kaggle import enabled as kaggle_enabled
from .gpu.manifest import digest
from .gpu.repository import Repository

router = APIRouter(prefix="/api/gpu")


def _require_account(repo, owner_id: str, account_id: str | None, *, require_allocation: bool = True) -> dict | None:
    """Resolve an explicit account without allowing owner or state fallback."""
    if account_id in (None, ""):
        return None
    if not isinstance(account_id, str):
        raise APIError("ACCOUNT_INVALID", "account_id必须为字符串", 422)
    try:
        account = repo.get("accounts", account_id, owner_id)
    except (AttributeError, KeyError, TypeError, ValueError):
        account = None
    if not account:
        try:
            account = next(
                (
                    row
                    for row in repo.list("accounts", owner_id)
                    if row.get("id") == account_id or row.get("account_id") == account_id
                ),
                None,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            account = None
    if not account:
        raise APIError("ACCOUNT_NOT_FOUND", "指定的Google账号不存在或不属于当前用户", 404)
    if account.get("enabled", True) is False or account.get("disabled", False) is True:
        raise APIError("ACCOUNT_DISABLED", "指定的Google账号已停用", 409)
    if require_allocation and account.get("allow_allocation", True) is False:
        raise APIError("ACCOUNT_UNAVAILABLE", "指定账号不允许分配GPU", 409)
    if account.get("authorized", account.get("connected", True)) is False:
        raise APIError("ACCOUNT_UNAUTHORIZED", "指定账号尚未完成授权", 409)
    return account


def _provider(repo, owner_id: str, account_id: str | None):
    """Construct providers with account binding while tolerating old test doubles."""
    if kaggle_enabled():
        return KaggleProvider(repo, owner_id)
    try:
        return GoogleColabProvider(repo, owner_id, account_id=account_id)
    except TypeError as exc:
        if "account_id" not in str(exc):
            raise
        return GoogleColabProvider(repo, owner_id)


@router.post("/admin/compute/smoke")
def cuda_smoke(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    cfg = GpuConfig.from_env()
    if body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "此测试会真实消费点数，必须明确确认", 403)
    if not cfg.real_enabled:
        raise APIError("REAL_GPU_DISABLED", "真实 GPU 开关尚未启用")
    if not kaggle_enabled() and (cfg.unit_rate is None or cfg.unit_rate <= 0):
        raise APIError("RATE_UNKNOWN", "必须先填写每分钟估算费率和预算")
    account_id = body.get("account_id")
    if not isinstance(account_id, str) or not account_id.strip():
        raise APIError("ACCOUNT_REQUIRED", "真实冒烟必须明确选择Google账号", 422)
    key = request.headers.get("Idempotency-Key")
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "必须提供固定幂等键，重试沿用原值")
    repo = Repository(st)
    _require_account(repo, user["id"], account_id)
    previous = next((j for j in repo.list("jobs", user["id"]) if j.get("idempotency_key") == key), None)
    raw = {
        "kind": "system-cuda-smoke",
        "requested_spec": body.get("spec"),
        "max_runtime_minutes": min(5, cfg.max_runtime_minutes),
        "execution_policy": {
            "provider": "kaggle" if kaggle_enabled() else "colab",
            "account_id": account_id,
            "max_runtime_minutes": min(5, cfg.max_runtime_minutes),
            "shutdown_after_job": True,
        },
    }
    if previous:
        if previous.get("body_hash") != digest(raw):
            raise APIError("IDEMPOTENCY_CONFLICT", "幂等键已用于其他参数", 409)
        return ok({"id": previous["id"], "job_id": previous["id"], "state": previous["state"]}, request, 202)
    google = GoogleService(repo, user["id"], account_id=account_id)
    if not kaggle_enabled() and not google.status()["colab"]["connected"]:
        raise APIError("GOOGLE_NOT_CONNECTED", "请先完成个人 Colab OAuth 授权")
    specs = _provider(repo, user["id"], account_id).list_specs()
    eligible = [
        s
        for s in specs
        if s.get("eligible") is True and (s.get("key") or {}).get("variant") in {"GPU", "VARIANT_GPU"}
    ]
    requested = body.get("spec")
    if requested:
        eligible = [s for s in eligible if s.get("key") == requested.get("key", requested)]
    if not eligible:
        raise APIError("GPU_UNAVAILABLE", "当前账号没有符合请求的可用 GPU 规格")
    project_id = "system-smoke-" + user["id"][:40]
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请稍后用相同幂等键重试", 503, True)
        if not st.get("projects", project_id, user["id"]):
            st.create(
                "projects", user["id"], {"title": "GPU 可行性检查", "system": True}, record_id=project_id
            )
        job, created = repo.create_job(user["id"], raw, key)
        if created:
            snapshot = {
                "system_smoke": True,
                "manifest": {
                    "executor": "cuda-smoke",
                    "parameters": {},
                    "state": "published",
                    "reviewed": True,
                },
                "resolved_params": {},
                "api_graph": {},
                "bound_graph": {},
                "model_revision": None,
                "workflow_sha256": digest({"system": "cuda-smoke", "version": 1}),
                "dependency_lock": {},
            }
            job = repo.update(
                "jobs",
                job["id"],
                {
                    "project_id": project_id,
                    "provider": "kaggle" if kaggle_enabled() else "colab",
                    "account_id": account_id,
                    "requested_account_id": account_id,
                    "execution_policy": raw["execution_policy"],
                    "model_id": "system-cuda-smoke",
                    "workflow_key": "system/cuda-smoke",
                    "workflow_version": 1,
                    "snapshot": snapshot,
                    "requested_spec": eligible[0],
                    "max_runtime_minutes": raw["max_runtime_minutes"],
                    "estimated_units": (0.0 if kaggle_enabled() else raw["max_runtime_minutes"] * cfg.unit_rate),
                    "unit_rate": cfg.unit_rate,
                },
                user["id"],
            )
            repo.event(
                job["id"],
                user["id"],
                "queued",
                f"已确认真实 CUDA 冒烟：仅测试 GPU，最长{raw['max_runtime_minutes']}分钟，结束即释放运行时",
            )
    return ok(
        {
            "id": job["id"],
            "job_id": job["id"],
            "state": job["state"],
            "estimated_units": job["estimated_units"],
        },
        request,
        202,
    )
