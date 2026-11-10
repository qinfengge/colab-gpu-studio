"""Independent personal AI Studio API."""

from __future__ import annotations

import copy
import io
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from .account_routes import router as account_router
from .admission import _drive, admit_job, repo_registry
from .atelier_routes import router as atelier_router
from .batch_routes import build_batch_router
from .common import APIError, admin, current_user, envelope, get_store, ok, own
from .compute_routes import _require_account
from .compute_routes import router as compute_router
from .config import settings
from .db import Store
from .gpu import loras as lora_registry
from .gpu.compute import GpuConfig, MockColabProvider
from .gpu.google import GoogleError
from .gpu.manifest import bind_parameters, resolve_parameters, validate_manifest
from .gpu.rates import estimate as estimate_units
from .gpu.rates import mode_param
from .gpu.registry import Registry
from .gpu.repository import Repository
from .gpu.scheduler import Scheduler
from .history_routes import router as history_router
from .instance_routes import router as instance_router
from .job_history import job_timing_details, job_timings
from .job_schemas import JobInput
from .oauth_config_routes import router as oauth_config_router
from .presenters import public_asset, public_job
from .review_routes import router as review_router
from .security import sign_session
from .storage import asset_path, delete_asset_file, save_asset


def usage(st, owner):
    repo = Repository(st)
    cfg = GpuConfig.from_env()
    now = time.time()
    rows = repo.list("usage", owner)
    real = [r for r in rows if not r.get("mock") and not r.get("simulated") and r.get("provider") != "mock"]
    estimate = sum(float(r.get("units_estimate", r.get("units", 0)) or 0) for r in real)
    daily = sum(
        float(r.get("units_estimate", r.get("units", 0)) or 0)
        for r in real
        if float(r.get("created_epoch", 0)) > now - 86400
    )
    leases = [r for r in repo.list("leases", owner) if r.get("state") != "deleted"]
    measured = [r for r in real if r.get("units_actual") is not None]
    measured_actual = sum(float(r.get("units_actual") or 0) for r in measured)
    measured_estimate = sum(float(r.get("units_estimate", r.get("units", 0)) or 0) for r in measured)
    from .gpu import ccu

    ccu_state = ccu.read_state()
    recent = []
    jobs_sorted = sorted(
        repo.list("jobs", owner), key=lambda item: str(item.get("updated_at") or ""), reverse=True
    )
    for job in jobs_sorted[:12]:
        if job.get("provider") == "mock":
            continue
        recent.append(
            {
                "job_id": job.get("id"),
                "state": job.get("state"),
                "finished_at": job.get("stopped_at") or job.get("updated_at"),
                "estimated_units": job.get("estimated_units"),
                "actual_units": job.get("actual_units"),
                "ccu_before": job.get("ccu_before"),
                "ccu_after": job.get("ccu_after"),
                "ccu_attribution": job.get("ccu_attribution"),
                "ccu_source": job.get("ccu_source"),
            }
        )
    return {
        "estimated_units": estimate if cfg.unit_rate is not None else None,
        "actual_units": measured_actual if measured else None,
        "daily_estimated_units": daily if cfg.unit_rate is not None else None,
        "unit_rate": cfg.unit_rate,
        "rate_unit": "compute_units_per_minute",
        "rate_source": "administrator_estimate" if cfg.unit_rate is not None else "unknown",
        "daily_limit": cfg.daily_units,
        "monthly_limit": cfg.monthly_units,
        "max_job_units": cfg.max_job_units,
        "max_runtime_minutes": cfg.max_runtime_minutes,
        "active_runtimes": len(leases),
        "ledger": rows,
        "reconciliation": {
            "measured_jobs": len(measured),
            "unmeasured_jobs": len(real) - len(measured),
            "measured_actual_units": round(measured_actual, 6) if measured else None,
            "measured_estimated_units": round(measured_estimate, 6) if measured else None,
            "variance_units": round(measured_actual - measured_estimate, 6) if measured else None,
            "recent": recent,
        },
        "ccu": {
            **ccu.summarize(ccu_state.get("snapshots") or {}),
            "sampled_at": ccu_state.get("updated_at"),
            "snapshots": ccu_state.get("snapshots") or {},
            "anomalies": ccu_state.get("anomalies") or [],
        },
        "notice": (
            "月费/实扣以 Google 账户为准：实测值取任务前后余额差；"
            "估算值按管理员费率推算，可能与实际不一致。Mock 不创建 GPU，不产生真实点数消费。"
        ),
    }


@asynccontextmanager
async def lifespan(app):
    if not getattr(app.state, "store", None):
        app.state.store = Store(settings.database_url)
    from .gpu.instances import InstanceService
    from .gpu.workbench import WorkbenchService

    app.state.repo = Repository(app.state.store)
    app.state.instance_service = InstanceService(app.state.repo, config=GpuConfig.from_env())

    def admit_workbench(owner, body, key, *, instance=None):
        parsed = JobInput.model_validate(body)
        return admit_job(app.state.store, {"id": owner}, parsed.model_dump(), key)

    app.state.workbench_service = WorkbenchService(
        repo=app.state.repo, registry=Registry(app.state.repo), admit_job=admit_workbench
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Colab GPU Studio", version=__version__, lifespan=lifespan)
router = APIRouter()
_login_limits = {}


@app.middleware("http")
async def security_context(request, call_next):
    request.state.request_id = str(uuid.uuid4())
    try:
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != settings.site_url.rstrip("/"):
                from urllib.parse import urlsplit

                if urlsplit(origin).netloc != request.url.netloc:
                    raise APIError("CSRF_ORIGIN", "跨站写入被拒绝", 403)
        if int(request.headers.get("Content-Length", "0")) > settings.max_upload_bytes + 1048576:
            raise APIError("PAYLOAD_TOO_LARGE", "请求超过上传限制", 413)
        response = await call_next(request)
    except APIError as exc:
        response = error_response(request, exc)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "private, no-store"
    return response


def error_response(request, exc):
    error = {"code": exc.code, "message": exc.message, "retryable": exc.retryable}
    result = envelope(error=error, request_id=getattr(request.state, "request_id", None))
    result.update(code=exc.code, message=exc.message)
    return JSONResponse(result, status_code=exc.status)


@app.exception_handler(APIError)
async def api_error(request, exc):
    return error_response(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Do not return rejected input values (they may contain secrets).
    return error_response(
        request, APIError("VALIDATION_ERROR", "请求格式错误，请检查参数类型与必填字段", 422)
    )


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return error_response(request, APIError("VALIDATION_ERROR", str(exc)[:300], 422))


@app.exception_handler(KeyError)
async def key_error(request, exc):
    return error_response(request, APIError("NOT_FOUND", "资源不存在", 404))


@app.exception_handler(Exception)
async def internal_error(request, exc):
    return error_response(request, APIError("INTERNAL_ERROR", "后台操作失败，请查看已脱敏的任务事件", 500))


@app.get("/api/health")
def health(st=Depends(get_store)):
    from sqlalchemy import text

    with st.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok", "version": __version__, "application": "colab-gpu-studio", "independent": True}


class Login(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


@app.post("/api/auth/login")
def login(body: Login, request: Request):
    key = request.client.host if request.client else "unknown"
    previous = [t for t in _login_limits.get(key, []) if t > time.monotonic() - 60]
    if len(previous) >= 10:
        raise APIError("RATE_LIMITED", "登录尝试过于频繁，请稍后重试", 429)
    previous.append(time.monotonic())
    _login_limits[key] = previous
    if not settings.studio_password or not secrets.compare_digest(body.password, settings.studio_password):
        raise APIError("INVALID_CREDENTIALS", "密码不正确或尚未初始化", 401)
    token = sign_session(
        {"uid": "owner", "role": "admin", "iat": int(time.time()), "exp": int(time.time()) + 43200}
    )
    response = JSONResponse({"access_token": token, "token_type": "bearer", "expires_in": 43200})
    response.set_cookie(
        "studio_session",
        token,
        httponly=True,
        secure=settings.site_url.startswith("https://"),
        samesite="lax",
        max_age=43200,
    )
    return response


@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


@router.get("/projects")
def projects(request: Request, user=Depends(current_user), st=Depends(get_store)):
    return ok(st.list("projects", user["id"]), request)


@router.post("/projects")
def create_project(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    title = str(body.get("title", "")).strip()[:200]
    if not title:
        raise APIError("VALIDATION_ERROR", "项目名称不能为空")
    return ok(
        st.create(
            "projects",
            user["id"],
            {"title": title, "atelier_project_id": str(body.get("atelier_project_id", ""))[:128]},
        ),
        request,
        201,
    )


@router.patch("/projects/{project_id}")
def update_project(project_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    own(st, "projects", project_id, user)
    if "title" not in body:
        raise APIError("PROJECT_INVALID", "仅支持修改本地项目名称")
    title = str(body["title"]).strip()[:200]
    if not title:
        raise APIError("PROJECT_INVALID", "项目名不能为空")
    return ok(st.update("projects", project_id, {"title": title, "atelier_project_id": str(body.get("atelier_project_id", ""))[:128]}, user["id"]), request)


@router.get("/models")
def models(request: Request, user=Depends(current_user), st=Depends(get_store)):
    _, registry = repo_registry(st, user)
    return ok(
        [
            {
                **r,
                "display_name": r.get("display_name", r.get("name", r["model_id"])),
                "reason": r.get("disabled_reason"),
            }
            for r in registry.models(user["id"])
        ],
        request,
    )


@router.post("/models/register")
def register_model(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    repo, _ = repo_registry(st, user)
    model_id = str(body.get("model_id", "")).strip()
    if (
        not model_id
        or len(model_id) > 100
        or any(x.get("model_id") == model_id for x in repo.list("models", user["id"]))
    ):
        raise APIError("MODEL_INVALID", "模型 ID 无效或已存在")
    allowed = {
        "model_id",
        "display_name",
        "revision",
        "source",
        "license",
        "modality",
        "resource_profile",
        "variants",
    }
    payload = {k: v for k, v in body.items() if k in allowed}
    payload.update(
        enabled=False, is_mock=False, disabled_reason="待管理员来源审查、许可验证和真实 GPU 冒烟测试"
    )
    return ok(repo.create("models", user["id"], payload), request, 201)


@router.get("/workflows")
def workflows(
    request: Request, model_id: str | None = None, user=Depends(current_user), st=Depends(get_store)
):
    _, registry = repo_registry(st, user)
    return ok(registry.workflows(user["id"], model_id), request)


@router.get("/workflows/{workflow_id}/schema")
def workflow_schema(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    manifest = row["manifest"]
    cap = min(manifest.get("max_runtime_minutes", 30), GpuConfig.from_env().max_runtime_minutes)
    spec = manifest.get("compute_spec")
    picked = mode_param(manifest)
    if picked:
        estimates = {m: estimate_units(repo, user["id"], row["workflow_key"], manifest, spec, cap, mode=m)[0] for m in picked[1]["variants"]}
        return ok({**manifest, "estimated_units": estimates[picked[1]["default"]], "estimates": estimates}, request)
    units, _ = estimate_units(repo, user["id"], row["workflow_key"], manifest, spec, cap)
    return ok({**manifest, "estimated_units": units}, request)




def cloud_sync_workflow(st, owner, row):
    if os.getenv("GPU_STORAGE", "local") != "drive":
        return row
    from .gpu.drive import DriveStorage

    storage = DriveStorage({"owner_id": owner}, Repository(st))
    folder = f"workflows/{row['workflow_key']}/{row['workflow_version']}"
    records = {}
    for name, field in [
        ("manifest.json", "manifest"),
        ("workflow.api.json", "api_graph"),
        ("workflow.editable.json", "editable_graph"),
        ("object_info.json", "object_info"),
    ]:
        content = json.dumps(row.get(field, {}), ensure_ascii=False, sort_keys=True).encode()
        record = storage.put(folder + "/" + name, content, "application/json")
        if not storage.verify(record, content):
            raise APIError("DRIVE_VERIFY_FAILED", "工作流保存后校验失败")
        records[field] = record
    return Repository(st).update("workflows", row["id"], {"drive_files": records}, owner)


@router.post("/workflows/import")
def import_workflow(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    if len(json.dumps(body)) > 5 * 1024 * 1024:
        raise APIError("WORKFLOW_TOO_LARGE", "工作流 JSON 过大", 413)
    repo, registry = repo_registry(st, user)
    manifest = body.get("manifest") or {}
    body = {
        **body,
        "workflow_key": body.get("workflow_key", manifest.get("workflow_key")),
        "model_id": body.get("model_id", manifest.get("model_id")),
    }
    row = registry.import_workflow(user["id"], body)
    row = cloud_sync_workflow(st, user["id"], row)
    return ok(row, request, 201)


@router.post("/workflows/{workflow_id}/validate")
def validate_workflow(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    _, registry = repo_registry(st, user)
    return ok(registry.validate(user["id"], workflow_id), request)


@router.post("/workflows/{workflow_id}/smoke-test")
def smoke_workflow(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    repo, registry = repo_registry(st, user)
    row = own(repo, "workflows", workflow_id, user)
    model = next((m for m in registry.models(user["id"]) if m["model_id"] == row["model_id"]), None)
    if not model or not model.get("is_mock"):
        raise APIError(
            "REAL_SMOKE_REQUIRES_REVIEW",
            "真实工作流须先通过来源审查并完成受预算保护的 GPU 冒烟；不能提交客户端伪造证据",
        )
    validate_manifest(row["manifest"], row["api_graph"], row["object_info"])
    params = resolve_parameters(row["manifest"], {})
    graph = bind_parameters(row["manifest"], row["api_graph"], params)
    from .gpu.compute import MockGPUExecutor

    test_job = {
        "id": str(uuid.uuid4()),
        "provider": "mock",
        "snapshot": {"resolved_params": params, "bound_graph": graph},
        "execution_id": str(uuid.uuid4()),
    }
    executor = MockGPUExecutor()
    executor.start({"runtime_name": "mock-static-smoke"}, test_job, test_job["execution_id"])
    result = executor.poll({"runtime_name": "mock-static-smoke"}, test_job)
    if result.get("state") != "completed" or not result.get("outputs"):
        raise APIError("SMOKE_FAILED", "模拟执行冒烟未通过")
    return ok(
        repo.update(
            "workflows", workflow_id, {"smoke_test_passed": True, "smoke_test_kind": "mock"}, user["id"]
        ),
        request,
    )


@router.post("/workflows/{workflow_id}/publish")
def publish_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    admin(user)
    _, registry = repo_registry(st, user)
    return ok(registry.publish(user["id"], workflow_id, confirmed=body.get("confirmed") is True), request)


@router.post("/workflows/rollback")
def rollback_workflow(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    _, registry = repo_registry(st, user)
    return ok(registry.rollback(user["id"], body.get("workflow_key"), body.get("version")), request)


@router.post("/workflows/{workflow_id}/deactivate")
def deactivate_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    admin(user)
    if not isinstance(body, dict) or body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "停用必须明确confirmed")
    _, registry = repo_registry(st, user)
    return ok(registry.deactivate(user["id"], workflow_id), request)


@router.get("/presets")
def presets(
    request: Request,
    workflow_version_id: str | None = None,
    user=Depends(current_user),
    st=Depends(get_store),
):
    rows = Repository(st).list("presets", user["id"])
    return ok(
        [r for r in rows if not workflow_version_id or r.get("workflow_version_id") == workflow_version_id],
        request,
    )


@router.post("/presets")
def save_preset(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    workflow = own(repo, "workflows", body.get("workflow_version_id"), user)
    overrides = body.get("overrides", {})
    resolve_parameters(workflow["manifest"], overrides)
    row = repo.create(
        "presets",
        user["id"],
        {
            "name": str(body.get("name", "未命名预设"))[:100],
            "workflow_version_id": workflow["id"],
            "overrides": overrides,
        },
    )
    return ok(row, request, 201)








@router.get("/loras")
def list_loras(request: Request, workflow_key: str | None = None, user=Depends(current_user), st=Depends(get_store)):
    # LoRAs always live in Drive (like the model cache), whatever GPU_STORAGE says about results.
    files, error = [], None
    try:
        files = _drive(Repository(st), user).list_loras()
    except Exception:  # noqa: BLE001 - Drive down must not hide the curated list
        error = "暂时读不到 Drive 里的 LoRA 文件夹"
    return ok({"curated": lora_registry.curated_for(workflow_key), "files": files, "folder": "AI-Studio/我的文件夹/loras/h3",
               "max_bytes": lora_registry.MAX_BYTES, "error": error}, request)


@router.post("/loras/upload")
def lora_upload(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    name, size = str(body.get("name") or "").strip(), body.get("size")
    if "/" in name or not lora_registry.is_lora_file(name, size if isinstance(size, int) else None):
        raise APIError("VALIDATION_ERROR", "只接受 1.5GB 以内的 .safetensors 文件")
    origin = request.headers.get("origin") or f"{request.url.scheme}://{request.url.netloc}"
    return ok({"upload_url": _drive(Repository(st), user).lora_upload_session(name, size, origin)}, request)


@router.post("/jobs", status_code=202)
def create_job(
    body: JobInput,
    request: Request,
    idempotency_key: str | None = Header(default=None),
    user=Depends(current_user),
    st=Depends(get_store),
):
    return ok(public_job(admit_job(st, user, body.model_dump(), idempotency_key)), request, 202)


@router.get("/jobs")
def jobs(request: Request, project_id: str | None = None, user=Depends(current_user), st=Depends(get_store)):
    if project_id:
        own(st, "projects", project_id, user)
    return ok([public_job(j) for j in Repository(st).list("jobs", user["id"], project_id)], request)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    job = own(repo, "jobs", job_id, user)
    result = public_job(job)
    result["events"] = [e for e in repo.list("events", user["id"]) if e.get("job_id") == job_id][::-1]
    result["logs"] = result["events"]
    now = time.time()
    result["timings"] = job_timings(job, result["events"], now=now)
    result["timing_details"] = job_timing_details(job, result["events"], now=now)
    allowed={'id','state','attempt','started_at','finished_at','error_code'}
    result['attempts']=sorted([{k:v for k,v in row.items() if k in allowed}
        for row in repo.list('executions',user['id']) if row.get('job_id')==job_id],key=lambda row:row.get('attempt',0))
    return ok(result, request)


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    own(repo, "jobs", job_id, user)
    return ok([e for e in repo.list("events", user["id"]) if e.get("job_id") == job_id][::-1], request)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    own(Repository(st), "jobs", job_id, user)
    return ok(public_job(Scheduler(st).cancel(user["id"], job_id)), request)


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    original = own(Repository(st), "jobs", job_id, user)
    if original["state"] not in {"failed", "cancelled", "completed"}:
        raise APIError("JOB_ACTIVE", "原任务尚未结束，不能重复提交")
    key = request.headers.get("Idempotency-Key")
    if not key:
        raise APIError("IDEMPOTENCY_REQUIRED", "再次生成需提供新的幂等键")
    raw=copy.deepcopy(original.get('request_body') or {})
    if raw.get('kind') or any((original.get('snapshot') or {}).get(flag) or original.get(flag) for flag in ('system_smoke','workflow_smoke','prep')):
        raise APIError('SPECIAL_JOB_RETRY_REQUIRED','诊断或预热任务需回到对应入口重新确认',409)
    if original.get('resolved_seed') is not None:
        raw.setdefault('params',{})['seed']=original['resolved_seed']
    raw['retry_of']=original['id']
    return ok(public_job(admit_job(st, user, raw, key)), request, 202)


@router.post('/jobs/{job_id}/attempts')
def retry_attempt(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo=Repository(st)
    own(repo,'jobs',job_id,user)
    try:
        job=repo.retry_attempt(job_id,user['id'],request.headers.get('Idempotency-Key'))
    except ValueError as exc:
        raise APIError('ATTEMPT_RETRY_REJECTED',str(exc),409) from exc
    return ok(public_job(job),request,202)


@router.get('/jobs/{job_id}/attempts')
def job_attempts(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo=Repository(st)
    own(repo,'jobs',job_id,user)
    allowed={'id','job_id','state','attempt','started_at','finished_at','error_code','created_at'}
    rows=[{k:v for k,v in row.items() if k in allowed} for row in repo.list('executions',user['id']) if row.get('job_id')==job_id]
    return ok(sorted(rows,key=lambda row:row.get('attempt',0)),request)


@router.get("/jobs/{job_id}/outputs")
def outputs(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    job = own(Repository(st), "jobs", job_id, user)
    rows = [own(st, "assets", aid, user) for aid in job.get("output_asset_ids", [])]
    return ok([public_asset(a) for a in rows], request)


@router.get("/gallery")
def gallery(
    request: Request, project_id: str | None = None, user=Depends(current_user), st=Depends(get_store)
):
    if project_id:
        own(st, "projects", project_id, user)
    return ok(
        [
            public_asset(a)
            for a in st.list("assets", user["id"], project_id)
            if a.get("source_job_id") or a.get("gpu_job_id")
        ],
        request,
    )


@app.get("/api/assets")
def reference_assets(request: Request, project_id: str, user=Depends(current_user), st=Depends(get_store)):
    own(st, "projects", project_id, user)
    rows = [
        public_asset(a)
        for a in st.list("assets", user["id"], project_id)
        if not a.get("mock")
        and not (a.get("metadata") or {}).get("mock")
        and str(a.get("mime", "")).startswith(("image/", "video/", "audio/"))
    ]
    return ok(rows, request)


@app.post("/api/assets")
async def upload_asset(
    request: Request,
    project_id: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(current_user),
    st=Depends(get_store),
):
    own(st, "projects", project_id, user)
    content = await file.read(settings.max_upload_bytes + 1)
    asset = save_asset(st, user["id"], project_id, content, file.filename or "upload.bin")
    from .gpu.kaggle import enabled as kaggle_enabled
    if kaggle_enabled() and os.getenv('GPU_STORAGE', 'local') == 'drive':
        from starlette.concurrency import run_in_threadpool

        from .kaggle_storage import sync_input_asset
        try:
            asset = await run_in_threadpool(sync_input_asset, st, user['id'], asset)
        except APIError:
            asset = st.update('assets', asset['id'], {'drive_sync_status': 'failed'}, user['id'])
    return ok(public_asset(asset), request, 201)


@app.post('/api/assets/{asset_id}/sync-drive')
def sync_asset_drive(asset_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .kaggle_storage import sync_input_asset
    asset = own(st, 'assets', asset_id, user)
    if asset.get('source_job_id'):
        raise APIError('INPUT_REQUIRED', '此接口只同步原始素材', 422)
    return ok(public_asset(sync_input_asset(st, user['id'], asset)), request)


@app.get("/api/assets/{asset_id}/download")
@app.get("/api/assets/{asset_id}/preview")
def download_asset(asset_id: str, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    return FileResponse(asset_path(asset), media_type=asset["mime"], filename=asset["filename"])


@app.get("/api/assets/{asset_id}/thumbnail")
def thumbnail(asset_id: str, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    if not asset["mime"].startswith("image/"):
        raise APIError("THUMBNAIL_UNAVAILABLE", "非图片素材不提供图片缩略图", 404)
    with Image.open(asset_path(asset)) as image:
        image.thumbnail((480, 480))
        target = io.BytesIO()
        image.convert("RGB").save(target, "WEBP", quality=80)
    return Response(target.getvalue(), media_type="image/webp")


@app.patch("/api/assets/{asset_id}")
def favorite_asset(
    asset_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    own(st, "assets", asset_id, user)
    if not isinstance(body.get("favorite"), bool):
        raise APIError("VALIDATION_ERROR", "favorite必须为布尔值")
    return ok(
        public_asset(st.update("assets", asset_id, {"favorite": body["favorite"]}, user["id"])), request
    )


@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    delete_asset_file(asset)
    st.delete("assets", asset_id, user["id"])
    return ok({"deleted": True}, request)


@router.get("/usage")
def get_usage(request: Request, user=Depends(current_user), st=Depends(get_store)):
    return ok(usage(st, user["id"]), request)


@router.get("/settings")
def get_settings(request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    cfg = GpuConfig.from_env()
    return ok(
        {
            "offline_mode": os.getenv("STUDIO_OFFLINE_TEST") == "1",
            "provider_default": "mock",
            "compute_provider": os.getenv("GPU_COMPUTE_PROVIDER", "colab"),
            "real_enabled": cfg.real_enabled,
            "google": GoogleService(Repository(st), user["id"]).status(),
            "budget": usage(st, user["id"]),
            "storage": os.getenv("GPU_STORAGE", "local"),
            "independent": True,
        },
        request,
    )


@router.get("/compute/specs")
def compute_specs(
    request: Request,
    provider: str = "mock",
    account_id: str | None = None,
    user=Depends(current_user),
    st=Depends(get_store),
):
    if provider == "mock":
        return ok(MockColabProvider().list_specs(), request)
    if provider == "kaggle":
        from .gpu.kaggle import KaggleProvider
        if not GpuConfig.from_env().real_enabled or os.getenv("STUDIO_OFFLINE_TEST") == "1":
            raise APIError("REAL_GPU_DISABLED", "真实 GPU 尚未启用", 409)
        return ok(KaggleProvider(Repository(st), user['id']).list_specs(), request)
    if provider != "colab":
        raise APIError("PROVIDER_INVALID", "不支持该 Provider")
    if os.getenv("STUDIO_OFFLINE_TEST") == "1":
        raise APIError("OFFLINE_MODE", "站点处于离线测试模式，不能连接 Colab。可改查模拟 GPU。", status=409)
    if not GpuConfig.from_env().real_enabled:
        raise APIError("REAL_GPU_DISABLED", "真实 GPU 未启用，不能连接 Colab。可改查模拟 GPU。", status=409)
    repo = Repository(st)
    _require_account(repo, user["id"], account_id, require_allocation=False)
    from .gpu.colab import GoogleColabProvider

    try:
        return ok(GoogleColabProvider(repo, user["id"], account_id=account_id or None).list_specs(), request)
    except APIError:
        raise
    except Exception as exc:
        raise APIError("COLAB_CHECK_FAILED", f"检查 Colab GPU 失败：{str(exc)[:160]}", status=502, retryable=True) from exc


@router.get("/compute/subscription")
def subscription(
    request: Request,
    account_id: str | None = None,
    user=Depends(current_user),
    st=Depends(get_store),
):
    from .gpu.colab import GoogleColabProvider

    repo = Repository(st)
    _require_account(repo, user["id"], account_id, require_allocation=False)
    return ok(GoogleColabProvider(repo, user["id"], account_id=account_id or None).subscription(), request)


@router.post("/admin/compute/stop")
def emergency_stop(request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    return ok(Scheduler(st).stop(user["id"]), request)


@router.post("/auth/google/start")
def google_start(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    result = GoogleService(Repository(st), user["id"]).start(body.get("purpose"))
    return ok({"authorization_url": result["authorization_url"]}, request)


@router.get("/auth/google/callback")
def google_callback(code: str, state: str, st=Depends(get_store)):
    from .gpu.google import GoogleService

    GoogleService(Repository(st), None).callback(code, state)
    return RedirectResponse("/?google=connected", status_code=303)


@router.post("/auth/google/disconnect")
def google_disconnect(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    return ok(GoogleService(Repository(st), user["id"]).disconnect(body.get("purpose")), request)


# White-listed Agent API: it is not a shell and has no budget or compute-create tool.
AGENT_TOOLS = {
    "workflow.list",
    "workflow.import_draft",
    "workflow.validate",
    "model.list",
    "model.inspect",
    "model.register_draft",
    "job.create",
    "job.get",
    "job.cancel",
    "job.list_logs_redacted",
    "storage.list_project_files",
    "usage.inspect",
}


def agent_user(request):
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
    expected = os.getenv("GPU_AGENT_TOKEN", "")
    owner = os.getenv("GPU_AGENT_OWNER_ID", "")
    if expected and owner and supplied and secrets.compare_digest(supplied, expected):
        return {"id": owner, "role": "agent"}
    return current_user(request)


@router.get("/agent/tools")
def agent_tools(request: Request):
    agent_user(request)
    return ok(sorted(AGENT_TOOLS), request)


@router.post("/agent/invoke")
def agent_invoke(body: dict, request: Request, st=Depends(get_store)):
    user = agent_user(request)
    tool, args = body.get("tool"), body.get("arguments", {})
    if tool not in AGENT_TOOLS or not isinstance(args, dict):
        raise APIError("TOOL_FORBIDDEN", "Agent 不具备该工具权限；预算、发布与直接开机不可绕过", 403)
    repo, registry = repo_registry(st, user)
    if tool == "usage.inspect":
        result = usage(st, user["id"])
    elif tool in {"workflow.list", "model.list"}:
        result = registry.workflows(user["id"]) if tool == "workflow.list" else registry.models(user["id"])
    elif tool == "model.inspect":
        result = next((m for m in registry.models(user["id"]) if m["model_id"] == args.get("model_id")), None)
    elif tool == "model.register_draft":
        if not args.get("model_id"):
            raise APIError("MODEL_INVALID", "缺少模型 ID")
        result = repo.create(
            "models",
            user["id"],
            {
                k: v
                for k, v in args.items()
                if k in {"model_id", "display_name", "source", "revision", "license", "resource_profile"}
            }
            | {"enabled": False, "is_mock": False, "disabled_reason": "待管理员审查"},
        )
    elif tool == "workflow.import_draft":
        result = registry.import_workflow(user["id"], args)
    elif tool == "workflow.validate":
        result = registry.validate(user["id"], args.get("id"))
    elif tool == "job.create":
        key = args.pop("idempotency_key", None)
        result = public_job(admit_job(st, user, JobInput(**args).model_dump(), key))
    elif tool in {"job.get", "job.cancel", "job.list_logs_redacted"}:
        job = own(repo, "jobs", args.get("id"), user)
        if tool == "job.cancel":
            job = Scheduler(st).cancel(user["id"], job["id"])
        result = (
            [e for e in repo.list("events", user["id"]) if e.get("job_id") == job["id"]]
            if tool.endswith("redacted")
            else public_job(job)
        )
    elif tool == "storage.list_project_files":
        own(st, "projects", args.get("project_id"), user)
        result = [public_asset(a) for a in st.list("assets", user["id"], args["project_id"])]
    else:
        raise APIError("TOOL_FORBIDDEN", "工具不可用", 403)
    return ok(result, request)


if os.getenv("STUDIO_ATELIER_ENABLED", "0").strip().lower() in {"1", "true"}:
    app.include_router(atelier_router)
app.include_router(account_router)
app.include_router(instance_router)
app.include_router(history_router)
app.include_router(compute_router)
app.include_router(review_router)
app.include_router(oauth_config_router)


@app.exception_handler(GoogleError)
async def google_error(request, exc):
    return error_response(request, APIError("GOOGLE_REQUEST_FAILED", str(exc)[:250], 422))


router.include_router(build_batch_router(JobInput, admit_job, public_job))
app.include_router(router, prefix="/api/gpu")
# Canonical task-book endpoints are available too; provider switches keep /api/jobs unchanged.
app.include_router(router, prefix="/api", include_in_schema=False)

# Same-origin production PWA. The API always registers before the SPA fallback.

_static_root = Path(__file__).resolve().parents[2] / "web" / "dist"
if not _static_root.exists():
    _static_root = Path(__file__).resolve().parents[3] / "web" / "dist"
if _static_root.exists():
    if (_static_root / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_static_root / "assets"), name="web-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise APIError("NOT_FOUND", "接口不存在", 404)
        target = (_static_root / path).resolve()
        if path and target.is_relative_to(_static_root.resolve()) and target.is_file():
            return FileResponse(target)
        return FileResponse(_static_root / "index.html")
