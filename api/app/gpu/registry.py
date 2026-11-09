"""Immutable, versioned workflow registry and safe bootstrap metadata."""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

from .manifest import digest, validate_manifest
from .repository import Repository

_TEMPLATE_DIR = Path(__file__).with_name("templates") / "demo-mock"
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
# Real publication is intentionally narrower than static validation.  This is
# a fixed review boundary, not a client-provided allowlist.
_PUBLISH_NODE_ALLOWLIST = {
    "MinimaxH3LatentUpscaler3DRefineHandoff",  # pinned + reviewed custom node (runtime_worker.REVIEWED_CUSTOM_NODES)
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "KSampler",
    "VAEDecode",
    "VAEEncode",
    "SaveImage",
    "SaveVideo",
    "EmptyLatentImage",
    "LoadImage",
    "ConditioningCombine",
    "ConditioningAverage",
    "LatentUpscale",
    "UpscaleModel",
    # ComfyUI core nodes used by the reviewed MOSS-Image workflow (checked against v0.43.2 object_info).
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "ModelSamplingAuraFlow",
    "EmptyFlux2LatentImage",
    # Added for the Krea-2 Turbo + MOSS dual-sampling workflow (also core nodes in v0.43.2).
    "ConditioningZeroOut",
    "PrimitiveStringMultiline",
    # Added for the MiniMax H3 image-to-video workflow (core nodes in ComfyUI v0.43.2).
    "MiniMaxH3ImageToVideo",
    "LoraLoaderModelOnly",
    "KSamplerSelect",
    "BasicScheduler",
    "RandomNoise",
    "BasicGuider",
    "SamplerCustomAdvanced",
    "VAEDecodeAudio",
    "CreateVideo",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3SigmaShift",
    "ConditioningKrea2Rebalance",
}


class Registry:
    def __init__(self, repo: Repository):
        self.repo = repo

    def models(self, owner_id):
        return self.repo.list("models", owner_id)

    def workflows(self, owner_id, model_id=None):
        rows = self.repo.list("workflows", owner_id)
        return [row for row in rows if model_id is None or row.get("model_id") == model_id]

    def get_workflow(self, owner_id, key, version):
        for row in self.workflows(owner_id):
            if row.get("workflow_key") == key and row.get("workflow_version") == version:
                return row
        raise KeyError(f"workflow {key}@{version}不存在")

    @staticmethod
    def _template():
        def load(name):
            with (_TEMPLATE_DIR / name).open(encoding="utf-8") as handle:
                return json.load(handle)

        return load("manifest.json"), load("api_graph.json"), load("object_info.json")

    def _bootstrap_locked(self, owner_id):
        known = {row.get("model_id") for row in self.models(owner_id)}
        model_specs = [
            {
                "model_id": "demo-mock",
                "name": "Atelier Demo Mock",
                "enabled": True,
                "is_mock": True,
                "license_status": "not_applicable",
                "description": "离线模拟用途；不生成真实AI媒体，不请求外网GPU。",
            },
            {
                "model_id": "moss",
                "name": "MOSS",
                "enabled": False,
                "is_mock": False,
                "disabled_reason": "待核验模型权重、节点兼容性、许可与实际硬件；当前禁止真实运行。",
            },
            {
                "model_id": "krea",
                "name": "Krea",
                "enabled": False,
                "is_mock": False,
                "disabled_reason": "待核验模型权重来源、许可、节点兼容性与账户/硬件条件；当前禁止真实运行。",
            },
            {
                "model_id": "h3",
                "name": "MiniMax H3",
                "enabled": False,
                "is_mock": False,
                "disabled_reason": "待核验正式许可、权重/运行时与硬件验证；不以公开前端或示例冒充可运行模型。",
            },
        ]
        for spec in model_specs:
            if spec["model_id"] not in known:
                self.repo.create("models", owner_id, spec)
        existing = [
            row
            for row in self.workflows(owner_id)
            if row.get("workflow_key") == "demo-mock" and row.get("workflow_version") == 1
        ]
        if existing:
            demo = existing[0]
        else:
            manifest, graph, info = self._template()
            manifest = validate_manifest(manifest, graph, info)
            demo = self.repo.create(
                "workflows",
                owner_id,
                {
                    "workflow_key": "demo-mock",
                    "workflow_version": 1,
                    "model_id": "demo-mock",
                    "manifest": manifest,
                    "api_graph": graph,
                    "object_info": info,
                    "editable_graph": copy.deepcopy(graph),
                    "sha256": digest({"manifest": manifest, "api_graph": graph, "object_info": info}),
                    "state": "published",
                    "smoke_test_passed": True,
                    "smoke_test_kind": "mock",
                    "active": True,
                    "description": "明确标识的离线模拟模板，仅用于演示。",
                },
            )
        from .builtin import seed_builtin_templates

        seed_builtin_templates(self, owner_id)
        return {"models": self.models(owner_id), "demo_workflow": demo}

    def bootstrap(self, owner_id):
        # Store.dispatch_lock is cross-process for file SQLite/Postgres and
        # shared across in-memory Store instances.  Retry rather than allowing
        # two callers to race the read-then-create bootstrap sequence.
        for attempt in range(100):
            with self.repo.dispatch_lock() as acquired:
                if acquired:
                    return self._bootstrap_locked(owner_id)
            time.sleep(0.005 * min(attempt + 1, 10))
        raise RuntimeError("workflow bootstrap锁获取失败")

    def import_workflow(self, owner_id, bundle):
        if not isinstance(bundle, dict):
            raise ValueError("workflow bundle必须是对象")  # noqa: TRY004
        source = bundle.get("workflow") if isinstance(bundle.get("workflow"), dict) else bundle
        key, model, manifest, graph, info = (
            source.get("workflow_key") or source.get("key"),
            source.get("model_id"),
            source.get("manifest"),
            source.get("api_graph"),
            source.get("object_info"),
        )
        if not (
            isinstance(key, str)
            and key
            and isinstance(model, str)
            and model
            and all(isinstance(x, dict) for x in (manifest, graph, info))
        ):
            raise ValueError("workflow bundle缺少key/model/manifest/graph/object_info")
        manifest_version = manifest.get("version")
        if manifest_version is not None:
            if not isinstance(manifest_version, str) or not _SEMVER.fullmatch(manifest_version):
                raise ValueError("manifest.version必须是semver")
            version = manifest_version
        else:
            version = None
        payload = {
            "workflow_key": key,
            "workflow_version": version,
            "model_id": model,
            "manifest": copy.deepcopy(manifest),
            "api_graph": copy.deepcopy(graph),
            "object_info": copy.deepcopy(info),
            "editable_graph": copy.deepcopy(source.get("editable_graph", graph)),
            "sha256": digest({"manifest": manifest, "api_graph": graph, "object_info": info}),
            "state": "draft",
            "smoke_test_passed": False,
            "smoke_test_kind": None,
            "active": False,
        }
        if version is not None:
            return self.repo.create("workflows", owner_id, payload)
        # Legacy bundles without manifest.version receive an integer version.
        # The unique database identity remains the final race arbiter.
        for _ in range(8):
            versions = []
            for row in self.workflows(owner_id):
                if row.get("workflow_key") == key and isinstance(row.get("workflow_version"), int):
                    versions.append(row["workflow_version"])
            payload["workflow_version"] = max(versions, default=0) + 1
            try:
                return self.repo.create("workflows", owner_id, payload)
            except ValueError as exc:
                if "唯一" not in str(exc) and "存在" not in str(exc):
                    raise
                time.sleep(0.005)
        raise ValueError("workflow版本创建竞争失败，请重试")

    def validate(self, owner_id, id):
        row = self.repo.get("workflows", id, owner_id)
        if row is None:
            raise KeyError(id)
        if row.get("state") == "published":
            raise ValueError("已发布workflow版本不可重新验证或覆写")
        manifest = validate_manifest(row["manifest"], row["api_graph"], row["object_info"])
        return self.repo.update(
            "workflows",
            id,
            {
                "manifest": manifest,
                "sha256": digest(
                    {"manifest": manifest, "api_graph": row["api_graph"], "object_info": row["object_info"]}
                ),
                "state": "validated",
            },
            owner_id,
        )

    def publish(self, owner_id, id, confirmed=False):
        row = self.repo.get("workflows", id, owner_id)
        if row is None:
            raise KeyError(id)
        if row.get("state") == "published":
            raise ValueError("workflow版本已经发布且不可覆写")
        if row.get("state") != "validated":
            raise ValueError("workflow必须先通过validate")
        if not confirmed:
            raise ValueError("发布必须明确confirmed")
        # Re-check the immutable graph and manifest at the publication boundary;
        # a draft may have been edited after its previous validation.
        validate_manifest(row["manifest"], row["api_graph"], row["object_info"])
        model = next(
            (item for item in self.models(owner_id) if item.get("model_id") == row.get("model_id")), None
        )
        if not model or not model.get("enabled"):
            raise ValueError("模型未启用，不能发布")
        if model.get("is_mock"):
            if not row.get("smoke_test_passed") or row.get("smoke_test_kind") != "mock":
                raise ValueError("demo mock必须先有成功的mock smoke证据")
        else:
            license_ok = bool(
                model.get("license_approved")
                or model.get("license_review_approved")
                or model.get("license_status") in {"approved", "verified", "valid", "ok"}
            )
            if not license_ok:
                raise ValueError("真实workflow必须有许可审查证据")
            if not (
                row.get("smoke_test_passed") is True
                and row.get("smoke_test_kind") == "real"
                and row.get("review_approved") is True
            ):
                raise ValueError("真实workflow必须有审查批准和真实smoke证据")
            smoke_job_id = row.get("smoke_job_id")
            smoke_job = self.repo.get("jobs", smoke_job_id, owner_id) if smoke_job_id else None
            if not isinstance(smoke_job, dict):
                raise ValueError("真实workflow必须有当前匹配的真实smoke任务")
            if (
                smoke_job.get("state") != "completed"
                or smoke_job.get("provider") not in {"colab", "kaggle"}
                or smoke_job.get("smoke_accepted") is not True
                or smoke_job.get("acceptance_state") != "accepted"
                or smoke_job.get("workflow_smoke") is not True
            ):
                raise ValueError("真实workflow必须有已完成且已验收的真实smoke任务")
            # Import locally: review_routes imports application wiring that
            # eventually reaches Registry, so a module-level import cycles.
            from .. import review_routes

            if not review_routes._job_matches_any_smoke(smoke_job, id, row, model):
                raise ValueError("真实smoke证据与当前workflow artifact不匹配")
            classes = {
                node.get("class_type") for node in row.get("api_graph", {}).values() if isinstance(node, dict)
            }
            if not classes.issubset(_PUBLISH_NODE_ALLOWLIST):
                raise ValueError("真实workflow包含未通过固定allowlist审查的节点")
        for sibling in self.workflows(owner_id):
            if sibling.get("workflow_key") == row.get("workflow_key") and sibling.get("active"):
                self.repo.update("workflows", sibling["id"], {"active": False}, owner_id)
        return self.repo.update("workflows", id, {"state": "published", "active": True}, owner_id)

    def rollback(self, owner_id, key, version):
        target = self.get_workflow(owner_id, key, version)
        if target.get("state") != "published":
            raise ValueError("只能回滚到已发布版本")
        for sibling in self.workflows(owner_id):
            if sibling.get("workflow_key") == key and sibling.get("active"):
                self.repo.update("workflows", sibling["id"], {"active": False}, owner_id)
        return self.repo.update("workflows", target["id"], {"active": True}, owner_id)

    def deactivate(self, owner_id, workflow_id):
        """Stop using a published version as the default. In-flight jobs stay untouched."""
        row = self.repo.get("workflows", workflow_id, owner_id)
        if row is None:
            raise KeyError(workflow_id)
        if row.get("state") != "published":
            raise ValueError("只能停用已发布的工作流版本")
        if not row.get("active"):
            return row
        return self.repo.update("workflows", workflow_id, {"active": False}, owner_id)
