"""Compute contracts and deterministic local doubles for GPU Studio."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class ComputeError(RuntimeError):
    """Safe, user-facing compute failure."""


class ProviderUnavailable(ComputeError):
    """The provider is disabled or not configured."""


class ComputeProvider(Protocol):
    def list_specs(self) -> list[dict]: ...
    def create(self, spec: dict, request_id: str, job_id: str, runtime_id: str) -> dict: ...
    def inspect(self, handle: dict) -> dict: ...
    def delete(self, handle: dict) -> bool: ...


class GPUExecutor(Protocol):
    def start(self, handle: dict, job: dict, execution_id: str) -> dict: ...
    def poll(self, handle: dict, job: dict) -> dict: ...


@dataclass(frozen=True)
class GpuConfig:
    enabled: bool = True
    real_enabled: bool = False
    max_runtime_minutes: int = 30
    daily_units: float = 10.0
    monthly_units: float = 100.0
    unit_rate: float | None = None
    max_job_units: float = 5.0
    create_retries: int = 2
    create_timeout_seconds: float = 30.0
    cleanup_retry_base_seconds: float = 2.0
    cleanup_retry_max_seconds: float = 300.0
    claim_ttl_seconds: float = 120.0
    spec_allowlist: tuple[str, ...] = ()
    # Trusted executor plugins are administrator-only and fail closed unless
    # explicitly enabled with an exact module:entrypoint@version allowlist.
    trusted_executor_enabled: bool = False
    trusted_executor_type: str = "trusted"
    trusted_executor_module: str = ""
    trusted_executor_entrypoint: str = ""
    trusted_executor_version: str = ""
    trusted_executor_allowlist: tuple[str, ...] = ()
    trusted_executor_cache_namespace: str = "trusted"

    @staticmethod
    def _bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _number(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None or not value.strip():
            return default
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric") from exc

    @classmethod
    def from_env(cls) -> "GpuConfig":
        max_runtime = int(cls._number("GPU_MAX_RUNTIME_MINUTES", 30))
        retries = int(cls._number("GPU_CREATE_RETRIES", 2))
        if max_runtime <= 0 or retries < 0:
            raise ValueError("GPU runtime must be positive and retries non-negative")
        rate_text = os.getenv("GPU_UNIT_RATE", "").strip()
        allowlist = tuple(
            item.strip() for item in os.getenv("GPU_SPEC_ALLOWLIST", "").split(",") if item.strip()
        )
        trusted_allowlist = tuple(
            item.strip()
            for item in os.getenv("GPU_TRUSTED_EXECUTOR_ALLOWLIST", "").split(",")
            if item.strip()
        )
        return cls(
            enabled=cls._bool("GPU_ENABLED", True),
            real_enabled=False if os.getenv("STUDIO_OFFLINE_TEST") == "1" else cls._bool("GPU_REAL_ENABLED", False),
            max_runtime_minutes=max_runtime,
            daily_units=cls._number("GPU_DAILY_UNITS", 10),
            monthly_units=cls._number("GPU_MONTHLY_UNITS", 100),
            unit_rate=None if not rate_text else float(rate_text),
            max_job_units=cls._number("GPU_MAX_JOB_UNITS", 5),
            create_retries=retries,
            create_timeout_seconds=cls._number("GPU_CREATE_TIMEOUT_SECONDS", 30),
            cleanup_retry_base_seconds=max(0.0, cls._number("GPU_CLEANUP_RETRY_BASE_SECONDS", 2)),
            cleanup_retry_max_seconds=max(0.0, cls._number("GPU_CLEANUP_RETRY_MAX_SECONDS", 300)),
            claim_ttl_seconds=max(1.0, cls._number("GPU_CLAIM_TTL_SECONDS", 120)),
            spec_allowlist=allowlist,
            trusted_executor_enabled=cls._bool("GPU_TRUSTED_EXECUTOR_ENABLED", False),
            trusted_executor_type=os.getenv("GPU_TRUSTED_EXECUTOR_TYPE", "trusted").strip(),
            trusted_executor_module=os.getenv("GPU_TRUSTED_EXECUTOR_MODULE", "").strip(),
            trusted_executor_entrypoint=os.getenv("GPU_TRUSTED_EXECUTOR_ENTRYPOINT", "").strip(),
            trusted_executor_version=os.getenv("GPU_TRUSTED_EXECUTOR_VERSION", "").strip(),
            trusted_executor_allowlist=trusted_allowlist,
            trusted_executor_cache_namespace=os.getenv("GPU_TRUSTED_EXECUTOR_CACHE_NAMESPACE", "trusted").strip(),
        )


class MockColabProvider:
    """Explicit fake provider. It never makes network calls."""

    is_mock = True
    _runtimes: dict[str, dict] = {}
    _deleted: set[str] = set()
    _tombstones: dict[str,dict] = {}

    def __init__(
        self,
        *,
        fail_create_times: int = 0,
        fail_delete_times: int = 0,
        create_error: Exception | None = None,
        faults: dict | None = None,
    ):
        faults = faults or {}
        self.fail_create_times = int(faults.get("create", fail_create_times))
        self.fail_delete_times = int(faults.get("delete", fail_delete_times))
        self.create_error = create_error or ComputeError("mock provider create failure")
        self.create_calls = self.inspect_calls = self.delete_calls = 0
        # These registries intentionally live at module/class scope: the
        # default factory may construct a fresh adapter on every short tick.
        # A handle, not an adapter instance, is the source of truth for mock
        # recovery just as a provider API would be after process restart.
        self.deleted = []
        self.handles = self._runtimes

    def list_specs(self) -> list[dict]:
        return [{"name": "mock-colab", "provider": "mock", "simulated": True}]

    def create(self, spec: dict, request_id: str, job_id: str, runtime_id: str) -> dict:
        self.create_calls += 1
        if self.create_calls <= self.fail_create_times:
            raise self.create_error
        name = f"mock-runtime-{runtime_id}"
        handle = {
            "runtime_name": name,
            "operation_name": f"mock-op-{request_id}",
            "ready": True,
            "deleted": False,
            "mock": True,
            "runtime_id": runtime_id,
            "request_id": request_id,
            "job_id": job_id,
        }
        self._runtimes[name] = handle
        self._deleted.discard(name)
        return dict(handle)

    def _verify_handle(self, handle):
        name=handle.get('runtime_name','')
        record=self._runtimes.get(name) or self._tombstones.get(name)
        if not record or not handle.get('mock') or any(handle.get(key)!=record.get(key) for key in ('runtime_id','request_id','owner_id')):
            raise ComputeError('mock runtime handle identity mismatch')
        return record

    def recover_handle(self, handle):
        record=self._runtimes.get(handle.get('runtime_name')) or self._tombstones.get(handle.get('runtime_name'))
        if record is None and handle.get('mock') is True and handle.get('owner_id') and handle.get('request_id') and handle.get('runtime_id') and handle.get('runtime_name') == f"mock-runtime-{handle['runtime_id']}":
            # The lease is persisted; the simulated provider registry is not.
            record = dict(handle)
            self._runtimes[handle['runtime_name']] = record
        if not record or any(record.get(key)!=handle.get(key) for key in ('runtime_id','request_id')):
            raise ComputeError('mock recovery identity mismatch')
        if record.get('owner_id') not in {None,handle.get('owner_id')}:
            raise ComputeError('mock recovery owner identity mismatch')
        record['owner_id']=handle.get('owner_id')
        recovered=dict(record)
        for key in ('job_id','owner_job_id'):  # job ownership comes from the lease-owned handle, not the create-time record
            if key in handle:
                recovered[key]=handle[key]
        return recovered

    def bind_owner(self, handle, owner_id):
        record=self._verify_handle(handle)
        if record.get('owner_id') not in {None,owner_id}:
            raise ComputeError('mock runtime owner identity mismatch')
        record['owner_id']=owner_id
        return dict(record)

    def inspect(self, handle: dict) -> dict:
        self.inspect_calls += 1
        self._verify_handle(handle)
        name = handle.get("runtime_name")
        if name in self._deleted or name not in self._runtimes:
            return {"ready": False, "deleted": True, "runtime_name": name, "mock": True}
        return {**self._runtimes[name], "ready": True, "deleted": False}

    def delete(self, handle: dict) -> bool:
        self._verify_handle(handle)
        self.delete_calls += 1
        if self.delete_calls <= self.fail_delete_times:
            return False
        name = handle.get("runtime_name", "")
        if not handle.get("mock") or not name.startswith("mock-runtime-"):
            raise ComputeError("refusing to delete an unverified runtime")
        if name not in self._deleted:
            self._deleted.add(name)
            self.deleted.append(name)
        record=self._runtimes.pop(name, None)
        if record:
            self._tombstones[name]=record
        from .mock_sessions import MockWorkerSessions
        MockWorkerSessions.close(name)
        return True


class MockGPUExecutor:
    """Deterministic executor; execution IDs make start idempotent across restart."""

    is_mock = True
    _executions: dict[str, dict] = {}

    def __init__(
        self,
        *,
        polls_before_complete: int = 0,
        fail_on_start: bool = False,
        fail_on_poll: bool = False,
        output: bytes | None = None,
        faults: dict | None = None,
    ):
        faults = faults or {}
        self.polls_before_complete = int(faults.get("polls_before_complete", polls_before_complete))
        self.fail_on_start = bool(faults.get("start", fail_on_start))
        self.fail_on_poll = bool(faults.get("poll", fail_on_poll))
        # Valid 1x1 PNG so the independent asset validator accepts the mock
        # artifact; the metadata and filename also visibly identify MOCK.
        self.output = output or __import__("base64").b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        self.started: list[str] = []
        self.start_calls = self.poll_calls = 0
        self.outputs: list[dict] = []
        self.max_concurrent = 0
        self._active = 0

    def start(self, handle: dict, job: dict, execution_id: str) -> dict:
        self.start_calls += 1
        if execution_id in self._executions:
            previous=self._executions[execution_id]
            if previous.get('owner_id')!=job.get('owner_id') or previous.get('runtime')!=handle.get('runtime_name'):
                raise ComputeError('mock execution owner or runtime mismatch')
            if previous.get('signature')!=job.get('compatibility_signature'):
                raise ComputeError('mock execution compatibility signature mismatch')
            return {"execution_id": execution_id, "idempotent": True}
        if self.fail_on_start:
            raise ComputeError("mock executor start failure")
        from .mock_sessions import MockWorkerSessions
        session = MockWorkerSessions.start(handle.get('runtime_name'), job.get('owner_id'),
                                           job.get('compatibility_signature'), execution_id)
        self._executions[execution_id] = {
            "signature": job.get('compatibility_signature'),
            "worker_session_id": session['id'],
            "restore_skipped": bool(session['restore_skipped'].get(execution_id)),
            "runtime": handle.get('runtime_name'),
            "owner_id": job.get('owner_id'),
            "remaining": self.polls_before_complete,
            "failed": False,
            "output": self.output,
        }
        self.started.append(execution_id)
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        return {"execution_id": execution_id, "mock": True}

    def poll(self, handle: dict, job: dict) -> dict:
        self.poll_calls += 1
        state = self._executions.get(job.get("execution_id"))
        if state is None:
            raise ComputeError("mock execution is not recoverable")
        if state.get('owner_id') != job.get('owner_id') or state.get('runtime') != handle.get('runtime_name'):
            raise ComputeError('mock execution owner or runtime mismatch')
        from .mock_sessions import MockWorkerSessions
        MockWorkerSessions.validate_poll(state['runtime'],state['owner_id'],job['execution_id'])
        if self.fail_on_poll:
            state["failed"] = True
        if state["failed"]:
            from .mock_sessions import MockWorkerSessions
            MockWorkerSessions.finish(state['runtime'],state['owner_id'],job['execution_id'])
            self._active = max(0, self._active - 1)
            return {
                "state": "failed",
                "outputs": [],
                "error_code": "gpu_execution_failed",
                "message": "mock executor failure",
            }
        if state["remaining"] > 0:
            state["remaining"] -= 1
            return {"state": "running", "outputs": []}
        from .mock_sessions import MockWorkerSessions
        MockWorkerSessions.finish(state['runtime'],state['owner_id'],job['execution_id'])
        self._active = max(0, self._active - 1)
        output = {
            "filename": "mock-output.png",
            "content": state["output"],
            "mime": "image/png",
            "metadata": {"mock": True, "label": "MOCK"},
        }
        self.outputs.append(output)
        timings = {"warm_reuse": 0.0} if state.get("restore_skipped") else {"restore_weights": 0.0}
        return {"state": "completed", "outputs": [output], "timings": timings}


class DisabledProvider:
    is_mock = False

    def __init__(self, reason: str = "real GPU provider is disabled"):
        self.reason = reason

    def list_specs(self) -> list[dict]:
        return []

    def create(self, spec: dict, request_id: str, job_id: str, runtime_id: str) -> dict:
        raise ProviderUnavailable(self.reason)

    def inspect(self, handle: dict) -> dict:
        raise ProviderUnavailable(self.reason)

    def delete(self, handle: dict) -> bool:
        raise ProviderUnavailable(self.reason)


class DisabledExecutor:
    def start(self, handle: dict, job: dict, execution_id: str) -> dict:
        raise ProviderUnavailable("real GPU executor is disabled")

    def poll(self, handle: dict, job: dict) -> dict:
        raise ProviderUnavailable("real GPU executor is disabled")


def default_provider_factory(
    job: dict, config: GpuConfig | None = None, *, store=None, owner_id: str | None = None
):
    config = config or GpuConfig.from_env()
    name = job.get("provider") or (job.get("execution_policy") or {}).get("provider", "mock")
    if name == "mock":
        return MockColabProvider()
    if name == "kaggle":
        if not config.real_enabled or os.getenv("STUDIO_OFFLINE_TEST") == "1":
            return DisabledProvider("真实 GPU 未启用")
        from .kaggle import KaggleProvider
        return KaggleProvider(store, owner_id or job.get("owner_id"))
    if name == "colab" and (os.getenv("STUDIO_OFFLINE_TEST") == "1" or not config.real_enabled):
        return DisabledProvider("real GPU provider is disabled or unconfigured")
    if name == "colab":
        try:
            from .colab import GoogleColabProvider
        except (ImportError, ModuleNotFoundError):
            return DisabledProvider("real Colab provider is unavailable")
        if hasattr(GoogleColabProvider, "from_environment"):
            return GoogleColabProvider.from_environment(
                store=store,
                owner_id=owner_id or job.get("owner_id"),
                account_id=job.get("account_id"),
            )
        return GoogleColabProvider(
            store,
            owner_id or job.get("owner_id"),
            account_id=job.get("account_id"),
        )
    return DisabledProvider(f"unknown GPU provider: {name}")


def _is_builtin_mcp_job(job: dict) -> bool:
    """System CUDA smoke/prep jobs are never routed through trusted plugins."""
    record = job if isinstance(job, dict) else {}
    snapshot = record.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    manifest = snapshot.get("manifest") if isinstance(snapshot.get("manifest"), dict) else {}
    values = {
        record.get("kind"),
        record.get("model_id"),
        snapshot.get("kind"),
        snapshot.get("model_id"),
        manifest.get("kind"),
        manifest.get("executor"),
    }
    if any(
        isinstance(value, str)
        and value in {"system", "prep", "system-prep", "cuda-prep", "system-cuda-smoke", "system-cuda-prep"}
        for value in values
    ):
        return True
    return any(snapshot.get(flag) is True or record.get(flag) is True for flag in ("system_smoke", "prep"))


def _mcp_executor(job: dict, *, store=None, owner_id: str | None = None):
    try:
        from .mcp import MCPGPUExecutor
    except (ImportError, ModuleNotFoundError):
        return DisabledExecutor()
    return MCPGPUExecutor.from_environment(
        store=store,
        owner_id=owner_id or job.get("owner_id"),
        account_id=job.get("account_id"),
    )


def default_executor_factory(
    job: dict, config: GpuConfig | None = None, *, store=None, owner_id: str | None = None
):
    config = config or GpuConfig.from_env()
    name = job.get("provider") or (job.get("execution_policy") or {}).get("provider", "mock")
    if name == "mock":
        return MockGPUExecutor()
    if name == "kaggle" and config.real_enabled and os.getenv("STUDIO_OFFLINE_TEST") != "1":
        from .kaggle import KaggleExecutor
        return KaggleExecutor(store, owner_id or job.get("owner_id"))
    if name == "colab" and config.real_enabled and os.getenv("STUDIO_OFFLINE_TEST") != "1":
        # Built-in diagnostics/prep retain the legacy MCP path forever.  They
        # intentionally have no model revision and therefore cannot satisfy
        # the trusted-plugin cache contract.
        if _is_builtin_mcp_job(job):
            return _mcp_executor(job, store=store, owner_id=owner_id)

        from .newgpu.executor_plugins import load_trusted_executor, select_registered_executor_type

        # Existing Comfy jobs preserve MCP; plugins require explicit selection.
        executor_type = select_registered_executor_type(config, job)
        if executor_type is None:
            return _mcp_executor(job, store=store, owner_id=owner_id)
        if not config.trusted_executor_enabled:
            return DisabledExecutor()
        return load_trusted_executor(config)
    return DisabledExecutor()


__all__ = [
    "ComputeError",
    "ProviderUnavailable",
    "ComputeProvider",
    "GPUExecutor",
    "GpuConfig",
    "MockColabProvider",
    "MockGPUExecutor",
    "default_provider_factory",
    "default_executor_factory",
]
