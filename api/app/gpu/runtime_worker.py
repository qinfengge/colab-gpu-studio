"""Trusted Colab worker installed by :mod:`app.gpu.mcp`.

This file is intentionally boring: it is the only code allowed to run inside a
remote Colab runtime.  The worker accepts a JSON snapshot, never a command or
Python string from the caller, and records every outcome in status.json.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

MAX_OUTPUT_BYTES = 100 * 1024 * 1024
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3"
DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{10,200}$")
DRIVE_TOKEN = re.compile(r"^[A-Za-z0-9._~+/=-]{20,4096}$")
FOLDER_MIME = "application/vnd.google-apps.folder"
UPLOAD_CHUNK = 64 * 1024 * 1024  # must be a multiple of 256 KiB
# Third-party ComfyUI nodes that were read line by line and pinned. Nothing else may be installed;
# no pip step runs for them (their only dependency, einops, is already in ComfyUI's requirements).
REVIEWED_CUSTOM_NODES = {
    ("https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus", "1ddcc33e9b674aaabbccc3808fe36e1a47bdd36d"):
        "Comfyui_Minimax_h3_latent_Upscaler",
    # Read-only reviewed fork: pure torch node, no install-time dependency beyond ComfyUI.
    ("https://github.com/huwhitememes/comfyui-krea2-conditioning", "729cda4fade982988a375b01928f515458407a5c"):
        "comfyui-krea2-conditioning",
}
REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# Retained Comfy services deliberately have no payload-controlled filesystem or
# network address.  The root is fixed in the worker image; tests may replace the
# module constant with an isolated directory, but callers cannot do so.
COMFY_SERVICE_ROOT = Path("/content/ai-studio/comfy-services")
COMFY_SERVICE_STATE = "state.json"
COMFY_SERVICE_LOCK = "manager.lock"
COMFY_JOB_LOCK = "job.lock"
COMFY_SERVICE_ACTIONS = frozenset({"health", "read_safe_log", "stop_current_job"})
COMFY_SERVICE_PORT = 8188
COMFY_SERVICE_HOST = "127.0.0.1"
MAX_SAFE_LOG_BYTES = 8 * 1024
_SERVICE_HANDLES: dict[int, Any] = {}
_SERVICE_LOG_HANDLES: dict[int, Any] = {}


class WorkerError(RuntimeError):
    pass


class _StopRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_NO_REDIRECT = build_opener(_StopRedirect)


def _safe_error(exc: BaseException) -> str:
    value = str(exc).replace("Authorization", "[redacted]").replace("Bearer", "[redacted]")
    value = re.sub(r"(?i)(token|secret|password|api[_-]?key)[=: ]+\S+", r"\1=[redacted]", value)
    return value[:1400] or exc.__class__.__name__


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _status(root: Path, state: str, **fields: Any) -> None:
    _atomic_json(root / "status.json", {"state": state, "updated_at": time.time(), **fields})


def _instance_service_name(instance_id: str) -> str:
    """Return a non-path-derived directory name for a validated UUID."""
    try:
        parsed = uuid.UUID(instance_id)
    except (AttributeError, ValueError, TypeError) as exc:
        raise WorkerError("instance_id must be a UUID") from exc
    if str(parsed) != str(instance_id).lower():
        raise WorkerError("instance_id must use canonical UUID form")
    return hashlib.sha256(("comfy-service:" + parsed.hex).encode("ascii")).hexdigest()[:32]


def _comfy_service_dir(instance_id: str) -> Path:
    """Resolve the only directory a retained service may use."""
    name = _instance_service_name(instance_id)
    root = COMFY_SERVICE_ROOT.expanduser().resolve()
    if COMFY_SERVICE_ROOT.exists() and COMFY_SERVICE_ROOT.is_symlink():
        raise WorkerError("Comfy service root must not be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    service = root / name
    if service.exists() and service.is_symlink():
        raise WorkerError("Comfy service directory must not be a symlink")
    resolved = service.resolve()
    if root not in resolved.parents:
        raise WorkerError("Comfy service directory escaped fixed root")
    return resolved


@contextlib.contextmanager
def _file_lock(path: Path):
    """Serialize worker-local service management without trusting a PID lock."""
    import fcntl  # Linux-only remote service lock; host imports also work on Windows.

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _service_fingerprint(
    commit: str, custom_nodes: list[dict], weights: list[dict], manifest: dict | None = None
) -> str:
    """Fingerprint the inputs that make a resident process safe to reuse.

    A declared dependency signature is authoritative when present.  For old
    manifests, the installed package set remains a deterministic fallback; it
    is intentionally separate from the warm-disk fingerprint below.
    """
    manifest = manifest or {}
    declared = manifest.get("dependency_signature", manifest.get("dependencies_signature"))
    if declared is None:
        declared = manifest.get("dependencies", manifest.get("requirements_sha256"))
    if declared is None:
        packages = sorted(f"{d.metadata.get('Name', '').lower()}=={d.version}" for d in importlib.metadata.distributions())
        declared = hashlib.sha256("\n".join(packages).encode()).hexdigest()
    body = {
        "version": 1,
        "comfy_commit": commit,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "custom_nodes": sorted(f"{n.get('repo')}@{n.get('commit')}" for n in custom_nodes),
        "weights": sorted(
            (w.get("target"), w.get("sha256"), w.get("size"), w.get("source", "huggingface"))
            for w in weights
        ),
        "dependencies": declared,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _process_metadata(pid: int) -> dict:
    """Read identity data from Linux procfs; failure means ownership is unknown."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise WorkerError("Comfy process ownership is unknown")
    proc = Path("/proc") / str(pid)
    try:
        raw_stat = (proc / "stat").read_text(encoding="utf-8")
        after_comm = raw_stat.rsplit(")", 1)[1].split()
        start_id = after_comm[19]
        raw_cmdline = (proc / "cmdline").read_bytes()
        cmdline = [item.decode("utf-8", "replace") for item in raw_cmdline.split(b"\0") if item]
        cwd = os.readlink(proc / "cwd")
    except (OSError, IndexError, UnicodeError) as exc:
        raise WorkerError("Comfy process ownership is unknown") from exc
    return {"start_id": start_id, "cmdline": cmdline, "cwd": cwd}


def _trusted_process_metadata(service_dir: Path, state: dict) -> dict:
    """Verify PID, start time, command and cwd before any reuse or signal."""
    pid = state.get("pid")
    expected_start = state.get("process_start_id", state.get("start_id"))
    if not isinstance(expected_start, str) or not expected_start:
        raise WorkerError("Comfy process ownership is unknown")
    metadata = _process_metadata(pid)
    if metadata.get("start_id") != expected_start:
        raise WorkerError("Comfy process ownership check failed")
    expected_cwd = (service_dir / "ComfyUI").resolve()
    actual_cwd = Path(str(metadata.get("cwd", ""))).resolve()
    command = metadata.get("cmdline")
    if not isinstance(command, list) or len(command) != 6:
        raise WorkerError("Comfy process ownership check failed")
    if Path(command[0]).name != Path(sys.executable).name or command[1] != "main.py":
        raise WorkerError("Comfy process ownership check failed")
    if command[2:] != ["--listen", COMFY_SERVICE_HOST, "--port", str(COMFY_SERVICE_PORT)]:
        raise WorkerError("Comfy process ownership check failed")
    if actual_cwd != expected_cwd:
        raise WorkerError("Comfy process ownership check failed")
    return metadata


def _read_service_state(service_dir: Path) -> dict | None:
    try:
        value = json.loads((service_dir / COMFY_SERVICE_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _service_api_url(path: str) -> str:
    if path not in {"/system_stats", "/interrupt"}:
        raise WorkerError("Comfy service endpoint is not allowed")
    return f"http://{COMFY_SERVICE_HOST}:{COMFY_SERVICE_PORT}{path}"


def _stop_service_if_owned(service_dir: Path) -> bool:
    """Stop only the process whose complete identity matches its state file."""
    state = _read_service_state(service_dir)
    if not state:
        return False
    _trusted_process_metadata(service_dir, state)
    pid = state["pid"]
    handle = _SERVICE_HANDLES.pop(pid, None)
    log_handle = _SERVICE_LOG_HANDLES.pop(pid, None)
    try:
        if handle is not None:
            if handle.poll() is None:
                handle.terminate()
                try:
                    handle.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    handle.kill()
            return True
        os.killpg(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    finally:
        if log_handle is not None:
            log_handle.close()
        (service_dir / COMFY_SERVICE_STATE).unlink(missing_ok=True)


def _start_comfy_service(service_dir: Path, fingerprint: str, cache_reason: str) -> dict:
    comfy = service_dir / "ComfyUI"
    if not comfy.is_dir() or comfy.is_symlink():
        raise WorkerError("Comfy service checkout is unavailable")
    log_path = service_dir / "comfy.log"
    log_handle = log_path.open("ab", buffering=0)
    args = [sys.executable, "main.py", "--listen", COMFY_SERVICE_HOST, "--port", str(COMFY_SERVICE_PORT)]
    try:
        process = subprocess.Popen(
            args,
            cwd=comfy,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
        )
        metadata = _process_metadata(process.pid)
        state = {
            "pid": process.pid,
            "process_start_id": metadata["start_id"],
            "start_id": metadata["start_id"],
            "env_fingerprint": fingerprint,
            "cache_reason": cache_reason,
            "cwd": str(comfy.resolve()),
            "argv": args,
            "host": COMFY_SERVICE_HOST,
            "port": COMFY_SERVICE_PORT,
            "updated_at": time.time(),
        }
        _SERVICE_HANDLES[process.pid] = process
        _SERVICE_LOG_HANDLES[process.pid] = log_handle
        _atomic_json(service_dir / COMFY_SERVICE_STATE, state)
        _http_json(_service_api_url("/system_stats"), timeout=5)
        return {**state, "system_stats": True, "service_dir": str(service_dir)}
    except Exception:
        try:
            if "process" in locals() and process.poll() is None:
                process.terminate()
        finally:
            log_handle.close()
        raise


def _ensure_comfy_service(
    instance_id: str,
    commit: str,
    custom_nodes: list[dict],
    weights: list[dict],
    *,
    manifest: dict | None = None,
    expected_fingerprint: str | None = None,
) -> dict:
    """Return a real, healthy resident service or fail closed."""
    service_dir = _comfy_service_dir(instance_id)
    fingerprint = expected_fingerprint or _service_fingerprint(commit, custom_nodes, weights, manifest)
    state = _read_service_state(service_dir)
    restart_reason = None
    if state is not None:
        _trusted_process_metadata(service_dir, state)
        if state.get("env_fingerprint") == fingerprint:
            try:
                stats = _http_json(_service_api_url("/system_stats"), timeout=5)
            except WorkerError:
                restart_reason = "restart_health_failed"
            else:
                state.update(cache_reason="hot_reuse", updated_at=time.time())
                _atomic_json(service_dir / COMFY_SERVICE_STATE, state)
                return {**state, "system_stats": stats, "service_dir": str(service_dir)}
        else:
            restart_reason = "restart_fingerprint_mismatch"
        _stop_service_if_owned(service_dir)
    reason = restart_reason or "cold_start"
    return _start_comfy_service(service_dir, fingerprint, reason)


def _validate_workbench_request(request: dict) -> dict:
    if not isinstance(request, dict):
        raise WorkerError("workbench action must be a JSON object")
    action = request.get("action")
    if action not in COMFY_SERVICE_ACTIONS:
        raise WorkerError("workbench action is not allowed")
    allowed = {"action", "instance_id"} | ({"max_bytes"} if action == "read_safe_log" else set())
    if set(request) - allowed:
        raise WorkerError("workbench action contains unsupported fields")
    _instance_service_name(request.get("instance_id"))
    if action == "read_safe_log":
        limit = request.get("max_bytes", MAX_SAFE_LOG_BYTES)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SAFE_LOG_BYTES:
            raise WorkerError("max_bytes is outside the safe bound")
    return request


def _safe_log(service_dir: Path, limit: int) -> str:
    try:
        raw = (service_dir / "comfy.log").read_bytes()[-limit:]
    except OSError as exc:
        raise WorkerError("safe log is unavailable") from exc
    text = raw.decode("utf-8", "replace")
    text = re.sub(r"(?i)authorization\s*:\s*bearer\s+\S+", "Authorization: Bearer [redacted]", text)
    text = re.sub(r"(?i)\b(token|secret|password|api[_-]?key)\s*[=:]\s*\S+", r"\1=[redacted]", text)
    return text[-limit:]


def _handle_workbench_action(request: dict) -> dict:
    request = _validate_workbench_request(request)
    service_dir = _comfy_service_dir(request["instance_id"])
    state = _read_service_state(service_dir)
    try:
        if state is None:
            raise WorkerError("service state is unavailable")
        _trusted_process_metadata(service_dir, state)
        if request["action"] == "health":
            stats = _http_json(_service_api_url("/system_stats"), timeout=5)
            return {
                "state": "healthy",
                "cache_reason": state.get("cache_reason", "unknown"),
                "pid": state["pid"],
                "process_start_id": state.get("process_start_id"),
                "system_stats": stats,
            }
        if request["action"] == "read_safe_log":
            return {
                "state": "ok",
                "cache_reason": state.get("cache_reason", "unknown"),
                "log": _safe_log(service_dir, request.get("max_bytes", MAX_SAFE_LOG_BYTES)),
            }
        _http_json(_service_api_url("/interrupt"), "POST", {}, timeout=5)
        return {"state": "interrupt_requested", "cache_reason": state.get("cache_reason", "unknown")}
    except Exception as exc:  # safe protocol never reports an unprobed healthy state
        return {"state": "unknown", "reason": _safe_error(exc)[:200]}


def _main_workbench_action(path: str) -> int:
    action_path = Path(path)
    if action_path.is_symlink() or not action_path.is_file():
        return 2
    try:
        request = json.loads(action_path.read_text(encoding="utf-8"))
        result = _handle_workbench_action(request)
    except Exception as exc:  # malformed/unsupported requests fail closed
        result = {"state": "unknown", "reason": _safe_error(exc)[:200]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _run(
    args: list[str], *, cwd: Path | None = None, timeout: float, capture: bool = True
) -> subprocess.CompletedProcess:
    if not isinstance(args, list) or not args or any(not isinstance(item, str) for item in args):
        raise WorkerError("internal command must be an argument list")
    try:
        return subprocess.run(
            args, cwd=cwd, timeout=timeout, check=True, shell=False, capture_output=capture, text=True
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkerError("controlled command timed out") from exc
    except subprocess.CalledProcessError as exc:
        tail = ((exc.stderr or "") + (exc.stdout or "")).strip()[-600:]
        raise WorkerError(
            f"controlled command failed ({Path(args[0]).name} {' '.join(args[1:3])}): {tail}"
        ) from exc
    except OSError as exc:
        raise WorkerError("controlled command failed") from exc


def _snapshot(payload: dict) -> tuple[dict, dict, dict]:
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        raise WorkerError("missing immutable snapshot")
    manifest = snapshot.get("manifest")
    graph = snapshot.get("bound_graph")
    if not isinstance(manifest, dict):
        raise WorkerError("missing workflow manifest")
    if manifest.get("state") != "published" or manifest.get("reviewed") is not True:
        raise WorkerError("workflow is not reviewed and published")
    _reviewed_custom_nodes(manifest.get("custom_nodes"))
    if snapshot.get("custom_nodes"):
        raise WorkerError("custom_nodes belong in the reviewed manifest")
    if not isinstance(graph, dict):
        graph = {}
    return snapshot, manifest, graph


def _reviewed_custom_nodes(items: Any) -> list[dict]:
    """Each entry must be exactly a pinned (repo, commit) from REVIEWED_CUSTOM_NODES."""
    items = items or []
    if not isinstance(items, list):
        raise WorkerError("custom_nodes must be a list")
    for item in items:
        if not isinstance(item, dict) or REVIEWED_CUSTOM_NODES.get((item.get("repo"), item.get("commit"))) != item.get("dir"):
            raise WorkerError("custom node is not a reviewed, pinned repository")
    return items


def _install_custom_nodes(comfy: Path, items: list[dict], timeout: float) -> None:
    for item in items:
        folder = comfy / "custom_nodes" / item["dir"]
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(mode=0o700, parents=True)
        _run(["git", "init", "-q"], cwd=folder, timeout=30)
        _run(["git", "fetch", "-q", "--depth", "1", item["repo"], item["commit"]], cwd=folder, timeout=timeout)
        _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=folder, timeout=timeout)
    _verify_custom_nodes(comfy, items)


def _verify_custom_nodes(comfy: Path, items: list[dict]) -> None:
    for item in items:
        head = _run(["git", "rev-parse", "HEAD"], cwd=comfy / "custom_nodes" / item["dir"], timeout=30).stdout.strip()
        if head.lower() != item["commit"].lower():
            raise WorkerError(f"custom node {item['dir']} is not the reviewed commit")


def _validate_weights(manifest: dict) -> list[dict]:
    weights = manifest.get("weight_files", manifest.get("weights"))
    if not isinstance(weights, list) or not weights:
        raise WorkerError("no reviewed model weights; refusing to run")
    checked = []
    for item in weights:
        if not isinstance(item, dict):
            raise WorkerError("weight manifest is invalid")
        source = item.get("source", "huggingface")
        if source not in {"huggingface", "drive"}:
            raise WorkerError("weight source is unknown")
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or not SHA256.fullmatch(sha256):
            raise WorkerError("weight sha256 is not locked")
        if source == "drive":
            drive_file_id = item.get("drive_file_id")
            if not isinstance(drive_file_id, str) or not DRIVE_ID.fullmatch(drive_file_id):
                raise WorkerError("Drive file id is invalid")
            size = item.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise WorkerError("Drive weight size must be a positive integer")
            target = item.get("target")
            if not isinstance(target, str) or not target or target.startswith("/") or ".." in Path(target).parts:
                raise WorkerError("weight target path traversal refused")
            checked.append({"source": "drive", "drive_file_id": drive_file_id, "sha256": sha256.lower(), "target": target, "size": size})
            continue
        repo_id, filename, revision = (item.get(key) for key in ("repo_id", "filename", "revision"))
        if not isinstance(repo_id, str) or not REPO_ID.fullmatch(repo_id):
            raise WorkerError("weight repo_id is not an allowed Hugging Face repository")
        if not isinstance(filename, str) or not filename or filename.startswith("/"):
            raise WorkerError("weight filename is invalid")
        path = Path(filename)
        if ".." in path.parts or path.is_absolute():
            raise WorkerError("weight filename path traversal refused")
        if not isinstance(revision, str) or not COMMIT.fullmatch(revision):
            raise WorkerError("weight revision must be a fixed commit")
        target = item.get("target", filename)
        if not isinstance(target, str) or not target or target.startswith("/") or ".." in Path(target).parts:
            raise WorkerError("weight target path traversal refused")
        size = item.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            raise WorkerError("weight size must be a positive integer")
        checked.append({"repo_id": repo_id, "filename": filename, "revision": revision, "sha256": sha256.lower(), "target": target, "size": size})
    return checked


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_cuda_smoke(root: Path) -> dict:
    """Run only real CUDA probes; a missing GPU is a hard failure."""
    try:
        nvidia = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            timeout=30,
        ).stdout.strip()
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        names = [str(torch.cuda.get_device_name(index)) for index in range(count)] if available else []
    except Exception as exc:
        raise WorkerError("CUDA smoke probe failed") from exc
    if not available or count < 1 or not nvidia:
        raise WorkerError("CUDA GPU is not available")
    payload = {
        "cuda_available": available,
        "device_count": count,
        "device_names": names,
        "nvidia_smi": nvidia[:4000],
    }
    content = json.dumps(payload, sort_keys=True).encode("utf-8")
    (root / "outputs").mkdir(mode=0o700, exist_ok=True)
    output = root / "outputs" / "cuda-smoke.json"
    output.write_bytes(content)
    return {
        "filename": output.name,
        "mime": "application/json",
        "path": str(output.relative_to(root)),
        "size": len(content),
        "metadata": {"cuda_smoke": True},
    }


class _Stages:
    """Wall-clock seconds per worker stage, mirrored into status.json so the server can show them."""

    def __init__(self, root: Path):
        self.root, self.timings, self._name, self._start = root, {}, None, 0.0

    def start(self, name: str) -> None:
        self.end()
        self._name, self._start = name, time.time()
        _status(self.root, "running", stage=name, timings=self.timings, stage_started=self._start)

    def end(self) -> None:
        if self._name:
            self.timings[self._name] = round(time.time() - self._start, 1)
            self._name = None


def _checkout_comfy(comfy: Path, commit: str, timeout: float) -> None:
    if not COMMIT.fullmatch(commit):
        raise WorkerError("ComfyUI commit must be a fixed 40-character commit")
    comfy.mkdir(mode=0o700, parents=True, exist_ok=True)
    _run(["git", "init", "-q"], cwd=comfy, timeout=30)
    _run(["git", "fetch", "-q", "--depth", "1", COMFY_REPO, commit], cwd=comfy, timeout=timeout)
    _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=comfy, timeout=timeout)


def _verify_checkout(comfy: Path, commit: str) -> None:
    current = _run(["git", "rev-parse", "HEAD"], cwd=comfy, timeout=30).stdout.strip()
    if current.lower() != commit.lower():
        raise WorkerError("ComfyUI checkout is not the reviewed commit")
    if not (comfy / "requirements.txt").is_file():
        raise WorkerError("reviewed ComfyUI checkout has no requirements.txt")


def _site_dir() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _site_state(site: Path) -> dict[str, int]:
    return {entry.name: entry.stat().st_mtime_ns for entry in site.iterdir()}


def _env_key(commit: str, timeout: float, custom_nodes: list[dict] | None = None) -> str:
    """Cache key: ComfyUI commit + Python + hash of the packages already on the image + requirements hash.

    Hashing the image's package list means the cache can only hit on a runtime whose torch/CUDA/etc.
    are the same as when the environment was built.
    """
    packages = sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in importlib.metadata.distributions())
    image = hashlib.sha256("\n".join(packages).encode()).hexdigest()[:12]
    url = COMFY_REPO.replace("https://github.com/", "https://raw.githubusercontent.com/") + f"/{commit}/requirements.txt"
    with urlopen(url, timeout=timeout) as response:
        requirements = hashlib.sha256(response.read(1024 * 1024)).hexdigest()[:12]
    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    key = f"comfy-{commit[:12]}-{py}-img{image}-req{requirements}"
    if custom_nodes:
        pins = "\n".join(sorted(f"{n['repo']}@{n['commit']}" for n in custom_nodes))
        key += "-nodes" + hashlib.sha256(pins.encode()).hexdigest()[:12]
    return key


def _pip_install(comfy: Path, timeout: float) -> None:
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    _run([*base, "-r", str(comfy / "requirements.txt")], cwd=comfy, timeout=timeout)
    _run([*base, "huggingface_hub"], cwd=comfy, timeout=timeout)


def _pack_env(comfy: Path, site: Path, before: dict[str, int], key: str, out: Path) -> dict:
    """Tar the ComfyUI checkout plus every site-packages entry the install added or changed."""
    after = _site_state(site)
    added = sorted(name for name, mtime in after.items() if before.get(name) != mtime)
    removed = sorted(name for name in before if name not in after)
    with tarfile.open(out, "w:gz", compresslevel=1) as tar:
        tar.add(comfy, arcname="ComfyUI")
        for name in added:
            tar.add(site / name, arcname=f"site/{name}")
        meta = json.dumps({"key": key, "removed": removed}).encode()
        info = tarfile.TarInfo("env.json")
        info.size = len(meta)
        tar.addfile(info, io.BytesIO(meta))
    return {"site_entries_added": len(added), "site_entries_removed": len(removed)}


def _unpack_env(archive: Path, root: Path, site: Path) -> None:
    stage = root / ".env-stage"
    shutil.rmtree(stage, ignore_errors=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(stage, filter="data")
    removed = json.loads((stage / "env.json").read_text())["removed"]
    for name in removed:
        _remove(site / name)
    for entry in (stage / "site").iterdir():
        _remove(site / entry.name)
        shutil.move(str(entry), str(site / entry.name))
    shutil.move(str(stage / "ComfyUI"), str(root / "ComfyUI"))
    shutil.rmtree(stage, ignore_errors=True)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


WARM_VERSION = 1


def _warm_fingerprint(commit: str, custom_nodes: list[dict], weights: list[dict]) -> str:
    """What a restored ComfyUI tree must match for the next job on this runtime to reuse it as-is."""
    body = {
        "v": WARM_VERSION,
        "commit": commit,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "nodes": sorted(f"{n['repo']}@{n['commit']}" for n in custom_nodes),
        "weights": sorted(
            (w["target"], w["sha256"], w.get("size"), w.get("source", "huggingface")) for w in weights
        ),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _warm_take(root: Path, fingerprint: str, custom_nodes: list[dict], weights: list[dict]) -> bool:
    """Move the previous job's restored ComfyUI tree into this job's root when it is the same environment.

    The marker is dropped while the tree is in use, so a job that dies mid-run can never leave a
    marker that points at a tree that has moved or half changed.
    """
    warm = root.parent / ".warm"
    marker = warm / "state.json"
    try:
        state = json.loads(marker.read_text("utf-8"))
    except (OSError, ValueError):
        return False
    tree = warm / "ComfyUI"
    if not isinstance(state, dict) or state.get("fingerprint") != fingerprint or not tree.is_dir():
        return False
    models = tree / "models"
    for item in weights:
        path = models / item["target"]
        if not path.is_file() or (item.get("size") and path.stat().st_size != item["size"]):
            return False
    for node in custom_nodes:
        if not (tree / "custom_nodes" / node["dir"]).is_dir():
            return False
    marker.unlink(missing_ok=True)
    shutil.rmtree(root / "ComfyUI", ignore_errors=True)
    tree.rename(root / "ComfyUI")
    # Per-job leftovers: reference images, previous outputs and the owner's own LoRAs.
    for leftover in ("input", "output", "temp", "models/loras/user"):
        shutil.rmtree(root / "ComfyUI" / leftover, ignore_errors=True)
    return True


def _warm_put(root: Path, fingerprint: str) -> None:
    """Park the restored tree where the next job on this runtime can pick it up."""
    tree = root / "ComfyUI"
    if not tree.is_dir():
        return
    warm = root.parent / ".warm"
    try:
        warm.mkdir(mode=0o700, exist_ok=True)
        (warm / "state.json").unlink(missing_ok=True)
        shutil.rmtree(warm / "ComfyUI", ignore_errors=True)
        tree.rename(warm / "ComfyUI")
        _atomic_json(warm / "state.json", {"fingerprint": fingerprint, "saved_at": time.time()})
    except OSError:  # the next job simply restores from the cache again
        shutil.rmtree(warm / "ComfyUI", ignore_errors=True)


def _restore_env(
    root: Path, commit: str, timeout: float, cache: "DriveCache | None", build: bool, custom_nodes: list[dict] | None = None
) -> dict:
    """ComfyUI + pip deps: from the Drive env cache when possible, otherwise installed fresh.

    ``build`` (CPU prep jobs only) packs a fresh install and uploads it; GPU jobs never upload,
    so the GPU is never kept alive for cache work.
    """
    comfy = root / "ComfyUI"
    site = _site_dir()
    started = time.time()
    record: dict[str, Any] = {}
    key = target = None
    if cache is not None:
        try:
            key = _env_key(commit, timeout, custom_nodes)
            target = f"env/{key}.tar.gz"
            record["key"] = key
            entry = cache.lookup(target, None, None)
            if entry:
                archive = root / "env.tar.gz"
                cache.fetch(entry, archive)
                if _hash_file(archive)[0] != entry["appProperties"]["sha256"]:
                    raise WorkerError("env cache sha256 mismatch")
                _unpack_env(archive, root, site)
                archive.unlink(missing_ok=True)
                _verify_checkout(comfy, commit)
                _verify_custom_nodes(comfy, custom_nodes or [])
                record.update(source="drive", seconds=round(time.time() - started, 1))
                return record
        except Exception as exc:  # noqa: BLE001 - a broken cache falls back to a normal install
            record["cache_error"] = _safe_error(exc)[:200]
            shutil.rmtree(comfy, ignore_errors=True)
    before = _site_state(site)
    _checkout_comfy(comfy, commit, timeout)
    _verify_checkout(comfy, commit)
    _pip_install(comfy, timeout)
    _install_custom_nodes(comfy, custom_nodes or [], timeout)
    record.update(source="install", seconds=round(time.time() - started, 1))
    if build and cache is not None and key:
        try:
            archive = root / "env.tar.gz"
            record.update(_pack_env(comfy, site, before, key, archive))
            sha256, md5 = _hash_file(archive)
            cache.store(target, archive, sha256, md5, archive.stat().st_size)
            record.update(cache_stored=True, cache_bytes=archive.stat().st_size)
            archive.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - the prep job still succeeded in installing
            record.update(cache_stored=False, cache_error=_safe_error(exc)[:200])
    return record


def _hash_file(path: Path) -> tuple[str, str]:
    """Return (sha256, md5) from one pass; md5 is only for comparing with Drive's checksum."""
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


class DriveCache:
    """Model cache in the user's Drive, reached with a short-lived drive.file token.

    drive.file only exposes files this app created, so the cache can never read
    anything else in the user's Drive.  A cached file is trusted when its stored
    sha256/size marker matches the reviewed manifest and the md5 computed while
    downloading matches the checksum Drive reports.  The sha256 marker is only
    written after a full sha256 verification of the Hugging Face download.
    """

    def __init__(self, token: str, folder_id: str):
        if not isinstance(token, str) or not DRIVE_TOKEN.fullmatch(token):
            raise WorkerError("drive cache token is invalid")
        if not isinstance(folder_id, str) or not DRIVE_ID.fullmatch(folder_id):
            raise WorkerError("drive cache folder id is invalid")
        self._token = token
        self.folder_id = folder_id
        self._folders: dict[tuple[str, str], str] = {}

    def _open(
        self, method: str, url: str, *, body: bytes | None = None, headers: dict | None = None, timeout=120
    ):
        merged = {"Authorization": "Bearer " + self._token}
        merged.update(headers or {})
        for _ in range(4):
            host = urlsplit(url).hostname or ""
            if urlsplit(url).scheme != "https" or not (
                host == "www.googleapis.com" or host.endswith(".googleusercontent.com")
            ):
                raise WorkerError("drive cache refuses a non-Google URL")
            try:
                return _NO_REDIRECT.open(
                    Request(url, data=body, method=method, headers=merged), timeout=timeout
                )
            except HTTPError as exc:
                location = exc.headers.get("Location") if exc.code in {301, 302, 303, 307} else None
                if not location:
                    raise
                url = location  # Drive may hand media downloads to a googleusercontent host
        raise WorkerError("drive cache redirected too many times")

    def _json(self, method: str, url: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json; charset=UTF-8"} if data is not None else {}
        with self._open(method, url, body=data, headers=headers, timeout=60) as response:
            raw = response.read(1024 * 1024)
        value = json.loads(raw.decode("utf-8")) if raw else {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _q(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _list(self, name: str, parent: str, folder: bool) -> list[dict]:
        op = "=" if folder else "!="
        query = f"name = '{self._q(name)}' and mimeType {op} '{FOLDER_MIME}' and trashed = false and '{parent}' in parents"
        params = urlencode(
            {"q": query, "pageSize": "20", "fields": "files(id,name,size,md5Checksum,appProperties)"}
        )
        return self._json("GET", f"{DRIVE_API}/files?{params}").get("files") or []

    def _folder(self, name: str, parent: str, create: bool) -> str | None:
        cached = self._folders.get((parent, name))
        if cached:
            return cached
        found = self._list(name, parent, True)
        if found:
            folder_id = found[0]["id"]
        elif create:
            made = self._json(
                "POST",
                f"{DRIVE_API}/files?fields=id",
                {"name": name, "mimeType": FOLDER_MIME, "parents": [parent]},
            )
            folder_id = made.get("id")
            if not isinstance(folder_id, str):
                raise WorkerError("drive cache folder creation failed")
        else:
            return None
        self._folders[(parent, name)] = folder_id
        return folder_id

    def _parent(self, target: str, create: bool) -> tuple[str | None, str]:
        parts = Path(target).parts
        parent = self.folder_id
        for part in parts[:-1]:
            parent = self._folder(part, parent, create)
            if parent is None:
                return None, parts[-1]
        return parent, parts[-1]

    def lookup(self, target: str, sha256: str | None, size: int | None) -> dict | None:
        """Entry whose stored sha256/size marker matches; with sha256=None any marked entry (env archives)."""
        parent, name = self._parent(target, create=False)
        if parent is None:
            return None
        for entry in self._list(name, parent, False):
            props = entry.get("appProperties") or {}
            stored_size = int(entry.get("size") or 0)
            if not props.get("sha256") or (sha256 is not None and props["sha256"] != sha256) or stored_size <= 0:
                continue
            if size is not None and stored_size != size:
                continue
            if props.get("size") not in (None, str(stored_size)):
                continue
            return entry
        return None

    def fetch(self, entry: dict, destination: Path) -> None:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        digest = hashlib.md5(usedforsecurity=False)
        written = 0
        url = f"{DRIVE_API}/files/{quote(entry['id'])}?alt=media"
        with self._open("GET", url, timeout=300) as response, temporary.open("wb") as out:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                out.write(block)
                digest.update(block)
                written += len(block)
        if written != int(entry["size"]) or digest.hexdigest() != entry.get("md5Checksum"):
            temporary.unlink(missing_ok=True)
            raise WorkerError("drive cached file failed size/md5 check")
        os.replace(temporary, destination)

    def store(self, target: str, source: Path, sha256: str, md5: str, size: int) -> None:
        parent, name = self._parent(target, create=True)
        stale = self._list(name, parent, False)
        metadata = {
            "name": name,
            "parents": [parent],
            "appProperties": {"sha256": sha256, "size": str(size)},
        }
        init = self._open(
            "POST",
            f"{DRIVE_UPLOAD}/files?uploadType=resumable&fields=id,size,md5Checksum",
            body=json.dumps(metadata).encode(),
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(size),
                "X-Upload-Content-Type": "application/octet-stream",
            },
            timeout=60,
        )
        with init:
            location = init.headers.get("Location", "")
        if urlsplit(location).hostname != "www.googleapis.com":
            raise WorkerError("drive upload session location is invalid")
        result: dict = {}
        with source.open("rb") as stream:
            offset = 0
            while offset < size:
                chunk = stream.read(min(UPLOAD_CHUNK, size - offset))
                end = offset + len(chunk) - 1
                headers = {
                    "Content-Range": f"bytes {offset}-{end}/{size}",
                    "Content-Type": "application/octet-stream",
                }
                for attempt in range(3):
                    try:
                        with self._open(
                            "PUT", location, body=chunk, headers=headers, timeout=300
                        ) as response:
                            raw = response.read(1024 * 1024)
                        if raw:
                            result = json.loads(raw.decode("utf-8"))
                        break
                    except HTTPError as exc:
                        if exc.code == 308:  # chunk accepted, more expected
                            break
                        if exc.code < 500 or attempt == 2:
                            raise
                    except (URLError, OSError, TimeoutError):
                        if attempt == 2:
                            raise
                    time.sleep(2**attempt)
                offset = end + 1
        if int(result.get("size") or 0) != size or result.get("md5Checksum") != md5:
            raise WorkerError("drive upload verification failed")
        for old in stale:  # replace a stale copy only after the new one is verified
            self._json("PATCH", f"{DRIVE_API}/files/{quote(old['id'])}", {"trashed": True})


def _drive_cache(payload: dict, folder_key: str = "folder_id") -> DriveCache | None:
    """Model cache (``folder_id``) or runtime/env cache (``runtime_folder_id``) from the in-memory payload."""
    config = payload.get("drive_cache")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise WorkerError("drive_cache must be an object")
    return DriveCache(config.get("access_token"), config.get(folder_key))


def _check_loras(graph: dict, manifest: dict, user_files: list[dict]) -> None:
    """Fail before any download if every referenced LoRA is not in this job."""
    have = {w.get("target", "").removeprefix("loras/") for w in manifest.get("weight_files") or []}
    have |= {f.get("target") for f in user_files}
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") == "LoraLoaderModelOnly":
            name = node.get("inputs", {}).get("lora_name")
            if name not in have:
                raise WorkerError(f"LoRA 文件没有随任务下发：{name}")


def _restore_user_loras(comfy: Path, files: list[dict], cache: DriveCache | None) -> None:
    """The owner's own LoRAs from Drive (``loras/h3``) into models/loras/user/."""
    if cache is None:
        raise WorkerError("自定义 LoRA 需要 Drive 存储")
    loras = (comfy / "models" / "loras").resolve()
    for item in files:
        target = (loras / item["target"]).resolve()
        if loras not in target.parents or not target.name.endswith(".safetensors"):
            raise WorkerError("LoRA target is invalid")
        cache.fetch({"id": item["id"], "size": item["size"], "md5Checksum": item.get("md5Checksum")}, target)


def _restore_weights(
    comfy: Path, manifest: dict, timeout: float, cache: DriveCache | None, prep: bool
) -> list[dict]:
    """Bring every reviewed weight file in, four at a time."""
    checked = _validate_weights(manifest)
    models = (comfy / "models").resolve()
    models.mkdir(mode=0o700, parents=True, exist_ok=True)
    if prep and cache is not None:
        for item in checked:
            cache._parent(item["target"], create=True)  # noqa: SLF001

    def drive_entry(item: dict) -> dict:
        if cache is None:
            raise WorkerError("Drive weight import is unavailable: Drive cache is not configured")
        try:
            fields = urlencode({"fields": "id,size,md5Checksum,mimeType,trashed"})
            entry = cache._json("GET", f"{DRIVE_API}/files/{quote(item['drive_file_id'])}?{fields}")  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001 - make Drive failures explicit and safe
            raise WorkerError(f"Drive weight metadata unavailable: {_safe_error(exc)[:200]}") from exc
        if (
            entry.get("id") != item["drive_file_id"]
            or entry.get("trashed") is not False
            or not isinstance(entry.get("mimeType"), str)
            or entry.get("mimeType") == FOLDER_MIME
            or int(entry.get("size") or 0) != item["size"]
            or not isinstance(entry.get("md5Checksum"), str)
        ):
            raise WorkerError("Drive weight metadata failed id/trashed/type/size/md5 checks")
        return entry

    def verify_file(path: Path, item: dict) -> tuple[str, str, int]:
        if not path.is_file():
            raise WorkerError("Drive weight download produced no file")
        sha256, md5 = _hash_file(path)
        size = path.stat().st_size
        if sha256 != item["sha256"] or size != item["size"]:
            path.unlink(missing_ok=True)
            raise WorkerError("Drive weight sha256 or size mismatch")
        return sha256, md5, size

    def restore(item: dict, scratch: Path) -> dict:
        started = time.time()
        target = (models / item["target"]).resolve()
        if models not in target.parents:
            raise WorkerError("model target escaped Comfy models directory")
        is_drive = item.get("source") == "drive"
        record = {"target": item["target"], "sha256": item["sha256"]}
        if is_drive:
            record.update(source="drive", drive_file_id=item["drive_file_id"])
        if cache is not None:
            try:
                entry = cache.lookup(item["target"], item["sha256"], item["size"])
                if entry and not is_drive:
                    if prep:
                        return {**record, "source": "drive", "seconds": round(time.time() - started, 1)}
                    cache.fetch(entry, target)
                    return {**record, "source": "drive", "seconds": round(time.time() - started, 1)}
                if entry and is_drive:
                    cache.fetch(entry, target)
                    verify_file(target, item)
                    if prep:
                        target.unlink(missing_ok=True)
                    return {**record, "seconds": round(time.time() - started, 1)}
            except Exception as exc:  # noqa: BLE001 - Drive sources never fall back to Hugging Face
                if is_drive:
                    raise WorkerError(f"Drive weight import failed: {_safe_error(exc)[:200]}") from exc
                record["cache_error"] = _safe_error(exc)[:200]
                target.unlink(missing_ok=True)
        if is_drive:
            entry = drive_entry(item)
            try:
                cache.fetch(entry, target)  # type: ignore[union-attr]
                sha256, md5, size = verify_file(target, item)
                if prep:
                    cache.store(item["target"], target, sha256, md5, size)  # type: ignore[union-attr]
                    target.unlink(missing_ok=True)
                    record["cache_stored"] = True
                record["seconds"] = round(time.time() - started, 1)
                return record
            except WorkerError:
                target.unlink(missing_ok=True)
                raise
            except Exception as exc:  # noqa: BLE001 - preserve explicit Drive failure
                target.unlink(missing_ok=True)
                raise WorkerError(f"Drive weight import failed: {_safe_error(exc)[:200]}") from exc
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise WorkerError("Hugging Face client is unavailable") from exc
        try:
            source = Path(
                hf_hub_download(
                    item["repo_id"], item["filename"], revision=item["revision"], local_dir=str(scratch)
                )
            )
        except Exception as exc:
            raise WorkerError(f"reviewed model download failed: {_safe_error(exc)[:200]}") from exc
        if not source.is_file():
            raise WorkerError("reviewed model download produced no file")
        sha256, md5 = _hash_file(source)
        if sha256 != item["sha256"] or (item["size"] is not None and source.stat().st_size != item["size"]):
            raise WorkerError("downloaded model sha256 mismatch")
        record.update(source="huggingface")
        if prep:
            if cache is not None:
                try:
                    cache.store(item["target"], source, sha256, md5, source.stat().st_size)
                    record["cache_stored"] = True
                except Exception as exc:  # noqa: BLE001 - reported; the prep job is marked incomplete below
                    record.update(cache_stored=False, cache_error=_safe_error(exc)[:200])
            source.unlink(missing_ok=True)
        else:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        record["seconds"] = round(time.time() - started, 1)
        return record

    with tempfile.TemporaryDirectory(prefix="ai-studio-weight-") as scratch:
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(lambda item: restore(item, Path(scratch)), checked))


def _http_json(url: str, method: str = "GET", body: dict | None = None, timeout: float = 30) -> dict:
    request = Request(
        url, method=method, headers={"Accept": "application/json", "Content-Type": "application/json"}
    )
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urlopen(request, data=data, timeout=timeout) as response:
            raw = response.read(MAX_OUTPUT_BYTES + 1)
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        raise WorkerError("Comfy local API request failed") from exc
    if len(raw) > MAX_OUTPUT_BYTES:
        raise WorkerError("Comfy API response exceeds output bound")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise WorkerError("Comfy local API returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise WorkerError("Comfy local API returned invalid object")
    return value


_OUTPUT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
    ".mp4": "video/mp4", ".webm": "video/webm",
}
_INPUT_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")
MAX_INPUT_BYTES = 20 * 1024 * 1024


def _write_input_files(comfy: Path, snapshot: dict) -> None:
    """Reference images the job ships with go into ComfyUI/input, where LoadImage reads them."""
    folder = comfy / "input"
    folder.mkdir(exist_ok=True)
    for item in snapshot.get("input_files") or []:
        name = item.get("filename") if isinstance(item, dict) else None
        if not isinstance(name, str) or not _INPUT_NAME.fullmatch(name):
            raise WorkerError("input file name is not allowed")
        data = base64.b64decode(item.get("content_b64", ""), validate=True)
        if not data or len(data) > MAX_INPUT_BYTES:
            raise WorkerError("input file is empty or too large")
        (folder / name).write_bytes(data)


def _stream_output(response, target: Path, limit: int) -> int:
    total = 0
    partial = target.with_suffix(target.suffix + '.partial')
    try:
        with partial.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise WorkerError('outputs exceed configured bound')
                output.write(chunk)
        partial.replace(target)
        return total
    finally:
        partial.unlink(missing_ok=True)


def _run_comfy(root: Path, payload: dict, manifest: dict, graph: dict, stages: _Stages) -> list[dict]:
    commit = manifest.get("comfy_commit", manifest.get("comfy_version"))
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise WorkerError("Comfy workflow must lock a 40-character commit")
    allowlist = manifest.get("node_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        raise WorkerError("Comfy node allowlist is missing")
    for node in graph.values():
        if not isinstance(node, dict) or node.get("class_type") not in allowlist:
            raise WorkerError("bound graph contains an unreviewed node")
    comfy = root / "ComfyUI"
    _check_loras(graph, manifest, (payload.get("snapshot") or {}).get("lora_files") or [])
    timeout = min(float(payload.get("download_timeout_seconds", 600)), 600)
    nodes = _reviewed_custom_nodes(manifest.get("custom_nodes"))
    checked = _validate_weights(manifest)
    fingerprint = _warm_fingerprint(commit, nodes, checked)
    if _warm_take(root, fingerprint, nodes, checked):
        # Same runtime, same environment: the previous job left it restored. Nothing to download.
        stages.start("warm_reuse")
        env = {"source": "warm"}
        weights = [{"target": w["target"], "sha256": w["sha256"], "source": "warm", "seconds": 0.0} for w in checked]
    else:
        stages.start("restore_env")
        env = _restore_env(root, commit, timeout, _drive_cache(payload, "runtime_folder_id"), build=False,
                           custom_nodes=nodes)
        stages.start("restore_weights")
        weights = _restore_weights(comfy, manifest, timeout, _drive_cache(payload), prep=False)
    try:
        user_loras = (payload.get("snapshot") or {}).get("lora_files") or []
        if user_loras:
            stages.start("restore_loras")
            _restore_user_loras(comfy, user_loras, _drive_cache(payload))
        _write_input_files(comfy, payload.get("snapshot") or {})
        stages.start("comfy_start")
        log_path = root / "comfy.log"
        log = log_path.open("ab", buffering=0)
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                cwd=comfy,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
            deadline = time.time() + min(float(payload.get("max_runtime_minutes", 10)) * 60, 600 * 60)
            api = "http://127.0.0.1:8188"
            while time.time() < deadline:
                try:
                    _http_json(api + "/system_stats", timeout=5)
                    break
                except WorkerError:
                    if process.poll() is not None:
                        raise WorkerError("Comfy worker exited during startup")
                    time.sleep(1)
            else:
                raise WorkerError("Comfy worker startup timed out")
            stages.start("sampling")
            queued = _http_json(api + "/prompt", "POST", {"prompt": graph}, timeout=30)
            prompt_id = queued.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise WorkerError("Comfy queue did not return a prompt id")
            while time.time() < deadline:
                history = _http_json(api + "/history/" + prompt_id, timeout=30)
                item = history.get(prompt_id)
                if not isinstance(item, dict):
                    time.sleep(2)
                    continue
                status = item.get("status") or {}
                if status.get("status_str") in {"error", "failed"}:
                    detail = ""
                    for message in status.get("messages") or []:
                        if isinstance(message, list) and message and message[0] == "execution_error":
                            data = message[1] if len(message) > 1 and isinstance(message[1], dict) else {}
                            detail = f"{data.get('node_type')}: {data.get('exception_message')}"
                    raise WorkerError(f"Comfy workflow failed: {detail}"[:500])
                if status.get("completed") is not True and not item.get("outputs"):
                    time.sleep(2)
                    continue
                stages.end()
                outputs = []
                total = 0
                output_dir = root / "outputs"
                output_dir.mkdir(mode=0o700, exist_ok=True)
                for node_output in (item.get("outputs") or {}).values():
                    if not isinstance(node_output, dict):
                        continue
                    for category in ("images", "gifs", "videos"):
                        for entry in node_output.get(category, []) or []:
                            if not isinstance(entry, dict) or not isinstance(entry.get("filename"), str):
                                continue
                            filename = Path(entry["filename"]).name
                            mime = _OUTPUT_MIME.get(Path(filename).suffix.lower())
                            if mime is None:  # SaveVideo reports videos under "images"; judge by extension
                                continue
                            query = urlencode({'filename':filename,'subfolder':entry.get('subfolder',''),'type':entry.get('type','output')})
                            request = Request(
                                api + "/view?" + query, headers={"Accept": "application/octet-stream"}
                            )
                            target = output_dir / filename
                            output_limit = payload.get('_local_artifact_limit', MAX_OUTPUT_BYTES)
                            with urlopen(request, timeout=30) as response:
                                size = _stream_output(response, target, output_limit - total)
                            total += size
                            outputs.append(
                                {
                                    "filename": filename,
                                    "mime": mime,
                                    "path": str(target.relative_to(root)),
                                    "size": size,
                                    "metadata": {
                                        "comfy_commit": commit,
                                        "weights": weights,
                                        "env": env,
                                        "stage_seconds": stages.timings,
                                    },
                                }
                            )
                return outputs
            raise WorkerError("Comfy execution timed out")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log.close()
    finally:
        _warm_put(root, fingerprint)  # restore succeeded above, so the tree is reusable even if this job failed later


def _run_prep(root: Path, payload: dict, manifest: dict, stages: _Stages) -> list[dict]:
    """CPU-only cache fill: build the env archive and put every missing weight into Drive. No sampling."""
    commit = manifest.get("comfy_commit", manifest.get("comfy_version"))
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise WorkerError("Comfy workflow must lock a 40-character commit")
    if _drive_cache(payload) is None:
        raise WorkerError("prep needs the Drive cache")
    timeout = min(float(payload.get("download_timeout_seconds", 600)), 600)
    stages.start("prep_env")
    env = _restore_env(root, commit, timeout, _drive_cache(payload, "runtime_folder_id"), build=True,
                       custom_nodes=_reviewed_custom_nodes(manifest.get("custom_nodes")))
    stages.start("prep_weights")
    weights = _restore_weights(root / "ComfyUI", manifest, timeout, _drive_cache(payload), prep=True)
    stages.end()
    failed = [item["target"] for item in weights if item.get("cache_stored") is False]
    if env.get("source") != "drive" and env.get("cache_stored") is not True:
        failed.append("env")
    if failed:
        raise WorkerError("cache fill incomplete: " + ", ".join(failed))
    report = {
        "env": env,
        "weights": weights,
        "stage_seconds": stages.timings,
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
    }
    path = root / "outputs" / "prep-report.json"
    _atomic_json(path, report)
    return [
        {
            "filename": "prep-report.json",
            "mime": "application/json",
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "metadata": report,
        }
    ]


def run_job(payload: dict, root: Path, *, artifact_store=None) -> None:
    stages = _Stages(root)
    try:
        _status(root, "running")
        snapshot, manifest, graph = _snapshot(payload)
        if snapshot.get("system_smoke") is True:
            if manifest.get("executor") != "cuda-smoke":
                raise WorkerError("system smoke requires the cuda-smoke executor manifest")
            outputs = [run_cuda_smoke(root)]
        elif snapshot.get("prep") is True:
            outputs = _run_prep(root, payload, manifest, stages)
        else:
            local_payload = {**payload, '_local_artifact_limit': min(artifact_store.max_size, 2*1024**3)} if artifact_store is not None else {k:v for k,v in payload.items() if k!='_local_artifact_limit'}
            outputs = _run_comfy(root, local_payload, manifest, graph, stages)
        if artifact_store is not None:
            artifacts = []
            for item in outputs:
                source = root / item['path']
                if source.is_symlink() or root.resolve() not in source.resolve().parents:
                    raise WorkerError('worker output escaped execution directory')
                artifacts.append(artifact_store.publish_file(
                    payload['owner_id'], payload['execution_id'], source, item['mime']))
            _atomic_json(root / 'meta.json', {'execution_id': payload['execution_id'], 'artifacts': artifacts})
            _status(root, 'completed', artifacts=artifacts, timings=stages.timings)
        else:
            _atomic_json(root / "meta.json", {"execution_id": payload.get("execution_id"), "outputs": outputs})
            _status(root, "completed", outputs=outputs, timings=stages.timings)
    except Exception as exc:  # noqa: BLE001 - worker must persist every failure
        stages.end()
        message = _safe_error(exc)
        log = root / "comfy.log"
        if log.is_file():
            message = (message + " | comfy.log: " + log.read_text("utf-8", "replace")[-500:])[:1400]
        _status(root, "failed", error_code="gpu_worker_failed", error_message=message, timings=stages.timings)


class _MCPManifestStore:
    """Worker-side descriptor writer; media stays on disk for bounded MCP reads."""
    max_size = 2 * 1024**3

    def __init__(self, root):
        self.root = root.resolve()

    def publish_file(self, owner, execution, source, mime):
        import hashlib

        if not owner or not execution or source.is_symlink() or self.root not in source.resolve().parents:
            raise WorkerError('invalid artifact identity or path')
        size = source.stat().st_size
        if not 0 < size <= self.max_size:
            raise WorkerError('artifact size invalid')
        digest = hashlib.sha256()
        with source.open('rb') as file:
            while chunk := file.read(1024**2):
                digest.update(chunk)
        return {'filename': source.name, 'path': str(source.relative_to(self.root)),
            'mime': mime, 'size_bytes': size, 'sha256': digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "workbench":
        return _main_workbench_action(argv[1])
    if len(argv) != 1:
        return 2
    payload_path = Path(argv[0]).resolve()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise WorkerError("payload must be an object")
        root = Path(payload["execution_dir"]).resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if "drive_cache" in payload:  # keep the Drive token out of files that outlive the worker's read
            payload_path.chmod(0o600)
            _atomic_json(payload_path, {k: v for k, v in payload.items() if k != "drive_cache"})
            payload_path.chmod(0o600)
        artifacts = _MCPManifestStore(root) if payload.get('output_transport') == 'mcp-chunks-v1' else None
        run_job(payload, root, artifact_store=artifacts)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary is fail-closed
        try:
            root = (
                Path(payload.get("execution_dir", payload_path.parent)).resolve()
                if isinstance(payload, dict)
                else payload_path.parent
            )
            _status(root, "failed", error_code="gpu_worker_bootstrap_failed", error_message=_safe_error(exc))
        except Exception:  # noqa: BLE001 - status persistence must not mask exit code
            return 1
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
