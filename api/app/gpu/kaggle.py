"""Kaggle batch compute. Credentials stay on the Studio host, never in notebooks."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .compute import ComputeError

REF = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$")
TERMINAL = {"complete", "error", "cancelled", "canceled", "cancel_acknowledged"}


def enabled():
    return os.getenv("GPU_COMPUTE_PROVIDER", "colab") == "kaggle"


def configuration():
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    datasets = [x.strip() for x in os.getenv("KAGGLE_MODEL_DATASETS", "").split(",") if x.strip()]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", username) or not datasets or any(not REF.fullmatch(x) for x in datasets):
        raise ComputeError("请配置 KAGGLE_USERNAME 和 KAGGLE_MODEL_DATASETS（owner/dataset）")
    if len({x.split('/')[1] for x in datasets}) != len(datasets):
        raise ComputeError("模型 Dataset 的名称不能重复")
    return username, datasets


def cli(*args, timeout=120):
    if os.getenv("STUDIO_OFFLINE_TEST") == "1":
        raise ComputeError("离线模式禁止调用 Kaggle")
    # CLI output may contain URLs or account details; never surface raw error bodies.
    try:
        result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComputeError("Kaggle 请求未确认，请检查任务状态，勿重复提交") from exc
    if result.returncode:
        raise ComputeError("Kaggle 请求失败，请检查凭据、联网权限与剩余 GPU 配额")
    return result.stdout


def status(ref):
    output = cli("kernels", "status", ref)
    match = re.search(r'has status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', output)
    if not match:
        raise ComputeError("无法识别 Kaggle 任务状态")
    return match.group(1).lower()


def state_dir(handle):
    runtime = handle.get("runtime_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", runtime):
        raise ComputeError("无效的 Kaggle runtime identity")
    return Path(os.getenv("DATA_DIR", "./data")) / "kaggle" / runtime


def notebook(payload):
    """Inline trusted sources; do not clone a mutable application branch remotely."""
    source = Path(__file__).with_name("runtime_worker.py").read_text(encoding="utf-8")
    runner = Path(__file__).with_name("kaggle_runner.py").read_text(encoding="utf-8")
    def enc(value):
        return base64.b64encode(value.encode()).decode()
    code = f'''import base64, json, pathlib, subprocess, sys, tempfile
root = pathlib.Path(tempfile.mkdtemp(prefix="studio-"))
(root / "runtime_worker.py").write_bytes(base64.b64decode({enc(source)!r}))
(root / "kaggle_runner.py").write_bytes(base64.b64decode({enc(runner)!r}))
(root / "payload.json").write_bytes(base64.b64decode({enc(json.dumps(payload))!r}))
subprocess.run([sys.executable, str(root / "kaggle_runner.py"), str(root / "payload.json")], check=True, timeout={int(payload['max_runtime_minutes'])*60})
'''
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{"id": "studio-task", "cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": code}]}


class KaggleProvider:
    is_mock = False

    def __init__(self, store=None, owner_id=None):
        self.store, self.owner_id = store, owner_id

    def list_specs(self):
        configuration()
        return [{"name": "Kaggle T4", "provider": "kaggle", "eligible": True,
                 "availability": "requested; subject to Kaggle quota and capacity",
                 "key": {"variant": "GPU", "accelerator": "T4", "shape": "KAGGLE"}, "vram_gb": 16}]

    def create(self, spec, request_id, job_id, runtime_id):
        username, _ = configuration()
        slug = "studio-" + hashlib.sha256((str(self.owner_id) + ':' + runtime_id).encode()).hexdigest()[:32]
        handle = {"runtime_name": username + '/' + slug, "runtime_id": runtime_id,
                  "request_id": request_id, "job_id": job_id, "owner_id": self.owner_id, "ready": True}
        folder = state_dir(handle)
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / "handle.json"
        if marker.exists() and json.loads(marker.read_text()) != handle:
            raise ComputeError("Kaggle runtime identity mismatch")
        marker.write_text(json.dumps(handle), encoding="utf-8")
        return handle

    def _verify(self, handle):
        folder = state_dir(handle)
        original = json.loads((folder / "handle.json").read_text())
        if any(original.get(k) != handle.get(k) for k in ('runtime_name', 'runtime_id', 'request_id', 'job_id', 'owner_id')):
            raise ComputeError("Kaggle runtime identity mismatch")
        if self.owner_id != original['owner_id']:
            raise ComputeError("Kaggle runtime owner mismatch")
        return folder

    def inspect(self, handle):
        folder = self._verify(handle)
        return {**handle, "ready": True, "deleted": (folder / "released").exists()}

    def delete(self, handle):
        folder = self._verify(handle)
        # Official CLI has no stop-session command. Never claim release while running.
        # Keep notebook/output history; deletion is not a substitute for cancellation.
        if (folder / "submitted").exists() and status(handle['runtime_name']) not in TERMINAL:
            return False
        (folder / "released").touch()
        return True


class KaggleExecutor(KaggleProvider):
    def start(self, handle, job, execution_id):
        folder = self._verify(handle)
        username, datasets = configuration()
        if not handle['runtime_name'].startswith(username + '/'):
            raise ComputeError("Kaggle account changed during execution")
        snapshot = job['snapshot']
        datasets = snapshot.get('kaggle_datasets') or datasets
        if snapshot.get('lora_files'):
            raise ComputeError("请先将自定义 LoRA 放入私有 Kaggle Dataset")
        payload = {"execution_id": execution_id, "owner_id": job['owner_id'],
                   "snapshot": snapshot, "max_runtime_minutes": job['max_runtime_minutes'],
                   "dataset_refs": datasets}
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        marker = folder / "submitted"
        if marker.exists():
            if marker.read_text() != signature:
                raise ComputeError("Kaggle execution snapshot mismatch")
            # Submission may have succeeded before a timeout. Do not push another version.
            status(handle['runtime_name'])
            return {"execution_id": execution_id, "recovered": True}
        for index, dataset in enumerate(datasets):
            meta_dir = folder / f'dataset-{index}'
            meta_dir.mkdir(exist_ok=True)
            cli('datasets', 'metadata', dataset, '-p', str(meta_dir))
            info = json.loads((meta_dir / 'dataset-metadata.json').read_text(encoding='utf-8'))
            if info.get('isPrivate', info.get('is_private')) is not True:
                raise ComputeError('模型 Dataset 必须是私有的；拒绝启动 GPU')
        metadata = {"id": handle['runtime_name'], "title": handle['runtime_name'].split('/')[1],
                    "code_file": "task.ipynb", "language": "python", "kernel_type": "notebook",
                    "is_private": True, "enable_gpu": True, "enable_internet": True,
                    "dataset_sources": datasets, "competition_sources": [], "kernel_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (folder / "task.ipynb").write_text(json.dumps(notebook(payload)), encoding="utf-8")
        # O_EXCL survives worker restart and prevents duplicate launches after ambiguous failures.
        try:
            with marker.open('x') as stream:
                stream.write(signature)
        except FileExistsError:
            raise ComputeError("Kaggle submission already in progress") from None
        cli("kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4",
            "-t", str(int(job['max_runtime_minutes']) * 60))
        return {"execution_id": execution_id}

    def poll(self, handle, job):
        folder = self._verify(handle)
        remote = status(handle['runtime_name'])
        if remote not in TERMINAL:
            return {"state": "running", "outputs": []}
        if remote != 'complete':
            return {"state": "failed", "outputs": [], "error_code": "kaggle_execution_failed",
                    "message": "Kaggle 任务失败或已取消，请查看私有 Notebook 日志"}
        output = folder / "download"
        output.mkdir(exist_ok=True)
        cli("kernels", "output", handle['runtime_name'], "-p", str(output),
            "--file-pattern", r"^studio-result\.json$", "--force")
        result_file = output / "studio-result.json"
        if not result_file.is_file() or result_file.stat().st_size > 140 * 1024**2:
            raise ComputeError("Kaggle 结果缺失或超过大小限制")
        result = json.loads(result_file.read_text(encoding='utf-8'))
        if result.get('execution_id') != job['execution_id']:
            raise ComputeError("Kaggle output execution mismatch")
        total = 0
        for item in result.get('outputs', []):
            raw = base64.b64decode(item.pop('content_b64'), validate=True)
            total += len(raw)
            if total > 100 * 1024**2 or Path(item['filename']).name != item['filename']:
                raise ComputeError("Kaggle output exceeds bounds")
            item['content'] = raw
        return result


def validate_drive(repo, owner, account_id):
    from ..common import APIError
    from .google import GoogleService

    if os.getenv('GPU_STORAGE', 'local') != 'drive':
        raise APIError('DRIVE_STORAGE_REQUIRED', 'Kaggle 模式需要 GPU_STORAGE=drive', 422)
    if not account_id or not repo.get('accounts', account_id, owner):
        raise APIError('DRIVE_ACCOUNT_REQUIRED', '请配置属于当前用户的 KAGGLE_DRIVE_ACCOUNT_ID', 422)
    if not GoogleService(repo, owner, account_id=account_id).status().get('drive', {}).get('connected'):
        raise APIError('GOOGLE_NOT_CONNECTED', '请先完成该账号的 Google Drive 授权', 409)
