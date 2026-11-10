"""Trusted Kaggle entrypoint. Model bytes are read through symlinks, never copied."""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

import runtime_worker as worker


def mount_models(comfy, manifest, slugs, input_root=Path('/kaggle/input')):
    records = []
    models = comfy / 'models'
    models.mkdir(parents=True, exist_ok=True)
    for item in worker._validate_weights(manifest):
        relative = Path(item['target'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in item['target']:
            raise worker.WorkerError('invalid model target')
        candidates = []
        for ref in slugs:
            parts = ref.split('/')
            if any(not part or part in {'.', '..'} or '\\' in part for part in parts) or len(parts) > 2:
                raise worker.WorkerError('invalid Dataset reference')
            layouts = [input_root / parts[-1]]
            if len(parts) == 2:
                layouts.extend([input_root / 'datasets' / ref, input_root / ref])
            bases = {path.resolve() for path in layouts if path.is_dir()}
            for base in bases:
                if input_root.resolve() not in base.parents:
                    raise worker.WorkerError('Dataset mount escaped input directory')
                for candidate in (base / relative, base / item['sha256']):
                    path = candidate.resolve()
                    if base not in path.parents:
                        raise worker.WorkerError('model path escaped dataset')
                    if path.is_file() and path not in candidates:
                        candidates.append(path)
        if len(candidates) != 1:
            raise worker.WorkerError(f"model must exist in exactly one attached Dataset: {relative}")
        source = candidates[0]
        if item.get('size') is not None and source.stat().st_size != item['size']:
            raise worker.WorkerError('dataset model size mismatch')
        if worker._sha256(source) != item['sha256']:
            raise worker.WorkerError('dataset model sha256 mismatch')
        target = models / relative
        if models.resolve() not in target.parent.resolve().parents and target.parent.resolve() != models.resolve():
            raise worker.WorkerError('model target escaped ComfyUI')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        records.append({'target': item['target'], 'sha256': item['sha256'], 'source': 'kaggle-dataset'})
    return records


def main(payload):
    root = Path(tempfile.mkdtemp(prefix='studio-job-'))
    # All dependencies, inputs, logs and credentials stay outside /kaggle/working.
    # Only the bounded result envelope is retained as notebook output.
    worker._restore_weights = lambda comfy, manifest, *a, **kw: mount_models(comfy, manifest, payload['dataset_refs'])
    worker._warm_take = lambda *a, **kw: False
    worker._warm_put = lambda *a, **kw: None
    diagnostic = worker.run_cuda_smoke(root)
    gpu = json.loads((root / diagnostic['path']).read_text())
    worker.run_job(payload, root)
    result = json.loads((root / 'status.json').read_text())
    result['execution_id'] = payload['execution_id']
    result['gpu_diagnostics'] = gpu
    if result.get('error_message'):
        result['message'] = result['error_message']
    result['outputs'] = []
    if result['state'] == 'completed':
        meta = json.loads((root / 'meta.json').read_text())
        total = 0
        for item in meta['outputs']:
            path = (root / item['path']).resolve()
            if root not in path.parents or not path.is_file():
                raise worker.WorkerError('output path escaped execution')
            total += path.stat().st_size
            if total > 100 * 1024**2:
                raise worker.WorkerError('output exceeds 100MB')
            result['outputs'].append({'filename': path.name, 'mime': item['mime'],
                                      'content_b64': base64.b64encode(path.read_bytes()).decode()})
    Path('/kaggle/working/studio-result.json').write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
