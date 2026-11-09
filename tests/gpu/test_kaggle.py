import ast
import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from app.gpu import kaggle, runtime_worker
from app.gpu.compute import ComputeError, DisabledProvider, GpuConfig, default_provider_factory


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('KAGGLE_USERNAME', 'tester')
    monkeypatch.setenv('KAGGLE_MODEL_DATASETS', 'tester/models')
    calls = []
    state = {'remote': 'running', 'private': True, 'push_error': False}

    def fake_cli(*args, **kwargs):
        calls.append(args)
        if args[:2] == ('datasets', 'metadata'):
            dest = Path(args[args.index('-p') + 1])
            (dest / 'dataset-metadata.json').write_text(json.dumps({'isPrivate': state['private']}))
        elif args[:2] == ('kernels', 'status'):
            return f'{args[2]} has status "{state["remote"]}"'
        elif args[:2] == ('kernels', 'push') and state['push_error']:
            raise ComputeError('submission response lost')
        elif args[:2] == ('kernels', 'output'):
            dest = Path(args[args.index('-p') + 1])
            (dest / 'studio-result.json').write_text(json.dumps(state['result']))
        return ''

    monkeypatch.setattr(kaggle, 'cli', fake_cli)
    executor = kaggle.KaggleExecutor(owner_id='owner')
    handle = executor.create({}, 'request', 'job', 'runtime')
    job = {'id': 'job', 'owner_id': 'owner', 'execution_id': 'execution', 'max_runtime_minutes': 1,
           'snapshot': {'manifest': {}}}
    return executor, handle, job, calls, state


def test_private_notebook_no_credentials_and_bounded_runtime(adapter, monkeypatch):
    executor, handle, job, calls, state = adapter
    monkeypatch.setenv('KAGGLE_API_TOKEN', 'NEVER-IN-NOTEBOOK')
    executor.start(handle, job, 'execution')
    folder = kaggle.state_dir(handle)
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True
    assert metadata['dataset_sources'] == ['tester/models']
    nb = json.loads((folder / 'task.ipynb').read_text())
    assert 'NEVER-IN-NOTEBOOK' not in json.dumps(nb)
    ast.parse(nb['cells'][0]['source'])
    push = next(c for c in calls if c[:2] == ('kernels', 'push'))
    assert push[-2:] == ('-t', '60')


def test_public_dataset_fails_before_submission(adapter):
    executor, handle, job, calls, state = adapter
    state['private'] = False
    with pytest.raises(ComputeError, match='私有'):
        executor.start(handle, job, 'execution')
    assert not any(c[:2] == ('kernels', 'push') for c in calls)
    assert executor.delete(handle)


def test_ambiguous_submission_does_not_push_second_version(adapter):
    executor, handle, job, calls, state = adapter
    state['push_error'] = True
    with pytest.raises(ComputeError):
        executor.start(handle, job, 'execution')
    fresh = kaggle.KaggleExecutor(owner_id='owner')
    fresh.start(handle, job, 'execution')
    assert sum(c[:2] == ('kernels', 'push') for c in calls) == 1
    assert not fresh.delete(handle)
    state['remote'] = 'complete'
    assert fresh.delete(handle)


def test_snapshot_change_is_not_retried(adapter):
    executor, handle, job, calls, state = adapter
    executor.start(handle, job, 'execution')
    with pytest.raises(ComputeError, match='snapshot'):
        executor.start(handle, {**job, 'max_runtime_minutes': 2}, 'execution')


def test_owner_and_handle_identity_enforced(adapter):
    executor, handle, job, calls, state = adapter
    with pytest.raises(ComputeError):
        executor.inspect({**handle, 'runtime_name': 'other/task'})
    with pytest.raises(ComputeError):
        kaggle.KaggleProvider(owner_id='other').delete(handle)


def test_output_identity_and_decode(adapter):
    executor, handle, job, calls, state = adapter
    state.update(remote='complete', result={'execution_id': 'execution', 'state': 'completed',
        'outputs': [{'filename': 'result.png', 'mime': 'image/png', 'content_b64': base64.b64encode(b'image').decode()}]})
    result = executor.poll(handle, job)
    assert result['outputs'][0]['content'] == b'image'
    state['result']['execution_id'] = 'other'
    with pytest.raises(ComputeError, match='execution mismatch'):
        executor.poll(handle, job)


def test_real_gate_blocks_kaggle(monkeypatch):
    provider = default_provider_factory({'provider': 'kaggle'}, GpuConfig(real_enabled=False))
    assert isinstance(provider, DisabledProvider)
    monkeypatch.setenv('STUDIO_OFFLINE_TEST', '1')
    assert isinstance(default_provider_factory({'provider': 'kaggle'}, GpuConfig(real_enabled=True)), DisabledProvider)


def runner(monkeypatch):
    monkeypatch.setitem(sys.modules, 'runtime_worker', runtime_worker)
    path = Path(kaggle.__file__).with_name('kaggle_runner.py')
    spec = importlib.util.spec_from_file_location('test_kaggle_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mount_hash_validation_and_no_copy(monkeypatch, tmp_path):
    import hashlib
    module = runner(monkeypatch)
    content = b'private model weights'
    source = tmp_path / 'input' / 'models' / 'checkpoints' / 'model.safetensors'
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    weight = {'target': 'checkpoints/model.safetensors', 'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content)}
    monkeypatch.setattr(runtime_worker, '_validate_weights', lambda manifest: [weight])
    comfy = tmp_path / 'ComfyUI'
    try:
        module.mount_models(comfy, {}, ['models'], tmp_path / 'input')
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 1314:
            pytest.skip('Windows symlink privilege; Linux CI verifies direct mounting')
        raise
    link = comfy / 'models' / weight['target']
    assert link.is_symlink()
    assert link.resolve() == source.resolve()
    source.write_bytes(b'corrupt model')
    with pytest.raises(runtime_worker.WorkerError, match='size mismatch'):
        module.mount_models(tmp_path / 'another', {}, ['models'], tmp_path / 'input')


def test_missing_model_never_downloads(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    monkeypatch.setattr(runtime_worker, '_validate_weights', lambda manifest: [{'target': 'missing', 'sha256': 'a'*64}])
    with pytest.raises(runtime_worker.WorkerError, match='exactly one'):
        module.mount_models(tmp_path / 'comfy', {}, ['models'], tmp_path / 'input')
