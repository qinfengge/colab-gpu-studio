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


def test_scheduler_completes_kaggle_job_with_zero_cu(adapter, monkeypatch):
    from app.gpu.scheduler import Scheduler

    from tests.gpu.test_scheduler import MemoryStore, make_job

    executor, handle, unused_job, calls, state = adapter
    store = MemoryStore()
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')
    job = make_job(store, owner='owner', provider='kaggle', job_id='integration',
                   estimated_units=0, budget_units=0, unit_rate=0,
                   execution_policy={'provider': 'kaggle', 'lifecycle': 'save'})
    scheduler = Scheduler(store, config=GpuConfig(real_enabled=True, unit_rate=2),
                          provider_factory=lambda j: kaggle.KaggleProvider(store, j['owner_id']),
                          executor_factory=lambda j: kaggle.KaggleExecutor(store, j['owner_id']))
    for _ in range(30):
        current = store.get('jobs', job['id'])
        if current.get('execution_id'):
            state.update(remote='complete', result={'execution_id': current['execution_id'], 'state': 'completed',
                'outputs': [{'filename': 'result.png', 'mime': 'image/png', 'content_b64': base64.b64encode(png).decode()}]})
        scheduler.tick()
        if store.get('jobs', job['id'])['state'] == 'completed':
            break
    assert store.get('jobs', job['id'])['state'] == 'completed'
    assert store.list('usage')[0]['units'] == 0
    assert len(store.list('assets')) == 1
    assert sum(c[:2] == ('kernels', 'push') for c in calls) == 1

@pytest.mark.parametrize('layout', ['models', 'tester/models', 'datasets/tester/models'])
def test_dataset_mount_layouts(monkeypatch, tmp_path, layout):
    import hashlib
    module = runner(monkeypatch)
    content = b'weights'
    digest = hashlib.sha256(content).hexdigest()
    source = tmp_path / 'input' / layout / digest
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    monkeypatch.setattr(runtime_worker, '_validate_weights', lambda manifest: [
        {'target': 'checkpoints/model.safetensors', 'sha256': digest, 'size': len(content)}])
    module.mount_models(tmp_path / 'comfy', {}, ['tester/models'], tmp_path / 'input')
    assert (tmp_path / 'comfy/models/checkpoints/model.safetensors').resolve() == source.resolve()


def test_cancel_acknowledged_releases_but_cancel_requested_does_not(adapter):
    executor, handle, job, calls, state = adapter
    executor.start(handle, job, job['execution_id'])
    state['remote'] = 'KernelWorkerStatus.CANCEL_REQUESTED'
    assert not executor.delete(handle)
    state['remote'] = 'KernelWorkerStatus.CANCEL_ACKNOWLEDGED'
    assert executor.delete(handle)


def test_smoke_admission_needs_drive_but_no_colab_or_cu_rate(monkeypatch):
    import uuid

    from app import review_routes
    from app.db import Store
    from app.gpu.google import GoogleService
    from app.gpu.repository import Repository

    from tests.test_review_admission import OWNER, body, request, valid_model, valid_workflow

    monkeypatch.setenv('GPU_COMPUTE_PROVIDER', 'kaggle')
    monkeypatch.setenv('GPU_STORAGE', 'drive')
    monkeypatch.setenv('KAGGLE_USERNAME', 'tester')
    monkeypatch.setenv('KAGGLE_MODEL_DATASETS', 'tester/models')
    drive_id = str(uuid.uuid4())
    monkeypatch.setenv('KAGGLE_DRIVE_ACCOUNT_ID', drive_id)
    monkeypatch.setattr(GoogleService, 'status', lambda _: {'drive': {'connected': True}, 'colab': {'connected': False}})
    monkeypatch.setattr(review_routes.GpuConfig, 'from_env', lambda: GpuConfig(real_enabled=True, unit_rate=None))
    st = Store('sqlite:///:memory:')
    repo = Repository(st)
    repo.create('accounts', 'owner', {'enabled': True, 'allow_allocation': False}, record_id=drive_id)
    project = st.create('projects', 'owner', {'title': 'Kaggle test'})
    model = valid_model(repo)
    review_routes.review_model(model['id'], {'confirmed': True, 'license_accepted': True}, request(), OWNER, st)
    workflow = valid_workflow(repo)
    review_routes.review_workflow(workflow['id'], {'confirmed': True}, request(), OWNER, st)
    spec = kaggle.KaggleProvider().list_specs()[0]
    response = review_routes.real_smoke(workflow['id'], {'confirmed': True, 'project_id': project['id'], 'spec': spec},
                                       request({'Idempotency-Key': 'kaggle-smoke'}), OWNER, st)
    job = repo.get('jobs', body(response)['job_id'], 'owner')
    assert job['provider'] == 'kaggle'
    assert job['estimated_units'] == job['budget_units'] == job['unit_rate'] == 0
    assert job['snapshot']['kaggle_datasets'] == ['tester/models']
    assert job['account_id'] == drive_id
    assert repo.list('leases') == []

    # Exercise the acceptance -> publication -> normal admission path with local evidence.
    from app.admission import admit_job
    from app.gpu.registry import Registry
    from app.storage import save_asset

    from tests.test_review_admission import PNG
    asset = save_asset(st, 'owner', project['id'], PNG, 'smoke.png')
    repo.update('jobs', job['id'], {'state': 'completed', 'output_asset_ids': [asset['id']],
                                  'gpu_diagnostics': {'cuda_available': True, 'device_count': 2}}, 'owner')
    repo.create('leases', 'owner', {'owner_job_id': job['id'], 'provider': 'kaggle', 'state': 'deleted',
                                  'runtime_id': 'runtime', 'request_id': 'request'})
    review_routes.accept_smoke(workflow['id'], {'job_id': job['id'], 'confirmed': True}, request(), OWNER, st)
    Registry(repo).publish('owner', workflow['id'], confirmed=True)
    normal = admit_job(st, OWNER, {'project_id': project['id'], 'model_id': model['model_id'],
        'workflow_key': workflow['workflow_key'], 'workflow_version': 1, 'params': {}, 'asset_ids': [],
        'execution_policy': {'provider': 'kaggle', 'lifecycle': 'save'}}, 'normal-kaggle')
    assert normal['provider'] == 'kaggle'
    assert normal['unit_rate'] == 0
    assert normal['drive_account_id'] == drive_id


@pytest.mark.parametrize('verified', [True, False])
def test_original_input_backup_verifies_drive_bytes(monkeypatch, verified):
    from app import kaggle_storage
    from app.common import APIError
    from app.db import Store
    from app.storage import save_asset

    from tests.test_review_admission import PNG

    st = Store('sqlite:///:memory:')
    project = st.create('projects', 'owner', {'title': 'Inputs'})
    asset = save_asset(st, 'owner', project['id'], PNG, 'input.png')
    calls = []
    monkeypatch.setattr(kaggle_storage, 'validate_drive', lambda *args: None)
    monkeypatch.setenv('KAGGLE_DRIVE_ACCOUNT_ID', 'drive-account')

    class Drive:
        def __init__(self, job, repo):
            assert job['drive_account_id'] == 'drive-account'
        def put(self, key, content, mime):
            assert key.startswith('inputs/') and content == PNG
            calls.append('put')
            return {'id': 'cloud-file'}
        def verify(self, record, content):
            calls.append('verify')
            return verified
        def close(self):
            calls.append('close')

    monkeypatch.setattr(kaggle_storage, 'DriveStorage', Drive)
    if verified:
        result = kaggle_storage.sync_input_asset(st, 'owner', asset)
        assert result['drive_sync_status'] == 'synced'
    else:
        with pytest.raises(APIError):
            kaggle_storage.sync_input_asset(st, 'owner', asset)
        assert st.get('assets', asset['id'], 'owner')['drive_sync_status'] == 'failed'
        assert kaggle_storage.read_asset(asset) == PNG
    assert calls == ['put', 'verify', 'close']


def test_dataset_preparation_preserves_verified_weights_and_rejects_corruption(tmp_path):
    import hashlib

    script = Path(__file__).resolve().parents[2] / 'scripts/prepare_kaggle_dataset.py'
    spec = importlib.util.spec_from_file_location('prepare_kaggle_dataset', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'models'
    source.mkdir()
    content = b'private weights'
    (source / 'model.bin').write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = {'weight_files': [{'target': 'model.bin', 'sha256': digest, 'size': len(content)}]}
    destination = tmp_path / 'upload'
    module.prepare(manifest, source, destination, 'tester/models', 'other')
    assert (destination / digest).read_bytes() == content
    metadata = json.loads((destination / 'dataset-metadata.json').read_text())
    assert metadata['id'] == 'tester/models'
    with pytest.raises(ValueError, match='empty'):
        module.prepare(manifest, source, destination, 'tester/models', 'other')
    (source / 'model.bin').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='mismatch'):
        module.prepare(manifest, source, tmp_path / 'invalid', 'tester/models', 'other')
    assert not (tmp_path / 'invalid').exists()
    manifest['weight_files'][0]['target'] = '../outside.bin'
    (tmp_path / 'outside.bin').write_bytes(content)
    with pytest.raises(ValueError, match='unsafe'):
        module.prepare(manifest, source, tmp_path / 'escape', 'tester/models', 'other')
