"""Drive backup for original inputs; generated outputs use the scheduler's uploader."""
import os

from .common import APIError
from .gpu.drive import DriveStorage
from .gpu.kaggle import validate_drive
from .gpu.repository import Repository
from .storage import read_asset


def sync_input_asset(st, owner, asset):
    repo = Repository(st)
    account_id = os.getenv('KAGGLE_DRIVE_ACCOUNT_ID', '').strip()
    validate_drive(repo, owner, account_id)
    storage = DriveStorage({'owner_id': owner, 'drive_account_id': account_id}, repo)
    try:
        content = read_asset(asset)
        record = storage.put(f"inputs/{asset['project_id']}/{asset['id']}/{asset['filename'][:120]}", content, asset['mime'])
        if not storage.verify(record, content):
            raise ValueError('Drive verification failed')
        return st.update('assets', asset['id'], {'drive_record': record, 'drive_sync_status': 'synced'}, owner)
    except Exception as exc:
        st.update('assets', asset['id'], {'drive_sync_status': 'failed'}, owner)
        raise APIError('DRIVE_SYNC_FAILED', '素材已保存本地，Drive 同步失败，请稍后重试同步', 502, True) from exc
    finally:
        storage.close()
