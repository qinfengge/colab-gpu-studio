"""Independent GPU repository backed by app.db; no Atelier imports."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Mapping

from sqlalchemy import String, UniqueConstraint, inspect, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Mapped, mapped_column

from app import db as db_module
from app.db import Base, CommonMixin, Store, utcnow

from .manifest import digest


class GpuModelRow(CommonMixin, Base):
    __tablename__ = "gpu_models"


class GpuWorkflowRow(CommonMixin, Base):
    __tablename__ = "gpu_workflows"
    workflow_key: Mapped[str] = mapped_column(String(255), nullable=False)
    # Stored as text so both legacy integer versions and semver versions have
    # one database identity.  _row converts digit-only legacy values back to int.
    workflow_version: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "workflow_key", "workflow_version", name="uq_gpu_workflows_owner_key_version"
        ),
    )


class GpuJobRow(CommonMixin, Base):
    __tablename__ = "gpu_jobs"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (UniqueConstraint("owner_id", "idempotency_key", name="uq_gpu_jobs_owner_idempotency"),)


class GpuBatchRow(CommonMixin, Base):
    __tablename__ = "gpu_batches"
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (UniqueConstraint("owner_id", "idempotency_key", name="uq_gpu_batches_owner_key"),)


class GpuExecutionRow(CommonMixin, Base):
    __tablename__ = "gpu_executions"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuLeaseRow(CommonMixin, Base):
    __tablename__ = "gpu_leases"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuEventRow(CommonMixin, Base):
    __tablename__ = "gpu_events"


class GpuPresetRow(CommonMixin, Base):
    __tablename__ = "gpu_presets"


class GpuUsageRow(CommonMixin, Base):
    __tablename__ = "gpu_usage"


class GpuFileRow(CommonMixin, Base):
    __tablename__ = "gpu_files"


class GpuCredentialRow(CommonMixin, Base):
    __tablename__ = "gpu_credentials"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuAccountRow(CommonMixin, Base):
    __tablename__ = "gpu_accounts"


class GpuInstanceRow(CommonMixin, Base):
    __tablename__ = "gpu_instances"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    current_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    __table_args__ = (
        UniqueConstraint("owner_id", "current_job_id", name="uq_gpu_instances_owner_current_job"),
    )


class GpuInstanceUsageSegmentRow(CommonMixin, Base):
    __tablename__ = "gpu_instance_usage_segments"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuAllocationCursorRow(CommonMixin, Base):
    __tablename__ = "gpu_allocation_cursors"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuOperationRow(CommonMixin, Base):
    __tablename__ = "gpu_operations"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


_GPU_MODELS = {
    "batches": GpuBatchRow,
    "executions": GpuExecutionRow,
    "models": GpuModelRow,
    "workflows": GpuWorkflowRow,
    "jobs": GpuJobRow,
    "leases": GpuLeaseRow,
    "events": GpuEventRow,
    "presets": GpuPresetRow,
    "usage": GpuUsageRow,
    "files": GpuFileRow,
    "credentials": GpuCredentialRow,
    "accounts": GpuAccountRow,
    "instances": GpuInstanceRow,
    "instance_usage_segments": GpuInstanceUsageSegmentRow,
    "allocation_cursors": GpuAllocationCursorRow,
    "operations": GpuOperationRow,
}
_GPU_TABLES = [model.__table__ for model in _GPU_MODELS.values()]
for _kind, _model in _GPU_MODELS.items():
    db_module.MODELS.setdefault("gpu_" + _kind, _model)


def ensure_schema(store: Store) -> None:
    for attempt in range(6):
        try:
            with store.db_lock():
                Base.metadata.create_all(store.engine, tables=_GPU_TABLES, checkfirst=True)
                # create_all does not migrate a database made by an older
                # build.  Add the two identity columns and the unique index
                # idempotently before any workflow can be written.
                columns = {column["name"] for column in inspect(store.engine).get_columns("gpu_workflows")}
                with store.engine.begin() as connection:
                    if "workflow_key" not in columns:
                        connection.execute(
                            text("ALTER TABLE gpu_workflows ADD COLUMN workflow_key VARCHAR(255)")
                        )
                    if "workflow_version" not in columns:
                        connection.execute(
                            text("ALTER TABLE gpu_workflows ADD COLUMN workflow_version VARCHAR(64)")
                        )
                    if store.engine.dialect.name in {"sqlite", "postgresql"}:
                        connection.execute(
                            text(
                                "CREATE UNIQUE INDEX IF NOT EXISTS uq_gpu_workflows_owner_key_version ON gpu_workflows (owner_id, workflow_key, workflow_version)"
                            )
                        )
                    for table, column_specs in {
                        "gpu_jobs": {"account_id": "VARCHAR(64)", "instance_id": "VARCHAR(64)"},
                        "gpu_executions": {"account_id": "VARCHAR(64)", "instance_id": "VARCHAR(64)"},
                        "gpu_leases": {"account_id": "VARCHAR(64)", "instance_id": "VARCHAR(64)"},
                        "gpu_credentials": {"account_id": "VARCHAR(64)"},
                    }.items():
                        existing = {column["name"] for column in inspect(store.engine).get_columns(table)}
                        for column, column_type in column_specs.items():
                            if column not in existing:
                                connection.execute(
                                    text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
                                )
                        for column in column_specs:
                            connection.execute(
                                text(
                                    f"CREATE INDEX IF NOT EXISTS ix_{table}_{column}_v020 "
                                    f"ON {table} ({column})"
                                )
                            )
                    connection.execute(
                        text(
                            "CREATE UNIQUE INDEX IF NOT EXISTS uq_gpu_instances_owner_current_job "
                            "ON gpu_instances (owner_id, current_job_id)"
                        )
                    )
                    # DDL alone cannot reconstruct the v0.2 identity graph.
                    # Keep startup safe and local: this helper only repairs
                    # owner/account/instance identity metadata and never calls
                    # OAuth, Colab, or Drive.
                    from .legacy_migration import upgrade_legacy_identity

                    upgrade_legacy_identity(connection)
            return
        except OperationalError:
            if attempt == 5:
                raise
            time.sleep(0.03 * (attempt + 1))


def retarget_handle(handle, job_id):
    """Return the runtime handle re-owned by job_id (providers check handle ownership against the lease owner)."""
    if not isinstance(handle, dict):
        return handle
    return {**handle, 'job_id': job_id, 'owner_job_id': job_id}


class Repository:
    _CLEANUP_CLAIM_TTL_SECONDS = 300.0
    _CLEANUP_RETRY_BASE_SECONDS = 2.0
    _CLEANUP_RETRY_MAX_SECONDS = 300.0

    def __init__(self, store: Store):
        self.store = store
        ensure_schema(store)

    def dispatch_lock(self):
        """Expose the Store's cross-process tick lock to scheduler callers."""
        return self.store.dispatch_lock()

    @staticmethod
    def _model(kind):
        try:
            return _GPU_MODELS[kind]
        except KeyError as exc:
            raise ValueError(f"未知GPU实体类型: {kind}") from exc

    @staticmethod
    def _row(row):
        if row is None:
            return None
        reserved = {"id", "owner_id", "project_id", "state", "version", "created_at", "updated_at"}
        result = {
            "id": row.id,
            "owner_id": row.owner_id,
            "project_id": row.project_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
        result.update({k: copy.deepcopy(v) for k, v in (row.data or {}).items() if k not in reserved})
        # v0.2 stores identity in typed columns as well as JSON.  Old rows may
        # have only the typed value after Alembic 0003; expose and reconcile it
        # at the repository boundary instead of silently dropping ownership.
        for field in ("account_id", "instance_id", "current_job_id"):
            if hasattr(row, field):
                value = getattr(row, field)
                if value is not None:
                    result[field] = copy.deepcopy(value)
        if isinstance(row, GpuWorkflowRow):
            workflow_version = row.workflow_version
            if isinstance(workflow_version, str) and workflow_version.isdigit():
                workflow_version = int(workflow_version)
            result["workflow_key"] = row.workflow_key
            result["workflow_version"] = workflow_version
        if row.state is not None:
            result["state"] = row.state
        if row.version:
            result["version"] = row.version
        if isinstance(row, GpuJobRow):
            result.update(idempotency_key=row.idempotency_key, body_hash=row.body_hash)
        return result

    def create(self, kind, owner_id, data, project_id=None, record_id=None):
        model = self._model(kind)
        if not isinstance(data, Mapping):
            raise ValueError("GPU记录data必须是对象")  # noqa: TRY004
        payload = copy.deepcopy(dict(data))
        rid = record_id or str(uuid.uuid4())
        effective_project = project_id if project_id is not None else payload.get("project_id")
        workflow_key = workflow_version = None
        if kind == "workflows":
            workflow_key = payload.get("workflow_key")
            workflow_version = payload.get("workflow_version")
            if not isinstance(workflow_key, str) or not workflow_key:
                raise ValueError("workflow_key必填")
            if not (
                (
                    isinstance(workflow_version, int)
                    and not isinstance(workflow_version, bool)
                    and workflow_version > 0
                )
                or (isinstance(workflow_version, str) and workflow_version)
            ):
                raise ValueError("workflow_version格式无效")
        if kind == "files" and effective_project is None:
            storage_key = payload.get("storage_key")
            parts = storage_key.split("/") if isinstance(storage_key, str) else []
            if len(parts) >= 2 and parts[0] == "gpu":
                source_job = self.get("jobs", parts[1], owner_id)
                effective_project = source_job.get("project_id") if source_job else None
        kwargs = {
            "id": rid,
            "owner_id": owner_id,
            "project_id": effective_project,
            "state": payload.get("state"),
            "data": payload,
            "version": 0,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        for field in ("account_id", "instance_id", "current_job_id"):
            if hasattr(model, field) and field in payload:
                kwargs[field] = payload[field]
        if model is GpuWorkflowRow:
            kwargs.update(workflow_key=workflow_key, workflow_version=str(workflow_version))
        if model is GpuJobRow:
            key = payload.get("idempotency_key") or f"__direct__:{rid}"
            kwargs.update(
                idempotency_key=key,
                body_hash=payload.get("body_hash") or digest(payload.get("request_body", payload)),
            )
        try:
            with self.store.db_lock(), self.store.Session.begin() as session:
                row = model(**kwargs)
                session.add(row)
                session.flush()
                return self._row(row)
        except IntegrityError as exc:
            raise ValueError("GPU记录已存在或违反唯一约束") from exc

    def get(self, kind, id, owner_id=None):
        model = self._model(kind)
        with self.store.db_lock(), self.store.Session() as session:
            stmt = select(model).where(model.id == id)
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            return self._row(session.scalar(stmt))

    def list(self, kind, owner_id=None, project_id=None):
        model = self._model(kind)
        with self.store.db_lock(), self.store.Session() as session:
            stmt = select(model).order_by(model.created_at.desc())
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            if project_id is not None:
                stmt = stmt.where(model.project_id == project_id)
            return [self._row(row) for row in session.scalars(stmt).all()]

    def update(self, kind, id, patch, owner_id=None):
        model = self._model(kind)
        if not isinstance(patch, Mapping):
            raise ValueError("patch必须是对象")  # noqa: TRY004
        with self.store.db_lock(), self.store.Session.begin() as session:
            stmt = select(model).where(model.id == id)
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            row = session.scalar(stmt)
            if row is None:
                raise KeyError(id)
            if kind == "workflows" and row.state == "published" and set(patch) - {"active"}:
                raise ValueError("已发布workflow版本不可覆写")
            changes = copy.deepcopy(dict(patch))
            row.data = {**(row.data or {}), **changes}
            for field in ("account_id", "instance_id", "current_job_id"):
                if hasattr(model, field) and field in changes:
                    setattr(row, field, changes[field])
            if "state" in changes:
                row.state = changes["state"]
            if "project_id" in changes:
                row.project_id = changes["project_id"]
            if kind == "workflows":
                if "workflow_key" in changes:
                    if not isinstance(changes["workflow_key"], str) or not changes["workflow_key"]:
                        raise ValueError("workflow_key格式无效")
                    row.workflow_key = changes["workflow_key"]
                if "workflow_version" in changes:
                    value = changes["workflow_version"]
                    if not (
                        (isinstance(value, int) and not isinstance(value, bool) and value > 0)
                        or (isinstance(value, str) and value)
                    ):
                        raise ValueError("workflow_version格式无效")
                    row.workflow_version = str(value)
            row.version += 1
            row.updated_at = utcnow()
            if kind=='jobs' and row.data.get('batch_id'):
                self._refresh_batch(session,row.data['batch_id'],row.owner_id)
            session.flush()
            return self._row(row)

    def create_job(self, owner_id, body, idempotency_key, *, admitted=None):
        if not isinstance(body, Mapping):
            raise ValueError("job body必须是对象")  # noqa: TRY004
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key必填")
        raw = copy.deepcopy(dict(body))
        body_hash = digest(raw)
        for attempt in range(8):
            try:
                with self.store.db_lock(), self.store.Session.begin() as session:
                    existing = session.scalar(
                        select(GpuJobRow).where(
                            GpuJobRow.owner_id == owner_id, GpuJobRow.idempotency_key == idempotency_key
                        )
                    )
                    if existing:
                        if existing.body_hash != body_hash:
                            raise ValueError("同一owner的idempotency_key已用于不同body")
                        return self._row(existing), False
                    policy = raw.get("execution_policy") or {}
                    payload = {
                        **raw,
                        "request_body": raw,
                        "idempotency_key": idempotency_key,
                        "body_hash": body_hash,
                        "state": "queued",
                        "provider": policy.get("provider", "mock"),
                        "workflow_key": raw.get("workflow_key"),
                        "workflow_version": raw.get("workflow_version"),
                        "model_id": raw.get("model_id"),
                        "params": copy.deepcopy(raw.get("params", {})),
                        "resolved_seed": None,
                        "snapshot": None,
                        "max_runtime_minutes": policy.get("max_runtime_minutes"),
                        "estimated_units": None,
                        "unit_rate": None,
                        "requested_outcome": None,
                        "cancel_requested": False,
                        "output_asset_ids": [],
                    }
                    if admitted is not None:
                        payload.update(copy.deepcopy(admitted))
                    row = GpuJobRow(
                        id=str(uuid.uuid4()),
                        owner_id=owner_id,
                        project_id=raw.get("project_id"),
                        state="queued",
                        data=payload,
                        version=0,
                        idempotency_key=idempotency_key,
                        body_hash=body_hash,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                    session.add(row)
                    session.flush()
                    return self._row(row), True
            except IntegrityError:
                if attempt == 7:
                    raise ValueError("GPU job创建竞争失败，请重试")
                time.sleep(0.01 * (attempt + 1))
            except OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 7:
                    raise
                time.sleep(0.02 * (attempt + 1))
        raise RuntimeError("GPU job创建未完成")

    def create_batch(self, owner_id, jobs, idempotency_key, policy, request_body=None):
        for attempt in range(8):
            try:
                return self._create_batch_once(owner_id,jobs,idempotency_key,policy,request_body)
            except IntegrityError:
                if attempt==7:
                    raise ValueError('batch creation contention; retry')
            except OperationalError as exc:
                if 'locked' not in str(exc).lower() or attempt==7:
                    raise
            time.sleep(0.02*(attempt+1))

    def _refresh_batch(self, session, batch_id, owner_id):
        session.flush()
        batch=session.get(GpuBatchRow,batch_id)
        if not batch or batch.owner_id!=owner_id:
            return
        jobs=list(session.scalars(select(GpuJobRow).where(
            GpuJobRow.owner_id==owner_id,GpuJobRow.id.in_(batch.data.get('job_ids',[])))))
        counts={state:sum(job.state==state for job in jobs) for state in ('completed','failed','cancelled')}
        finished=sum(counts.values())
        if jobs and finished==len(jobs):
            state=next((name for name,count in counts.items() if count==len(jobs)),'partial')
        else:
            state='running' if any(job.state!='queued' for job in jobs) else 'queued'
        batch.state=state
        batch.data={**batch.data,'state':state,'finished_jobs':finished,'counts':counts}
        batch.version+=1
        batch.updated_at=utcnow()

    def _create_batch_once(self, owner_id, jobs, idempotency_key, policy, request_body=None):
        if not isinstance(jobs, list) or not 1 <= len(jobs) <= 100:
            raise ValueError("batch requires 1..100 jobs")
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200:
            raise ValueError("invalid idempotency key")
        if policy.get("mode", "save") not in {"save", "smart", "batch"}:
            raise ValueError("invalid lifecycle mode")
        body_hash = digest(request_body if request_body is not None else {"jobs": jobs, "policy": policy})
        with self.store.db_lock(), self.store.Session.begin() as session:
            existing = session.scalar(select(GpuBatchRow).where(
                GpuBatchRow.owner_id == owner_id, GpuBatchRow.idempotency_key == idempotency_key))
            if existing:
                if existing.body_hash != body_hash:
                    raise ValueError("batch idempotency conflict")
                return self._row(existing), False
            batch_id = str(uuid.uuid4())
            payload = {"policy_snapshot": copy.deepcopy(policy), "total_jobs": len(jobs),
                       "finished_jobs": 0, "mode": policy.get("mode", "save"), "job_ids": [],
                       "request_key":idempotency_key,"request_hash":body_hash}
            for index, job in enumerate(jobs):
                job_id = str(uuid.uuid4())
                data = copy.deepcopy(job)
                raw_jobs=(request_body or {}).get('jobs',[])
                data['request_body']=copy.deepcopy(raw_jobs[index] if index<len(raw_jobs) else {
                    k:v for k,v in job.items() if k in {'project_id','model_id','workflow_key','workflow_version','params','asset_ids','execution_policy','preset_id','loras'}})
                data.update(batch_id=batch_id, queue_order=index, state="queued",
                            execution_policy={**data.get("execution_policy", {}), **policy},
                            provider=data.get("execution_policy", {}).get("provider", "mock"))
                if data['provider'] not in {'mock', 'colab', 'kaggle'}:
                    raise ValueError('batch provider is not authorized')
                data['compatibility_signature'] = digest({
                    'provider':data['provider'],
                    'environment':{k:v for k,v in (data.get('snapshot') or {}).items()
                                   if k not in {'resolved_params','bound_graph','input_files'}},
                    'workflow_key':data.get('workflow_key'), 'workflow_version':data.get('workflow_version'),
                    'requested_spec':data.get('requested_spec')})
                row = GpuJobRow(id=job_id, owner_id=owner_id, project_id=data.get('project_id'),
                                state='queued', data=data, version=0,
                                idempotency_key=f'batch:{batch_id}:{index}', body_hash=digest(job),
                                created_at=utcnow(), updated_at=utcnow())
                session.add(row)
                payload['job_ids'].append(job_id)
            batch = GpuBatchRow(id=batch_id, owner_id=owner_id,
                                project_id=jobs[0].get('project_id'), state='queued', version=0,
                                data=payload, idempotency_key=idempotency_key, body_hash=body_hash,
                                created_at=utcnow(), updated_at=utcnow())
            session.add(batch)
            session.flush()
            return self._row(batch), True

    def attach_idle_lease(self, owner_id, job_id, lease_id, now):
        with self.store.db_lock(), self.store.Session.begin() as session:
            job=session.get(GpuJobRow,job_id)
            lease=session.get(GpuLeaseRow,lease_id)
            if not job or not lease or job.owner_id!=owner_id or lease.owner_id!=owner_id:
                return False
            if (lease.state!='idle' or job.state!='waiting_for_capacity' or job.data.get('cancel_requested')
                    or not job.data.get('compatibility_signature')
                    or job.data['compatibility_signature']!=lease.data.get('compatibility_signature')
                    or now>=min(float(lease.data.get('idle_deadline_at') or 0),float(lease.data.get('deadline_epoch') or 0))):
                return False
            lease.state='active'
            lease.data={**lease.data,'state':'active','owner_job_id':job_id,'current_job_id':job_id,
                        'last_active_at':now,'idle_deadline_at':None,
                        **({'handle':retarget_handle(lease.data['handle'],job_id)} if isinstance(lease.data.get('handle'),dict) else {})}
            job.data={**job.data,'lease_id':lease_id,'deadline_epoch':lease.data['deadline_epoch']}
            for row in (lease,job):
                row.version+=1
                row.updated_at=utcnow()
            return True

    def handoff_lease(self, owner_id, job_id, next_job_id, lease_id, now):
        """Transfer a lease under the same admission fence as allocation/bind."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False
            return self._handoff_lease_locked(owner_id, job_id, next_job_id, lease_id, now)

    def _handoff_lease_locked(self, owner_id, job_id, next_job_id, lease_id, now):
        """Finish old job and transfer exclusive lease ownership in one DB transaction."""
        with self.store.db_lock(), self.store.Session.begin() as session:
            job=session.get(GpuJobRow,job_id)
            following=session.get(GpuJobRow,next_job_id)
            lease=session.get(GpuLeaseRow,lease_id)
            def association(row, key):
                data_value = (row.data or {}).get(key)
                typed_value = getattr(row, key, None)
                if data_value not in (None, '') and typed_value not in (None, '') and data_value != typed_value:
                    return False, None
                return True, data_value if data_value not in (None, '') else typed_value

            valid = True
            associations = {}
            for name, row in (("job", job), ("following", following), ("lease", lease)):
                for key in ("account_id", "instance_id"):
                    ok, value = association(row, key) if row is not None else (False, None)
                    valid = valid and ok
                    associations[f"{name}_{key}"] = value
            instance_id = associations.get('job_instance_id') or associations.get('lease_instance_id')
            instance = session.get(GpuInstanceRow, instance_id) if instance_id else None
            if instance is not None:
                for key in ("account_id", "current_job_id"):
                    ok, value = association(instance, key)
                    valid = valid and ok
                    associations[f"instance_{key}"] = value
            if not valid or not all(row is not None and row.owner_id==owner_id for row in (job,following,lease)):
                raise ValueError('lease handoff association conflict')
            if instance_id and (instance is None or instance.owner_id != owner_id):
                raise ValueError('lease handoff instance ownership mismatch')
            if associations.get('job_instance_id') not in {None, instance_id} or associations.get('lease_instance_id') not in {None, instance_id}:
                raise ValueError('lease handoff instance mismatch')
            lease_account = associations.get('lease_account_id')
            if associations.get('job_account_id') not in {None, lease_account} or associations.get('following_account_id') not in {None, lease_account}:
                return False
            if instance is not None and (not associations.get('instance_account_id') or lease_account not in {None, associations['instance_account_id']}):
                return False
            if instance is not None and associations.get('instance_current_job_id') not in {None, job_id}:
                return False
            if job.state!='cleaning_up' or following.state not in {'queued','validating','waiting_for_capacity'}:
                return False
            if following.data.get('lease_id') not in {None,lease_id}:
                return False
            others=session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id==owner_id)).all()
            if any(row.id!=lease_id and row.data.get('owner_job_id')==next_job_id and row.state!='deleted' for row in others):
                return False
            if following.data.get('cancel_requested') or lease.data.get('owner_job_id')!=job_id:
                return False
            lease_account = (lease.data or {}).get('account_id') or lease.account_id
            job_account = (job.data or {}).get('account_id') or job.account_id
            following_account = (following.data or {}).get('account_id') or following.account_id
            if any(value is not None and value != lease_account for value in (job_account, following_account)):
                return False
            instance_data = copy.deepcopy(instance.data or {}) if instance is not None else {}
            account_id = instance_data.get('account_id') or (instance.account_id if instance is not None else None)
            if instance is not None:
                lease_account = (lease.data or {}).get('account_id') or lease.account_id
                job_account = (job.data or {}).get('account_id') or job.account_id
                following_account = (following.data or {}).get('account_id') or following.account_id
                if not account_id or any(value not in {None, account_id} for value in (lease_account, job_account, following_account)):
                    return False
                if instance.current_job_id not in {None, job_id}:
                    return False
                if instance.state in {'deleted', 'unknown', 'releasing', 'cleanup_pending', 'quarantined'}:
                    return False
            if (job.data.get('compatibility_signature')!=following.data.get('compatibility_signature')
                    or not job.data.get('compatibility_signature')
                    or lease.state not in {'active','busy'}
                    or now>=float(lease.data.get('deadline_epoch') or 0)):
                return False
            execution_id = job.data.get('execution_id')
            execution=session.get(GpuExecutionRow, execution_id) if execution_id else None
            next_execution_id = following.data.get('execution_id')
            next_execution = session.get(GpuExecutionRow, next_execution_id) if next_execution_id else None
            def execution_matches(execution, expected_job_id):
                if execution is None or execution.owner_id != owner_id:
                    return False
                data = execution.data or {}
                for key, expected in (("job_id", expected_job_id), ("lease_id", lease_id),
                                      ("instance_id", instance_id), ("account_id", associations.get('instance_account_id') or lease_account)):
                    json_value = data.get(key)
                    typed_value = getattr(execution, key, None)
                    if json_value not in (None, '') and typed_value not in (None, '') and json_value != typed_value:
                        return False
                    value = json_value if json_value not in (None, '') else typed_value
                    if value not in (None, '') and value != expected:
                        return False
                return True

            if execution is not None and not execution_matches(execution, job_id):
                return False
            if next_execution is not None and not execution_matches(next_execution, next_job_id):
                return False
            job.data={**job.data,'state':'completed','runtime_deleted':False,'stopped_at':now,
                      **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {})}
            job.state='completed'
            if instance is not None:
                job.instance_id = instance.id
                job.account_id = account_id
            following.data={**following.data,'lease_id':lease_id,
                            'deadline_epoch':lease.data['deadline_epoch'],
                            **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {})}
            if instance is not None:
                following.instance_id = instance.id
                following.account_id = account_id
            handle=lease.data.get('handle')
            # Handle ownership must follow the lease owner for every provider (real ones verify it).
            lease.data={**lease.data,'owner_job_id':next_job_id,'current_job_id':next_job_id,
                        'last_active_at':now,'state':'active',
                        **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {}),
                        **({'handle':retarget_handle(handle,next_job_id)} if isinstance(handle,dict) else {})}
            lease.state='active'
            if instance is not None:
                lease.instance_id = instance.id
                lease.account_id = account_id
                instance_data.update(current_job_id=next_job_id, instance_id=instance.id, account_id=account_id,
                                     lease_id=lease_id, last_seen_at=now, runtime_deleted=False, state='allocating')
                instance.data = instance_data
                instance.current_job_id = next_job_id
                instance.account_id = account_id
                instance.state = 'allocating'
            execution_id = job.data.get('execution_id')
            execution=session.get(GpuExecutionRow, execution_id) if execution_id else None
            if execution is not None:
                execution.state='completed'
                execution.data={**execution.data,'state':'completed','job_id':job_id,'lease_id':lease_id,
                                'finished_at':now,
                                **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {})}
                if instance is not None:
                    execution.instance_id=instance.id
                    execution.account_id=account_id
                execution.updated_at=utcnow()
                execution.version += 1
            next_execution_id = following.data.get('execution_id')
            next_execution = session.get(GpuExecutionRow, next_execution_id) if next_execution_id else None
            if next_execution is not None:
                next_execution.data={**next_execution.data,'job_id':next_job_id,'lease_id':lease_id,
                                     **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {})}
                if instance is not None:
                    next_execution.instance_id=instance.id
                    next_execution.account_id=account_id
                next_execution.updated_at=utcnow()
                next_execution.version += 1
            for row in (job,following,lease):
                row.version+=1
                row.updated_at=utcnow()
            if instance is not None:
                instance.version += 1
                instance.updated_at=utcnow()
            if job.data.get('batch_id'):
                self._refresh_batch(session,job.data['batch_id'],owner_id)
            return True

    @staticmethod
    def _cleanup_blocking(state):
        return state in {'creating', 'active', 'busy', 'idle', 'cleanup_pending', 'quarantined', 'releasing', 'deleting'}

    @classmethod
    def _claim_is_live(cls, data, now):
        claimed_at = data.get('cleanup_claimed_at')
        try:
            return claimed_at is not None and float(now) - float(claimed_at) < cls._CLEANUP_CLAIM_TTL_SECONDS
        except (TypeError, ValueError):
            return True

    def claim_orphan_lease(self, owner_id, lease_id, instance_id, token, now):
        """Atomically fence an orphan lease and its instance before provider I/O.

        Admission is intentionally used here instead of dispatch: allocation and
        lease binding take the same short admission lock, while the provider call
        happens after this transaction commits.
        """
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return None
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id))
                lease_instance_json = (lease.data or {}).get('instance_id') if lease is not None else None
                lease_instance_typed = lease.instance_id if lease is not None else None
                if (lease_instance_json not in (None, '') and lease_instance_typed not in (None, '')
                        and lease_instance_json != lease_instance_typed):
                    return None
                lease_instance_id = lease_instance_json or lease_instance_typed
                if lease is None or lease.state == 'deleted' or lease_instance_id != instance_id:
                    return None
                lease_data = lease.data or {}
                if lease_data.get('cleanup_claim_token') not in {None, token} and self._claim_is_live(lease_data, now):
                    return None
                others = session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all()
                if any(
                    row.id != lease_id
                    and ((row.data or {}).get('instance_id') or row.instance_id) == instance_id
                    and self._cleanup_blocking(row.state)
                    for row in others
                ):
                    return None
                instance = session.scalar(select(GpuInstanceRow).where(
                    GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                ))
                if instance is None:
                    return None
                instance_data = copy.deepcopy(instance.data or {})
                instance_account_json = instance_data.get('account_id')
                instance_account_typed = instance.account_id
                lease_account_json = lease_data.get('account_id')
                lease_account_typed = lease.account_id
                if (instance_account_json in (None, '') or instance_account_typed in (None, '')
                        or lease_account_json in (None, '') or lease_account_typed in (None, '')
                        or instance_account_json != instance_account_typed
                        or lease_account_json != lease_account_typed
                        or lease_account_json != instance_account_json):
                    return None
                current_job = instance_data.get('current_job_id') or instance.current_job_id
                if current_job or instance.state in {'allocating', 'ready', 'busy', 'retained_idle', 'deleted', 'unknown'}:
                    return None
                if instance_data.get('cleanup_claim_token') not in {None, token} and self._claim_is_live(instance_data, now):
                    return None
                instance_data.update(
                    cleanup_claim_token=token,
                    cleanup_claimed_at=now,
                    runtime_deleted=False,
                )
                instance.state = 'cleanup_pending'
                instance.version += 1
                instance.updated_at = utcnow()
                instance.data = instance_data
                lease_data = copy.deepcopy(lease.data or {})
                lease_data.update(cleanup_claim_token=token, cleanup_claimed_at=now, state='deleting')
                lease.data = lease_data
                lease.state = 'deleting'
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return self._row(lease), self._row(instance)

    def finish_orphan_lease(self, owner_id, lease_id, instance_id, token, now, *, confirmed, error=None, attempts=0):
        """Commit or release a previously fenced orphan cleanup claim."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id))
                lease_instance_id = (lease.data or {}).get('instance_id') if lease is not None else None
                lease_instance_id = lease_instance_id or (lease.instance_id if lease is not None else None)
                if lease is None or lease_instance_id != instance_id:
                    return False
                data = copy.deepcopy(lease.data or {})
                if data.get('cleanup_claim_token') != token or lease.state != 'deleting':
                    return False
                if confirmed:
                    others = session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all()
                    if any(
                        row.id != lease_id
                        and ((row.data or {}).get('instance_id') or row.instance_id) == instance_id
                        and self._cleanup_blocking(row.state)
                        for row in others
                    ):
                        return False
                    data.update(
                        state='deleted', last_seen=now, cleanup_claim_token=None,
                        remote_deleted=True, remote_deleted_at=now,
                        remote_delete_source='orphan_cleanup_provider_confirmation',
                    )
                    lease.state = 'deleted'
                else:
                    if isinstance(error, dict):
                        try:
                            delay = max(0.0, float(error.get("retry_after")))
                        except (TypeError, ValueError):
                            delay = min(
                                self._CLEANUP_RETRY_MAX_SECONDS,
                                self._CLEANUP_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)),
                            )
                        last_error = error.get("message") or error.get("error") or str(error)
                    else:
                        delay = min(
                            self._CLEANUP_RETRY_MAX_SECONDS,
                            self._CLEANUP_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)),
                        )
                        last_error = error or "delete not confirmed"
                    data.update(
                        state="cleanup_pending", delete_attempts=attempts,
                        cleanup_next_epoch=now + delay, last_error=last_error,
                        last_seen=now, cleanup_claim_token=None,
                    )
                    lease.state = "cleanup_pending"
                    instance = session.scalar(select(GpuInstanceRow).where(
                        GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                    ))
                    if instance is not None:
                        instance_data = copy.deepcopy(instance.data or {})
                        if instance_data.get("cleanup_claim_token") == token:
                            instance_data.update(cleanup_claim_token=None, cleanup_claimed_at=None)
                            instance.data = instance_data
                            instance.version += 1
                            instance.updated_at = utcnow()
                lease.data = data
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return True

    def claim_orphan_instance(self, owner_id, instance_id, lease_id, token, now):
        """Fence a historical instance before checking a deleted lease remotely.

        Old handoff rows may have a deleted lease without the provenance marker
        written by newer cleanup code.  The short admission transaction only
        claims a fully owned, terminal, unambiguous instance; provider I/O is
        deliberately performed after it commits.
        """
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return None
            with self.store.db_lock(), self.store.Session.begin() as session:
                instance = session.scalar(select(GpuInstanceRow).where(
                    GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                ))
                lease = session.scalar(select(GpuLeaseRow).where(
                    GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id
                ))
                if instance is None or lease is None or instance.state not in {"releasing", "cleanup_pending"}:
                    return None
                if lease.state != "deleted":
                    return None
                instance_data = copy.deepcopy(instance.data or {})
                lease_data = copy.deepcopy(lease.data or {})

                def identity(row, data, key):
                    json_value = data.get(key)
                    typed_value = getattr(row, key, None)
                    if json_value not in (None, "") and typed_value not in (None, "") and json_value != typed_value:
                        return None, False
                    return (json_value if json_value not in (None, "") else typed_value), True

                instance_identity = {}
                lease_identity = {}
                for key in ("account_id", "instance_id", "current_job_id"):
                    value, valid = identity(instance, instance_data, key)
                    if not valid:
                        return None
                    instance_identity[key] = value
                for key in ("account_id", "instance_id"):
                    value, valid = identity(lease, lease_data, key)
                    if not valid:
                        return None
                    lease_identity[key] = value
                if (
                    not instance_identity["account_id"]
                    or instance_identity["account_id"] != lease_identity["account_id"]
                    or instance_identity["instance_id"] not in (None, instance_id)
                    or lease_data.get("provider") not in {"mock", "colab", "kaggle"}
                    or instance_data.get("provider") not in {None, lease_data.get("provider")}
                    or not lease_data.get("runtime_id")
                    or not lease_data.get("request_id")
                ):
                    return None
                if lease_data.get("remote_delete_source") not in (None, ""):
                    return None
                for row_data in (instance_data, lease_data):
                    retry_at = row_data.get("cleanup_next_epoch")
                    if retry_at is not None:
                        try:
                            if float(now) < float(retry_at):
                                return None
                        except (TypeError, ValueError):
                            return None
                current_job_id = instance_identity["current_job_id"]
                owner_job_id = lease_data.get("owner_job_id")
                if not owner_job_id:
                    return None
                jobs = session.scalars(select(GpuJobRow).where(GpuJobRow.owner_id == owner_id)).all()
                by_id = {row.id: row for row in jobs}
                related = [row for row in jobs if row.id in {current_job_id, owner_job_id}
                           or ((row.data or {}).get("instance_id") or row.instance_id) == instance_id]
                if any(row.state not in {"completed", "failed", "cancelled"} for row in related):
                    return None
                provider = lease_data.get("provider")
                owner_job = by_id.get(owner_job_id)
                if owner_job is None or owner_job.state not in {"completed", "failed", "cancelled"}:
                    return None
                lease_instance_id = lease_identity["instance_id"]
                owner_job_data = owner_job.data or {}
                owner_job_instance = owner_job_data.get("instance_id") or owner_job.instance_id
                if lease_instance_id not in (None, instance_id):
                    # A completed handoff may have rebound this lease to the
                    # successor instance.  Accept it only when the lease and
                    # its current owner job agree exactly on that successor.
                    if owner_job_instance != lease_instance_id or lease_instance_id == instance_id:
                        return None
                for row in related:
                    row_data = row.data or {}
                    row_account = row_data.get("account_id") or row.account_id
                    row_instance = row_data.get("instance_id") or row.instance_id
                    row_provider = row_data.get("provider") or (row_data.get("execution_policy") or {}).get("provider")
                    allowed_instances = {None, instance_id}
                    if row.id == owner_job_id and lease_instance_id not in (None, instance_id):
                        allowed_instances.add(lease_instance_id)
                    if (
                        row_account not in (None, instance_identity["account_id"])
                        or row_instance not in allowed_instances
                        or row_provider not in (None, provider)
                    ):
                        return None
                if current_job_id and current_job_id not in by_id:
                    return None
                leases = session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all()
                runtime_id = lease_data.get("runtime_id")
                if any(
                    row.id != lease_id
                    and row.state != "deleted"
                    and (
                        ((row.data or {}).get("instance_id") or row.instance_id) == instance_id
                        or (row.data or {}).get("runtime_id") == runtime_id
                    )
                    for row in leases
                ):
                    return None
                claimed = instance_data.get("cleanup_claim_token")
                if claimed not in (None, token) and self._claim_is_live(instance_data, now):
                    return None
                instance_data.update(
                    cleanup_claim_token=token,
                    cleanup_claimed_at=now,
                    cleanup_claim_lease_id=lease_id,
                    cleanup_claim_instance_id=instance_id,
                    cleanup_claim_current_job_id=current_job_id,
                    runtime_deleted=False,
                )
                instance.state = "cleanup_pending"
                instance.data = instance_data
                instance.version += 1
                instance.updated_at = utcnow()
                lease_data.update(
                    cleanup_claim_token=token,
                    cleanup_claimed_at=now,
                    cleanup_claim_lease_id=lease_id,
                    cleanup_claim_instance_id=lease_identity["instance_id"],
                    cleanup_claim_owner_job_id=owner_job_id,
                    cleanup_claim_runtime_id=runtime_id,
                )
                lease.data = lease_data
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return self._row(instance), self._row(lease)

    def finish_orphan_instance(
        self, owner_id, instance_id, lease_id, token, now, *, confirmed, error=None, expected_version=None,
        expected_lease_version=None
    ):
        """Persist or release a fenced historical-instance reconciliation."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False
            with self.store.db_lock(), self.store.Session.begin() as session:
                instance = session.scalar(select(GpuInstanceRow).where(
                    GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                ))
                lease = session.scalar(select(GpuLeaseRow).where(
                    GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id
                ))
                if instance is None or lease is None or lease.state != "deleted":
                    return False
                if expected_lease_version is None or lease.version != expected_lease_version:
                    return False
                if expected_version is None or instance.version != expected_version or instance.state != "cleanup_pending":
                    return False
                data = copy.deepcopy(instance.data or {})
                if (
                    data.get("cleanup_claim_token") != token
                    or data.get("cleanup_claim_lease_id") != lease_id
                    or data.get("cleanup_claim_instance_id") != instance_id
                ):
                    return False
                lease_data = copy.deepcopy(lease.data or {})
                if (
                    lease_data.get("cleanup_claim_token") != token
                    or lease_data.get("cleanup_claim_lease_id") != lease_id
                    or lease_data.get("cleanup_claim_owner_job_id") != lease_data.get("owner_job_id")
                    or lease_data.get("cleanup_claim_runtime_id") != lease_data.get("runtime_id")
                ):
                    return False
                lease_instance_json = lease_data.get("instance_id")
                lease_instance_typed = lease.instance_id
                if (
                    lease_instance_json not in (None, "")
                    and lease_instance_typed not in (None, "")
                    and lease_instance_json != lease_instance_typed
                ):
                    return False
                if (lease_instance_json or lease_instance_typed) != lease_data.get("cleanup_claim_instance_id"):
                    return False
                owner_job_id = lease_data.get("owner_job_id")
                owner_job = session.scalar(select(GpuJobRow).where(
                    GpuJobRow.id == owner_job_id, GpuJobRow.owner_id == owner_id
                ))
                if owner_job is None or owner_job.state not in {"completed", "failed", "cancelled"}:
                    return False
                claimed_current_job = data.get("cleanup_claim_current_job_id")
                current_job_id = data.get("current_job_id")
                if current_job_id != claimed_current_job:
                    return False
                if current_job_id:
                    current_job = session.scalar(select(GpuJobRow).where(
                        GpuJobRow.id == current_job_id, GpuJobRow.owner_id == owner_id
                    ))
                    if current_job is None or current_job.state not in {"completed", "failed", "cancelled"}:
                        return False
                if confirmed:
                    leases = session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all()
                    runtime_id = lease_data.get("runtime_id")
                    if any(
                        row.id != lease_id
                        and row.state != "deleted"
                        and (
                            ((row.data or {}).get("instance_id") or row.instance_id) == instance_id
                            or (row.data or {}).get("runtime_id") == runtime_id
                        )
                        for row in leases
                    ):
                        return False
                    data.update(
                        current_job_id=None,
                        runtime_deleted=True,
                        remote_deleted=True,
                        remote_deleted_at=now,
                        remote_delete_source="orphan_reconciliation_provider_confirmation",
                        retention_reserved=0.0,
                        deleted_at=now,
                        deletion_confirmed_at=now,
                        last_seen_at=now,
                        cleanup_claim_token=None,
                        cleanup_claimed_at=None,
                        cleanup_claim_lease_id=None,
                        cleanup_claim_instance_id=None,
                        cleanup_claim_current_job_id=None,
                    )
                    instance.state = "deleted"
                    instance.current_job_id = None
                else:
                    data.update(
                        cleanup_claim_token=None,
                        cleanup_claimed_at=None,
                        cleanup_next_epoch=now + self._CLEANUP_RETRY_BASE_SECONDS,
                        last_error=error or "remote deletion not confirmed",
                        last_seen_at=now,
                        cleanup_claim_lease_id=None,
                        cleanup_claim_instance_id=None,
                        cleanup_claim_current_job_id=None,
                    )
                    instance.state = "cleanup_pending"
                instance.data = data
                instance.version += 1
                instance.updated_at = utcnow()
                lease_data.pop("cleanup_claim_token", None)
                lease_data.pop("cleanup_claimed_at", None)
                lease_data.pop("cleanup_claim_lease_id", None)
                lease_data.pop("cleanup_claim_instance_id", None)
                lease_data.pop("cleanup_claim_owner_job_id", None)
                lease_data.pop("cleanup_claim_runtime_id", None)
                if confirmed:
                    lease_data.update(
                        remote_deleted=True,
                        remote_deleted_at=now,
                        remote_delete_source="orphan_reconciliation_provider_confirmation",
                    )
                lease.data = lease_data
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return True

    def record_usage_once(self, owner_id, payload):
        # Deterministic primary key provides DB-enforced dedupe across processes.
        identity=digest([owner_id,payload['job_id'],payload.get('execution_id')])
        try:
            return self.create('usage',owner_id,payload,record_id=identity)
        except (IntegrityError, ValueError):
            existing=self.get('usage',identity,owner_id)
            if existing is None:
                raise
            return existing

    def retry_attempt(self, job_id, owner_id, key):
        """Requeue a cleaned failed Mock job; immutable snapshot and executions survive."""
        if not key or len(key)>200:
            raise ValueError('idempotency key required')
        with self.store.db_lock(), self.store.Session.begin() as session:
            job=session.scalar(select(GpuJobRow).where(GpuJobRow.id==job_id,GpuJobRow.owner_id==owner_id).with_for_update())
            if not job:
                raise KeyError(job_id)
            data=copy.deepcopy(job.data)
            keys=data.get('attempt_retry_keys') or []
            if key in keys:
                return self._row(job)
            if job.state not in {'failed','cancelled'}:
                raise ValueError('job must be failed or cancelled')
            if data.get('provider',(data.get('execution_policy') or {}).get('provider','mock'))!='mock':
                raise ValueError('same-job retry is currently Mock only')
            snapshot=data.get('snapshot') or {}
            if data.get('prep') or any(snapshot.get(k) or data.get(k) for k in ('system_smoke','workflow_smoke','prep')) or (data.get('request_body') or {}).get('kind'):
                raise ValueError('special job requires its confirmation entry')
            leases=session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id==owner_id)).all()
            if any(row.data.get('owner_job_id')==job_id and row.state!='deleted' for row in leases):
                raise ValueError('lease cleanup must be confirmed before retry')
            history=data.get('attempt_history') or []
            history.append({k:copy.deepcopy(v) for k,v in data.items() if k not in {'snapshot','request_body','attempt_history','attempt_retry_keys'}})
            transient={
                'execution_id','lease_id','outputs','output_asset_ids','persisted_outputs','error_code','error_message',
                'cancel_requested','requested_outcome','cleanup_final_state','cleanup_attempts','cleanup_next_epoch',
                'upload_attempts','started_epoch','stopped_at','deadline_epoch','validated_epoch','budget_reserved',
                'runtime_deleted','actual_units','prep_report','progress','awaiting_prep',
                'instance_completion_recorded','instance_id','instance_state','instance_bound','account_id',
                'requested_instance_id',
                'account_alias_snapshot','selection_reason','allocation_key','attempt_id','runtime_id','request_id',
                'stage','worker_timings','output_bytes','started_at','finished_at','poll_retry_count','poll_retry_at',
            }
            for k in transient:
                data.pop(k,None)
            policy=data.get('execution_policy')
            if isinstance(policy, Mapping) and ('instance_id' in policy or 'requested_instance_id' in policy):
                data['execution_policy']={**policy,'instance_id':None,'requested_instance_id':None}
            data.update(state='queued',attempt_history=history,attempt_retry_keys=[*keys,key],attempt_id=key)
            job.data=data
            job.state='queued'
            job.instance_id = None
            job.account_id = None
            job.version+=1
            job.updated_at=utcnow()
            session.add(GpuEventRow(id=str(uuid.uuid4()),owner_id=owner_id,project_id=job.project_id,state='queued',
                version=0,created_at=utcnow(),updated_at=utcnow(),data={'job_id':job_id,'state':'queued','message':'Mock retry attempt queued; previous evidence retained'}))
            if data.get('batch_id'):
                self._refresh_batch(session,data['batch_id'],owner_id)
            session.flush()
            return self._row(job)

    def prepare_execution(self, job_id, owner_id, lease_id, now):
        """Persist execution identity before start, atomically with its attempt record."""
        with self.store.db_lock(), self.store.Session.begin() as session:
            job=session.scalar(select(GpuJobRow).where(GpuJobRow.id==job_id,GpuJobRow.owner_id==owner_id).with_for_update())
            lease=session.get(GpuLeaseRow,lease_id)
            if not job or not lease or lease.owner_id!=owner_id:
                raise KeyError(job_id)
            if job.state!='loading':
                raise ValueError('job is not ready to start')
            lease_data = lease.data or {}
            lease_owner_job = lease_data.get('owner_job_id')
            if lease_owner_job is not None and lease_owner_job != job_id:
                raise ValueError('lease does not belong to job')
            if job.data.get('lease_id') not in {None, lease_id}:
                raise ValueError('job lease identity mismatch')
            if lease.state not in {'creating', 'active', 'busy'}:
                raise ValueError('lease is not active')
            job_account = job.data.get('account_id') or job.account_id
            lease_account = lease_data.get('account_id') or lease.account_id
            if job_account is not None and lease_account is not None and job_account != lease_account:
                raise ValueError('lease account mismatch')
            job_instance = job.data.get('instance_id') or job.instance_id
            lease_instance = lease_data.get('instance_id') or lease.instance_id
            if job_instance is not None and lease_instance is not None and job_instance != lease_instance:
                raise ValueError('lease instance mismatch')
            existing_execution_id = job.data.get('execution_id')
            if existing_execution_id:
                execution = session.get(GpuExecutionRow, existing_execution_id)
                if execution is None or execution.owner_id != owner_id:
                    raise ValueError('execution association mismatch')
                expected = {'job_id': job_id, 'lease_id': lease_id,
                            'instance_id': job_instance or lease_instance,
                            'account_id': job_account or lease_account}
                for key, expected_value in expected.items():
                    json_value = (execution.data or {}).get(key)
                    typed_value = getattr(execution, key, None)
                    if json_value not in (None, '') and typed_value not in (None, '') and json_value != typed_value:
                        raise ValueError('execution association conflict')
                    value = json_value if json_value not in (None, '') else typed_value
                    if expected_value not in (None, '') and value != expected_value:
                        raise ValueError('execution association mismatch')
                return self._row(job)
            execution_id=str(uuid.uuid4())
            attempt=int(job.data.get('attempt_number') or 0)+1
            job.data={**job.data,'execution_id':execution_id,'attempt_number':attempt}
            job.version+=1
            job.updated_at=utcnow()
            lease_data = lease.data or {}
            instance_id = job.data.get('instance_id') or lease_data.get('instance_id') or lease.instance_id
            account_id = job.data.get('account_id') or lease_data.get('account_id') or lease.account_id
            session.add(GpuExecutionRow(id=execution_id,owner_id=owner_id,project_id=job.project_id,
                account_id=account_id,instance_id=instance_id,
                state='starting',version=0,created_at=utcnow(),updated_at=utcnow(),
                data={'job_id':job_id,'lease_id':lease_id,'attempt':attempt,
                      'worker_session_id':lease_id,'started_at':now,'state':'starting',
                      'instance_id':instance_id,'account_id':account_id}))
            session.flush()
            return self._row(job)

    def transition_job(self, job_id, owner_id, state, message, patch, now, expected_state=None, expected_execution_id=...):
        """Commit job, execution, batch aggregate and audit event together."""
        terminal={'completed','failed','cancelled'}
        progress=patch.get('progress')
        if progress is not None and (isinstance(progress,bool) or not isinstance(progress,(int,float)) or not 0<=progress<=1):
            raise ValueError('progress必须在0到1之间')
        with self.store.db_lock(), self.store.Session.begin() as session:
            job=session.scalar(select(GpuJobRow).where(GpuJobRow.id==job_id,GpuJobRow.owner_id==owner_id).with_for_update())
            if not job:
                raise KeyError(job_id)
            if expected_state is not None and job.state!=expected_state:
                return self._row(job)
            if expected_execution_id is not ... and job.data.get('execution_id')!=expected_execution_id:
                return self._row(job)
            if job.state in terminal:
                return self._row(job)
            job.state=state
            job.data={**job.data,**copy.deepcopy(patch),'state':state}
            job.version+=1
            job.updated_at=utcnow()
            execution_id=job.data.get('execution_id')
            if execution_id and state in terminal|{'running','uploading'}:
                execution=session.get(GpuExecutionRow,execution_id)
                if execution and execution.owner_id!=owner_id:
                    raise ValueError('execution owner mismatch')
                if execution and execution.data.get('job_id') not in {None,job_id}:
                    raise ValueError('execution job identity mismatch')
                if execution:
                    changes={'state':state}
                    if state in terminal:
                        changes.update(finished_at=now,error_code=job.data.get('error_code'))
                    execution.state=state
                    execution.data={**execution.data,**changes}
                    execution.version+=1
                    execution.updated_at=utcnow()
            data={'job_id':job_id,'state':state,'message':message}
            if progress is not None:
                data['progress']=progress
            session.add(GpuEventRow(id=str(uuid.uuid4()),owner_id=owner_id,
                project_id=job.project_id,state=state,data=data,version=0,
                created_at=utcnow(),updated_at=utcnow()))
            if job.data.get('batch_id'):
                self._refresh_batch(session,job.data['batch_id'],owner_id)
            session.flush()
            return self._row(job)

    def event(self, job_id, owner_id, state, message, progress=None):
        job = self.get("jobs", job_id, owner_id)
        if job is None:
            raise KeyError(job_id)
        if progress is not None and (
            isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0 <= progress <= 1
        ):
            raise ValueError("progress必须在0到1之间")
        data = {"job_id": job_id, "message": message, "state": state}
        if progress is not None:
            data["progress"] = progress
        return self.create("events", owner_id, data, project_id=job.get("project_id"))
