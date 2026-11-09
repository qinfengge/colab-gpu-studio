export type V020Provider = 'mock' | 'colab' | 'kaggle';

export type GpuAllocation = 'auto' | 'existing';
export type GpuLifecycle = 'save' | 'batch' | 'smart' | 'retain';
export type RetentionMode = 'save' | 'retain' | 'smart';

/** New execution policy fields are optional so legacy job payloads remain readable. */
export interface GpuExecutionPolicyV020 {
  provider: V020Provider;
  max_runtime_minutes: number;
  allocation?: GpuAllocation;
  account_id?: string | null;
  instance_id?: string | null;
  lifecycle?: GpuLifecycle;
  retain_idle_minutes?: number | null;
  /** Legacy fields: when present, the server validates them against lifecycle. */
  shutdown_after_job?: boolean;
  idle_grace_minutes?: number;
}

export interface GpuJobRequestV020 {
  project_id: string;
  model_id: string;
  workflow_key: string;
  workflow_version: string | number;
  params: Record<string, unknown>;
  asset_ids: string[];
  preset_id?: string;
  execution_policy: GpuExecutionPolicyV020;
}

export type GoogleAuthPurpose = 'colab' | 'drive';
export type GoogleAccountState = 'connected' | 'needs_reauth' | 'disabled' | 'error' | string;

export interface GoogleAuthorization {
  connected: boolean;
  state?: GoogleAccountState;
  scopes?: string[];
  expires_at?: string | null;
  last_checked_at?: string | null;
  error_code?: string | null;
}

export interface GoogleAccount {
  account_id: string;
  alias: string;
  state: GoogleAccountState;
  enabled: boolean;
  email_masked?: string | null;
  drive_account_id?: string | null;
  drive_account_alias?: string | null;
  colab: GoogleAuthorization;
  drive: GoogleAuthorization;
  allowed_project_ids?: string[];
  scheduling_policy?: string | null;
  budget?: Record<string, number | null>;
  active_instance_count?: number;
  last_checked_at?: string | null;
  last_error_code?: string | null;
  can_allocate?: boolean;
  /** Compatibility for older account presenters that used id. */
  id?: string;
}

export interface GoogleAccountCreateRequest {
  alias: string;
  enabled?: boolean;
  drive_account_id?: string | null;
  allowed_project_ids?: string[];
  scheduling_policy?: string;
  budget?: Record<string, number | null>;
}

export interface GoogleAccountUpdateRequest {
  alias?: string;
  enabled?: boolean;
  drive_account_id?: string | null;
  allowed_project_ids?: string[];
  scheduling_policy?: string;
  budget?: Record<string, number | null>;
}

export interface AccountRuntime {
  runtime_id: string;
  instance_id?: string | null;
  state?: string;
  provider?: V020Provider;
  gpu_reported?: string | null;
  created_at?: string | null;
  last_seen_at?: string | null;
  current_job_id?: string | null;
}

export type GpuInstanceState =
  | 'requested' | 'allocating' | 'ready' | 'busy' | 'retained_idle'
  | 'releasing' | 'deleted' | 'unknown' | 'cleanup_pending' | 'quarantined' | string;

export interface GpuInstance {
  instance_id: string;
  label?: string | null;
  account_id?: string | null;
  account_alias_snapshot?: string | null;
  provider?: V020Provider;
  state: GpuInstanceState;
  retention_mode?: RetentionMode;
  retention?: RetentionMode;
  version?: number;
  idle_minutes?: number | null;
  retain_requested_at?: string | null;
  idle_deadline_at?: string | null;
  hard_deadline_at?: string | null;
  manual_stop_requested?: boolean;
  current_job_id?: string | null;
  gpu_reported?: string | null;
  runtime_spec_requested?: Record<string, unknown> | null;
  runtime_spec_reported?: Record<string, unknown> | null;
  created_at?: string;
  last_seen_at?: string | null;
  retained_until?: string | null;
  estimated_units?: number | null;
  actual_units?: number | null;
  runtime_deleted?: boolean;
  last_error_code?: string | null;
}

export interface InstanceRetentionRequest {
  retention_mode: RetentionMode;
  idle_minutes: number;
  expected_version: number;
}

export interface GpuInstanceFilters {
  state?: GpuInstanceState;
  account_id?: string;
  provider?: V020Provider;
}

export type WorkbenchAction =
  | 'health'
  | 'list_allowed_workflows'
  | 'get_reviewed_graph'
  | 'set_reviewed_params'
  | 'queue_reviewed_workflow'
  | 'job_status'
  | 'read_safe_log'
  | 'preview_artifacts'
  | 'stop_current_job';

export interface Workbench {
  instance_id: string;
  state?: string;
  healthy?: boolean;
  service_state?: string;
  capabilities: WorkbenchAction[];
  allowed_workflows?: Array<Record<string, unknown>>;
  cache?: Record<string, unknown>;
  current_job_id?: string | null;
  last_checked_at?: string | null;
  message?: string | null;
}

export interface WorkbenchActionRequest {
  action: WorkbenchAction;
  params?: Record<string, unknown>;
}

export interface WorkbenchActionResult {
  action?: WorkbenchAction;
  state?: string;
  data?: unknown;
  message?: string;
  [key: string]: unknown;
}

/** Safe, user-visible job fields for v0.2.0; provider handles and tokens are never typed here. */
export interface GpuJobSafetyFields {
  account_id?: string | null;
  account_alias_snapshot?: string | null;
  instance_id?: string | null;
  instance_label?: string | null;
  instance_state?: GpuInstanceState | null;
  gpu_reported?: string | null;
  retained_until?: string | null;
  runtime_deleted?: boolean | null;
  stage?: string | null;
  actual_units?: number | null;
}
