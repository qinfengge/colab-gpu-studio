import type {
  AccountRuntime,
  GoogleAccount,
  GoogleAccountCreateRequest,
  GoogleAccountUpdateRequest,
  GoogleAuthPurpose,
  GpuExecutionPolicyV020,
  GpuInstance,
  GpuInstanceFilters,
  GpuJobRequestV020,
  InstanceRetentionRequest,
  RetentionMode,
  Workbench,
  WorkbenchAction,
  WorkbenchActionRequest,
  WorkbenchActionResult,
} from './v020-types';

export type {
  AccountRuntime,
  GoogleAccount,
  GoogleAccountCreateRequest,
  GoogleAccountUpdateRequest,
  GoogleAuthPurpose,
  GpuInstance,
  GpuInstanceFilters,
  InstanceRetentionRequest,
  RetentionMode,
  Workbench,
  WorkbenchAction,
  WorkbenchActionRequest,
  WorkbenchActionResult,
} from './v020-types';

export type Provider = 'mock' | 'colab' | 'kaggle';
export type ParamType = 'string' | 'text' | 'integer' | 'number' | 'boolean' | 'image' | 'video' | 'file' | string;

export interface ApiEnvelope<T> { data?: T; error?: { code?: string; message?: string; detail?: unknown } | string | null; request_id?: string; }
export interface Project { id: string; title: string; [key: string]: unknown; }
export interface GpuModel { model_id: string; display_name: string; enabled: boolean; reason?: string; [key: string]: unknown; }
export interface GpuOption { value: string | number | boolean; label?: string; }
export interface GpuParameter { id?: string; key?: string; name?: string; label?: string; type: ParamType; widget?: string; options?: Array<GpuOption | string | number | boolean>; default?: unknown; min?: number; max?: number; step?: number; group?: string; advanced?: boolean; required?: boolean; description?: string; [key: string]: unknown; }
export interface GpuManifest { model_id?: string; title?: string; description?: string; parameters?: GpuParameter[] | Record<string, Omit<GpuParameter, 'id'>>; [key: string]: unknown; }
export interface GpuWorkflow { id: string; workflow_key: string; workflow_version: string | number; model_id: string; state: string; active?: boolean; manifest?: GpuManifest; title?: string; description?: string; [key: string]: unknown; }
export interface GpuPreset { id: string; name: string; workflow_version_id: string; overrides: Record<string, unknown>; [key: string]: unknown; }
export type GpuExecutionPolicy = GpuExecutionPolicyV020;
export interface GpuJobRequest extends GpuJobRequestV020 { }
export interface GpuJob {
  id: string;
  job_id?: string;
  project_id: string;
  model_id?: string;
  workflow_key?: string;
  workflow_version?: string | number;
  state?: string;
  provider?: Provider;
  params?: Record<string, unknown>;
  resolved_params?: Record<string, unknown>;
  asset_ids?: string[];
  output_asset_ids?: string[];
  error?: string;
  error_message?: string;
  message?: string;
  progress?: number;
  idempotency_key?: string;
  estimated_units?: number | null;
  actual_units?: number | null;
  actual_cu?: number | null;
  unit_rate?: number | null;
  account_id?: string | null;
  account_alias_snapshot?: string | null;
  instance_id?: string | null;
  instance_label?: string | null;
  instance_state?: string | null;
  gpu_reported?: string | null;
  retained_until?: string | null;
  runtime_deleted?: boolean | null;
  stage?: string | null;
  created_at?: string;
  updated_at?: string;
  events?: GpuEvent[];
  [key: string]: unknown;
}
export interface GpuEvent { id?: string; state?: string; event_type?: string; message?: string; progress?: number; created_at?: string; payload?: Record<string, unknown>; [key: string]: unknown; }
export interface GalleryAsset { id: string; project_id: string; mime: string; filename?: string; thumbnail_url?: string; download_url?: string; favorite?: boolean; gpu_job_id?: string; source_job_id?: string; mock?: boolean; metadata?: Record<string, unknown>; [key: string]: unknown; }
export interface CcuSnapshot { alias?: string | null; balance?: number | null; rate?: number | null; assignments?: number | null; active_jobs?: number | null; active_leases?: number | null; sampled_at?: number | null; error?: string; }
export interface CcuAnomaly { account_id: string; alias?: string | null; assignments?: number | null; first_seen?: number | null; last_seen?: number | null; resolved_at?: number | null; }
export interface UsageReconciliationRow { job_id?: string; state?: string | null; estimated_units?: number | null; actual_units?: number | null; ccu_attribution?: string | null; ccu_source?: string | null; }
export interface GpuUsage {
  estimated_units?: number | null;
  actual_units?: number | null;
  unit_rate?: number | null;
  daily_limit?: number | null;
  monthly_limit?: number | null;
  max_runtime_minutes?: number | null;
  active_runtimes?: number | null;
  reconciliation?: {
    measured_jobs?: number;
    unmeasured_jobs?: number;
    measured_actual_units?: number | null;
    measured_estimated_units?: number | null;
    variance_units?: number | null;
    recent?: UsageReconciliationRow[];
  };
  ccu?: {
    accounts?: number;
    total_balance?: number | null;
    assignments?: number;
    sampled_at?: number | null;
    snapshots?: Record<string, CcuSnapshot>;
    anomalies?: CcuAnomaly[];
  };
  [key: string]: unknown;
}
export interface GpuSettings { provider_default?: Provider; real_enabled?: boolean; google?: { colab?: { connected?: boolean; configured?: boolean; scopes?: string[] }; drive?: { connected?: boolean; configured?: boolean; scopes?: string[] } }; budget?: Record<string, unknown>; storage?: string; [key: string]: unknown; }
export interface ComputeSpec {
  id?: string;
  name?: string;
  provider?: Provider;
  gpu?: string;
  vram_gb?: number;
  available?: boolean;
  eligible?: boolean;
  key?: Record<string, unknown> | string;
  reason?: string;
  [key: string]: unknown;
}
export interface WorkflowBundle { manifest: GpuManifest; api_graph: Record<string, unknown>; object_info: Record<string, unknown>; editable_graph?: Record<string, unknown>; }

const TOKEN_KEY = 'colab-gpu-studio-token';
const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';

export class ApiError extends Error {
  constructor(message: string, public status = 0, public code = 'API_ERROR', public requestId?: string, public detail?: unknown) { super(message); this.name = 'ApiError'; }
}
function isRecord(value: unknown): value is Record<string, unknown> { return Boolean(value && typeof value === 'object' && !Array.isArray(value)); }
function sessionToken(): string { try { return sessionStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; } }
export function isLoggedIn(): boolean { return Boolean(sessionToken()); }
export function logout(): void { try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* storage can be unavailable in private mode */ } if (typeof window !== 'undefined') window.dispatchEvent(new Event('gpu-auth-expired')); }

/** GPU APIs use {data,error,request_id}; direct responses remain readable during backend rollout. */
export function unwrapGpuResponse<T>(value: ApiEnvelope<T> | T): T {
  if (!isRecord(value) || (!Object.prototype.hasOwnProperty.call(value, 'data') && !Object.prototype.hasOwnProperty.call(value, 'error'))) return value as T;
  const envelope = value as ApiEnvelope<T>;
  if (envelope.error) {
    const error = typeof envelope.error === 'string' ? { message: envelope.error } : envelope.error;
    throw new ApiError(error.message || 'GPU 服务请求失败', 200, error.code || 'GPU_API_ERROR', envelope.request_id, error.detail);
  }
  return envelope.data as T;
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const token = sessionToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);
  const abortParent = () => controller.abort();
  init.signal?.addEventListener('abort', abortParent, { once: true });
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, headers, credentials: 'include', signal: controller.signal });
  } catch (cause) {
    if (controller.signal.aborted) throw new ApiError(init.signal?.aborted ? '请求已取消' : '请求超时或网络中断，请刷新后再试', 0, 'NETWORK_TIMEOUT');
    throw cause;
  } finally {
    window.clearTimeout(timeout);
    init.signal?.removeEventListener('abort', abortParent);
  }
  const text = await response.text();
  let body: unknown = undefined;
  try { body = text ? JSON.parse(text) : undefined; } catch { body = text; }
  if (!response.ok) {
    const data = isRecord(body) ? body : {};
    const error = isRecord(data.error) ? data.error : data;
    if (response.status === 401) logout();
    throw new ApiError(String(error.message || `请求失败（${response.status}）`), response.status, String(error.code || 'HTTP_ERROR'), typeof data.request_id === 'string' ? data.request_id : undefined, error.detail);
  }
  return unwrapGpuResponse<T>(body as ApiEnvelope<T> | T);
}
const post = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) });
const write = (method: 'POST' | 'PATCH', body: unknown, idempotencyKey: string): RequestInit => ({
  method,
  headers: { 'Idempotency-Key': idempotencyKey },
  body: JSON.stringify(body),
});
export function apiMessage(error: unknown): string { return error instanceof Error ? error.message : '网络连接失败，请稍后重试'; }

export async function login(password: string): Promise<string> {
  const result = await apiFetch<{ access_token?: string }>('/api/auth/login', post({ password }));
  if (!result.access_token) throw new ApiError('登录响应没有本站 access_token');
  sessionStorage.setItem(TOKEN_KEY, result.access_token);
  return result.access_token;
}
export const getProjects = () => apiFetch<Project[]>('/api/gpu/projects');
export const createProject = (title: string) => apiFetch<Project>('/api/gpu/projects', post({ title }));
export const getGpuModels = () => apiFetch<GpuModel[]>('/api/gpu/models');
export const getGpuWorkflows = (modelId?: string) => apiFetch<GpuWorkflow[]>(`/api/gpu/workflows${modelId ? `?model_id=${encodeURIComponent(modelId)}` : ''}`);
export const getGpuWorkflowSchema = (id: string) => apiFetch<GpuManifest>(`/api/gpu/workflows/${encodeURIComponent(id)}/schema`);
export const importGpuWorkflow = (bundle: WorkflowBundle) => apiFetch<GpuWorkflow>('/api/gpu/workflows/import', post(bundle));
export const validateGpuWorkflow = (id: string) => apiFetch<GpuWorkflow>(`/api/gpu/workflows/${encodeURIComponent(id)}/validate`, post({}));
export const publishGpuWorkflow = (id: string) => apiFetch<GpuWorkflow>(`/api/gpu/workflows/${encodeURIComponent(id)}/publish`, post({ confirmed: true }));
export const rollbackGpuWorkflow = (workflow_key: string, version: string | number) => apiFetch<GpuWorkflow>('/api/gpu/workflows/rollback', post({ workflow_key, version }));
export const deactivateGpuWorkflow = (id: string) => apiFetch<GpuWorkflow>(`/api/gpu/workflows/${encodeURIComponent(id)}/deactivate`, post({ confirmed: true }));

export type ActivationStep = { key: string; label: string; done: boolean; next?: boolean; running?: boolean; failed?: boolean };
export type ActivationInput = { key: string; label: string; description?: string; type: string; required: boolean };
export type WorkflowActivation = {
  compute_provider?: 'kaggle' | 'colab';
  workflow: { id: string; workflow_key: string; version: string; state: string; active: boolean };
  model: null | { model_id: string; review_approved: boolean; license_accepted: boolean; enabled: boolean; license?: string; source?: string };
  steps: ActivationStep[];
  smoke: null | { job_id: string; project_id?: string | null; state: string; error_code?: string; error_message?: string; max_runtime_minutes?: number | null };
  required_inputs?: ActivationInput[];
  smoke_job_completed: boolean;
  smoke_kind?: string;
  real_enabled: boolean;
  max_runtime_minutes?: number;
  blockers: string[];
  done: boolean;
};
export const getWorkflowActivation = (id: string) => apiFetch<WorkflowActivation>(`/api/gpu/workflows/${encodeURIComponent(id)}/activation`);
export const reviewGpuModel = (modelId: string) => apiFetch<unknown>(`/api/gpu/models/${encodeURIComponent(modelId)}/review`, post({ confirmed: true, license_accepted: true }));
export const reviewGpuWorkflow = (id: string) => apiFetch<GpuWorkflow>(`/api/gpu/workflows/${encodeURIComponent(id)}/review`, post({ confirmed: true }));
export interface RealSmokeGpuWorkflowRequest {
  confirmed: true;
  project_id: string;
  params: Record<string, unknown>;
  asset_ids: string[];
  account_id: string;
  spec: ComputeSpec;
}
export const realSmokeGpuWorkflow = (
  id: string,
  body: RealSmokeGpuWorkflowRequest,
  idempotencyKey: string,
) => apiFetch<unknown>(`/api/gpu/workflows/${encodeURIComponent(id)}/real-smoke`, {
  method: 'POST', headers: { 'Idempotency-Key': idempotencyKey }, body: JSON.stringify(body),
});
export const acceptGpuWorkflowSmoke = (id: string, jobId: string) => apiFetch<GpuWorkflow>(`/api/gpu/workflows/${encodeURIComponent(id)}/accept-smoke`, post({ confirmed: true, job_id: jobId }));
export const getGpuPresets = () => apiFetch<GpuPreset[]>('/api/gpu/presets');
export const saveGpuPreset = (body: { name: string; workflow_version_id: string; overrides: Record<string, unknown> }) => apiFetch<GpuPreset>('/api/gpu/presets', post(body));

export async function createGpuJob(body: GpuJobRequest, idempotencyKey: string): Promise<GpuJob> {
  return apiFetch<GpuJob>('/api/gpu/jobs', { method: 'POST', headers: { 'Idempotency-Key': idempotencyKey }, body: JSON.stringify(body) });
}
export const getGpuJobs = (projectId: string) => apiFetch<GpuJob[]>(`/api/gpu/jobs?project_id=${encodeURIComponent(projectId)}`);
export const getGpuJob = (id: string) => apiFetch<GpuJob>(`/api/gpu/jobs/${encodeURIComponent(id)}`);
export const cancelGpuJob = (id: string) => apiFetch<GpuJob>(`/api/gpu/jobs/${encodeURIComponent(id)}/cancel`, post({}));
export const retryGpuJob = (id: string, idempotencyKey: string) => apiFetch<GpuJob>(`/api/gpu/jobs/${encodeURIComponent(id)}/retry`, { method: 'POST', headers: { 'Idempotency-Key': idempotencyKey }, body: JSON.stringify({}) });
export const getGpuGallery = (projectId: string) => apiFetch<GalleryAsset[]>(`/api/gpu/gallery?project_id=${encodeURIComponent(projectId)}`);
export const getGpuUsage = () => apiFetch<GpuUsage>('/api/gpu/usage');
export const getGpuSettings = () => apiFetch<GpuSettings>('/api/gpu/settings');
export const getGpuComputeSpecs = (provider: Provider, accountId?: string | null) => {
  const query = `provider=${encodeURIComponent(provider)}`;
  const bound = accountId ? `&account_id=${encodeURIComponent(accountId)}` : '';
  return apiFetch<ComputeSpec[]>(`/api/gpu/compute/specs?${query}${bound}`);
};
export async function startGoogleAuth(purpose: 'drive' | 'colab'): Promise<string> {
  const result = await apiFetch<{ authorization_url?: string }>('/api/gpu/auth/google/start', post({ purpose }));
  if (!result.authorization_url) throw new ApiError('Google OAuth 响应没有 authorization_url');
  return result.authorization_url;
}
export const disconnectGoogle = (purpose: 'drive' | 'colab') => apiFetch<Record<string, unknown>>('/api/gpu/auth/google/disconnect', post({ purpose }));

// v0.2.0 account/instance/workbench contract. All mutating calls receive the
// stable idempotency key from the caller; the client never generates one.
export const getGoogleAccounts = () => apiFetch<GoogleAccount[]>('/api/gpu/accounts');
export const createGoogleAccount = (body: GoogleAccountCreateRequest, idempotencyKey: string) =>
  apiFetch<GoogleAccount>('/api/gpu/accounts', write('POST', body, idempotencyKey));
export const updateGoogleAccount = (accountId: string, body: GoogleAccountUpdateRequest, idempotencyKey: string) =>
  apiFetch<GoogleAccount>(`/api/gpu/accounts/${encodeURIComponent(accountId)}`, write('PATCH', body, idempotencyKey));
export const disconnectGoogleAccount = (accountId: string, idempotencyKey: string) =>
  apiFetch<GoogleAccount>(`/api/gpu/accounts/${encodeURIComponent(accountId)}/disconnect`, write('POST', {}, idempotencyKey));
export const startGoogleAccountAuth = (accountId: string, purpose: GoogleAuthPurpose, idempotencyKey: string) =>
  apiFetch<{ authorization_url: string; state?: string }>(
    `/api/gpu/accounts/${encodeURIComponent(accountId)}/auth/${purpose}/start`,
    write('POST', { purpose }, idempotencyKey),
  );
export const getAccountRuntimes = (accountId: string) =>
  apiFetch<AccountRuntime[]>(`/api/gpu/accounts/${encodeURIComponent(accountId)}/runtimes`);

export const listGpuInstances = (filters: GpuInstanceFilters = {}) => {
  const query = new URLSearchParams();
  if (filters.state) query.set('state', filters.state);
  if (filters.account_id) query.set('account_id', filters.account_id);
  if (filters.provider) query.set('provider', filters.provider);
  const suffix = query.toString() ? `?${query.toString()}` : '';
  return apiFetch<GpuInstance[]>(`/api/gpu/instances${suffix}`);
};
export const getGpuInstance = (instanceId: string) =>
  apiFetch<GpuInstance>(`/api/gpu/instances/${encodeURIComponent(instanceId)}`);
export const getInstanceJobs = (instanceId: string) =>
  apiFetch<GpuJob[]>(`/api/gpu/instances/${encodeURIComponent(instanceId)}/jobs`);
export const getInstanceEvents = (instanceId: string) =>
  apiFetch<GpuEvent[]>(`/api/gpu/instances/${encodeURIComponent(instanceId)}/events`);
export const setInstanceRetention = (instanceId: string, body: InstanceRetentionRequest, idempotencyKey: string) =>
  apiFetch<GpuInstance>(`/api/gpu/instances/${encodeURIComponent(instanceId)}/retention`, write('PATCH', body, idempotencyKey));
export const stopInstance = (instanceId: string, idempotencyKey: string) =>
  apiFetch<GpuInstance>(`/api/gpu/instances/${encodeURIComponent(instanceId)}/stop`, write('POST', { confirmed: true }, idempotencyKey));
export const getWorkbench = (instanceId: string) =>
  apiFetch<Workbench>(`/api/gpu/instances/${encodeURIComponent(instanceId)}/workbench`);
export const workbenchAction = (instanceId: string, action: WorkbenchAction, params: Record<string, unknown> = {}, idempotencyKey: string) =>
  apiFetch<WorkbenchActionResult>(
    `/api/gpu/instances/${encodeURIComponent(instanceId)}/workbench/actions`,
    write('POST', { ...params, action }, idempotencyKey),
  );

// Short aliases keep the domain names usable from settings/compute pages.
export const getAccounts = getGoogleAccounts;
export const createAccount = createGoogleAccount;
export const updateAccount = updateGoogleAccount;
export const disconnectAccount = disconnectGoogleAccount;
export const startAccountAuth = startGoogleAccountAuth;
export const listInstances = listGpuInstances;
export const getInstance = getGpuInstance;
export const listInstanceJobs = getInstanceJobs;
export const listInstanceEvents = getInstanceEvents;
export const stopGpuCompute = () => apiFetch<Record<string, unknown>>('/api/gpu/admin/compute/stop', post({}));
export const smokeTestGpuWorkflow = (id: string) => apiFetch<GpuWorkflow>(`/api/gpu/workflows/${encodeURIComponent(id)}/smoke-test`, post({}));

export async function uploadAsset(projectId: string, file: File, onProgress?: (progress: number) => void): Promise<GalleryAsset> {
  const body = new FormData(); body.append('project_id', projectId); body.append('file', file);
  const token = sessionToken();
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open('POST', `${API_BASE}/api/assets`); xhr.withCredentials = true;
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(Math.round(event.loaded / event.total * 100)); };
    xhr.onload = () => { let data: unknown; try { data = JSON.parse(xhr.responseText); } catch { reject(new ApiError('上传服务响应异常')); return; } if (xhr.status >= 200 && xhr.status < 300) { try { resolve(unwrapGpuResponse<GalleryAsset>(data as ApiEnvelope<GalleryAsset> | GalleryAsset)); } catch (error) { reject(error); } } else reject(new ApiError('素材上传失败', xhr.status)); };
    xhr.onerror = () => reject(new ApiError('网络中断，请检查素材列表后重试'));
    xhr.send(body);
  });
}
export const assetUrl = (asset: GalleryAsset, thumbnail = false) => thumbnail ? asset.thumbnail_url || `/api/assets/${asset.id}/thumbnail` : asset.download_url || `/api/assets/${asset.id}/download`;
export const updateAsset = (id: string, favorite: boolean) => apiFetch<GalleryAsset>(`/api/assets/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify({ favorite }) });
export const deleteAsset = (id: string) => apiFetch<void>(`/api/assets/${encodeURIComponent(id)}`, { method: 'DELETE' });

export interface GoogleClientConfig { configured: boolean; client_id?: string; redirect_uri: string; source?: string; }
export const getGoogleClientConfig = (accountId?: string) => apiFetch<GoogleClientConfig>(`/api/gpu/auth/google/config${accountId ? `?account_id=${encodeURIComponent(accountId)}` : ''}`);
export const saveGoogleClientConfig = (body: {account_id?: string; client_id?: string; client_secret?: string; client_json?: Record<string, unknown>; confirmed: true}) => apiFetch<GoogleClientConfig>('/api/gpu/auth/google/config', post(body));
export const getReferenceAssets = (projectId: string) => apiFetch<GalleryAsset[]>(`/api/assets?project_id=${encodeURIComponent(projectId)}`);

export function createGpuBatch(jobs: GpuJobRequest[], mode: 'save'|'smart'|'batch'|'retain', idleSeconds: number, idempotencyKey: string) {
  return apiFetch<{id:string}>('/api/gpu/batches', {method:'POST',headers:{'Idempotency-Key':idempotencyKey},body:JSON.stringify({jobs,mode,idle_seconds:Math.min(300, Math.max(0, idleSeconds))})});
}

export const retryGpuAttempt = (id: string, key: string) => apiFetch<GpuJob>(`/api/gpu/jobs/${encodeURIComponent(id)}/attempts`, { method: "POST", headers: { "Idempotency-Key": key }, body: JSON.stringify({}) });
