import { SettingsPanel } from './pages/SettingsPage';
import { GalleryPanel } from './pages/GalleryPage';
import { TasksPanel } from './pages/JobsPage';
import { CreatePanel } from './pages/CreatePage';
import { ComputePage, ThemePicker } from './pages/ComputePage';
import { Fragment, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { Sparkles, Zap, Image as ImageIcon, Cpu, Settings as SettingsIcon, Folder, ChevronDown, X, Plus, Check, Maximize2, Star, Download, Trash2, RotateCcw, Triangle } from 'lucide-react';
import {
  apiMessage, assetUrl, cancelGpuJob, createGpuJob, createGpuBatch, createProject, deleteAsset, disconnectGoogle,
  apiFetch,
  getGpuComputeSpecs, getGpuGallery, getGpuJob, getGpuJobs, getGpuModels, listGpuInstances,
  getGpuPresets, getReferenceAssets, getGpuSettings, getGpuUsage, getGpuWorkflowSchema, getGpuWorkflows, getProjects, importGpuWorkflow,
  isLoggedIn, login, publishGpuWorkflow, retryGpuJob, retryGpuAttempt, startGoogleAuth, stopGpuCompute, updateAsset, uploadAsset,
  validateGpuWorkflow, rollbackGpuWorkflow, deactivateGpuWorkflow, saveGpuPreset, ApiError, type ComputeSpec, type GalleryAsset, type GpuJob, type GpuPreset,
  type GpuManifest, type GpuModel, type GpuSettings, type GpuUsage, type GpuWorkflow, type Project, type Provider,
} from './lib/gpu-api';
import type { GpuAccount, GpuInstance } from './components/AccountDirectory';
import type { RuntimeChoice } from './pages/CreatePage';
import { GpuParamForm } from './components/GpuParamForm';
import { ReferencePicker, referenceAssets, supportsReferences } from './components/ReferencePicker';
import { GoogleOAuthSetup } from './components/GoogleOAuthSetup';
import { ComputeSpecs } from './components/ComputeSpecs';
import { resolveGpuCheckRequest } from './lib/colab-check';
import { jobStateLabel } from './lib/job-stage-rows';
import { jobDetailTimings, paramRows, stageSecondsTotal, taskStatusNote } from './lib/job-detail-copy';
import { resolveInstanceSubmission } from './lib/instance-submission';
import { projectOptionLabel } from './lib/project-label';
import { carriedDraft, mergeWorkflowDraft, readWorkflowDraft, writeWorkflowDraft } from './lib/workflow-draft';
import { formatInstanceTime } from './lib/instance-time';
import { JobHistoryDetails } from './components/JobHistoryDetails';
import styles from './gpu.module.css';

const tabs = ['创作', '任务', '作品', '算力'] as const;
type Tab = typeof tabs[number] | '设置';
const terminalStates = new Set(['completed', 'succeeded', 'failed', 'failed_final', 'cancelled', 'cancelled_local', 'blocked']);

function record(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function uniqueBy<T>(items: T[], keyOf: (item: T) => string): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = keyOf(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function instanceIdOf(instance: Pick<GpuInstance, 'id' | 'instance_id'>): string {
  return instance.id || instance.instance_id || '';
}

function keyId() {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `gpu-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function shown(value: unknown) {
  return value === null || value === undefined || value === '' ? '未知' : typeof value === 'object' ? JSON.stringify(value) : String(value);
}

function workflowState(value?: string, active?: boolean) {
  return active ? '已发布 · 当前版本' : ({ draft: '草稿', validated: '已校验', published: '已发布 · 历史版本' }[value || ''] || value || '未知');
}

function progressPercent(value?: number) {
  return typeof value === 'number' ? Math.round(Math.max(0, Math.min(1, value)) * 100) : null;
}

function assetIdsFromParams(params: Record<string, unknown>, validIds: Set<string>) {
  return Object.values(params).filter((value): value is string => typeof value === 'string' && validIds.has(value));
}

function Login({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      await login(password);
      onLoggedIn();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <form className="loginCard" onSubmit={submit}>
        <div className="brand"><span className="brandMark"><Triangle size={16} strokeWidth={1.6} aria-hidden /></span><h1>Colab GPU Studio</h1></div>
        <p className="muted">输入访问密码进入工作台</p>
        <label>
          访问密码
          <input
            autoFocus
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            placeholder="请输入密码"
            required
          />
        </label>
        {error && <p className="status error" role="alert">{error}</p>}
        <button className="button big" disabled={busy} type="submit">
          {busy ? '登录中…' : '进入工作台'}
        </button>
      </form>
    </main>
  );
}

const PROJECT_KEY = 'colab-gpu-studio-project';
function savedProjectId(): string { try { return localStorage.getItem(PROJECT_KEY) || ''; } catch { return ''; } }
function saveProjectId(id: string) { try { if (id) localStorage.setItem(PROJECT_KEY, id); } catch { /* storage can be unavailable */ } }

function ProjectSheet({
  open,
  projects,
  projectId,
  onClose,
  onSelect,
}: {
  open: boolean;
  projects: Project[];
  projectId: string;
  onClose: () => void;
  onSelect: (id: string) => void;
}) {
  if (!open) return null;
  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="切换项目" onClick={(e) => e.stopPropagation()}>
        <div className={styles.cardHead}>
          <h2>切换项目</h2>
          <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={onClose}><X size={18} strokeWidth={1.6} aria-hidden /></button>
        </div>
        {projects.length === 0 && <p className="muted">还没有项目，点右上角的 ＋ 新建一个。</p>}
        <ul className="projectList">
          {projects.map((p) => (
            <li key={p.id}>
              <button
                type="button"
                className="projectRow"
                aria-current={p.id === projectId ? 'true' : undefined}
                onClick={() => { onSelect(p.id); onClose(); }}
              >
                <span>{projectOptionLabel(p, projects)}</span>
                {p.id === projectId && <Check size={16} strokeWidth={1.8} aria-label="当前项目" />}
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function NewProjectDialog({
  open,
  onClose,
  onCreate,
}: {
  open: boolean;
  onClose: () => void;
  onCreate: (title: string) => Promise<boolean>;
}) {
  const [title, setTitle] = useState('');
  const [creating, setCreating] = useState(false);
  if (!open) return null;

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    const name = title.trim();
    if (!name || creating) return;
    setCreating(true);
    try {
      if (await onCreate(name)) {
        setTitle('');
        onClose();
      }
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <form className={styles.dialog} role="dialog" aria-modal="true" aria-label="新建项目" onClick={(e) => e.stopPropagation()} onSubmit={handleCreate}>
        <div className={styles.cardHead}>
          <h2>新建项目</h2>
          <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={onClose}><X size={18} strokeWidth={1.6} aria-hidden /></button>
        </div>
        <div className="field">
          <label htmlFor="project-new">项目名称</label>
          <input id="project-new" autoFocus maxLength={80} placeholder="例如：符玄和青雀" value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>
        <div className={styles.inlineForm}>
          <button className="button secondary" type="button" onClick={onClose}>取消</button>
          <button className="button" disabled={creating || !title.trim()} type="submit">{creating ? '创建中…' : '创建'}</button>
        </div>
      </form>
    </div>
  );
}

export default function App() {
  const [loggedIn, setLoggedIn] = useState(isLoggedIn());
  const [tab, setTab] = useState<Tab>('创作');
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState('');
  const [showProjectSheet, setShowProjectSheet] = useState(false);
  const [showNewProject, setShowNewProject] = useState(false);
  const currentProjectId = useRef(projectId);
  currentProjectId.current = projectId;

  const [models, setModels] = useState<GpuModel[]>([]);
  const [workflows, setWorkflows] = useState<GpuWorkflow[]>([]);
  const [presets, setPresets] = useState<GpuPreset[]>([]);
  const [modelId, setModelId] = useState('');
  const [workflowId, setWorkflowId] = useState('');
  const [manifest, setManifest] = useState<GpuManifest | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [gallery, setGallery] = useState<GalleryAsset[]>([]);
  const [referenceImages, setReferenceImages] = useState<GalleryAsset[]>([]);
  const [jobs, setJobs] = useState<GpuJob[]>([]);
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
  const [runtime, setRuntime] = useState<Provider>('mock');
  const [maxRuntime, setMaxRuntime] = useState(30);
  const [batchCount,setBatchCount] = useState(1);
  const [batchMode,setBatchMode] = useState<'save'|'smart'|'batch'|'retain'>('save');
  const [runtimeChoice, setRuntimeChoice] = useState<RuntimeChoice>({ accountId: '', instanceId: '', lifecycle: 'save', retainIdleMinutes: 15 });
  const [accounts, setAccounts] = useState<GpuAccount[]>([]);
  const [instances, setInstances] = useState<GpuInstance[]>([]);
  const [resourceError, setResourceError] = useState('');
  const resourceRefresh = useRef<(() => void) | null>(null);
  const submitting = useRef(false);
  const [settings, setSettings] = useState<GpuSettings | null>(null);
  const [usage, setUsage] = useState<GpuUsage | null>(null);
  const [specs, setSpecs] = useState<ComputeSpec[]>([]);
  const [specsNote, setSpecsNote] = useState('');
  const [error, setError] = useState('');
  const [jobError, setJobError] = useState('');
  const [loading, setLoading] = useState('');
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [confirm, setConfirm] = useState<'submit' | 'stop' | null>(null);
  const [detail, setDetail] = useState<GpuJob | null>(null);
  const [viewer, setViewer] = useState<GalleryAsset | null>(null);
  const [againJob, setAgainJob] = useState<GpuJob | null>(null);
  const [loginRefresh, setLoginRefresh] = useState(0);
  const [refreshingJobs, setRefreshingJobs] = useState(false);
  const idempotencyKey = useRef('');
  const retryKeys = useRef<Record<string, string>>({});
  const instanceRetentionKeys = useRef<Record<string, string>>({});
  const jobSequence = useRef(0);
  const draftReady = useRef('');
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const workflowIdRef = useRef(workflowId);
  workflowIdRef.current = workflowId;
  const incomingDraft = useRef<Record<string, unknown> | null>(null);
  const draftLocation = useRef({ projectId: '', workflowId: '' });

  const activeProject = projects.find((p) => p.id === projectId);
  const activeWorkflow = workflows.find((item) => item.id === workflowId);

  const activeJobsCount = useMemo(() => {
    return jobs.filter((j) => !terminalStates.has(String(j.state || '').toLowerCase())).length;
  }, [jobs]);

  useEffect(() => {
    const expire = () => setLoggedIn(false);
    window.addEventListener('gpu-auth-expired', expire);
    return () => window.removeEventListener('gpu-auth-expired', expire);
  }, []);

  useEffect(() => {
    if (!loggedIn) return;
    let alive = true;
    Promise.allSettled([
      getProjects(),
      getGpuModels(),
      getGpuWorkflows(),
      getGpuPresets(),
      getGpuSettings(),
      getGpuUsage(),
    ]).then((results) => {
      if (!alive) return;
      const [projR, modR, wfR, preR, setR, useR] = results;
      if (projR.status === 'fulfilled') {
        setProjects(projR.value);
        // Restore the last chosen project when it still exists; otherwise fall back to the first one.
        const list = projR.value;
        setProjectId((curr) => {
          if (curr && list.some((p) => p.id === curr)) return curr;
          const saved = savedProjectId();
          return (saved && list.some((p) => p.id === saved) ? saved : list[0]?.id) || '';
        });
      } else setError(apiMessage(projR.reason));
      if (modR.status === 'fulfilled') setModels(modR.value);
      if (wfR.status === 'fulfilled') {
        const list = wfR.value;
        setWorkflows(list);
        setWorkflowId((curr) => {
          if (curr && list.some((w) => w.id === curr)) return curr;
          const active = list.find((w) => w.active && w.state === 'published') || list.find((w) => w.state === 'published') || list[0];
          if (active) {
            setModelId(active.model_id);
            return active.id;
          }
          return '';
        });
      }
      if (preR.status === 'fulfilled') setPresets(preR.value);
      if (setR.status === 'fulfilled') setSettings(setR.value);
      if (useR.status === 'fulfilled') setUsage(useR.value);
    }).catch((cause) => alive && setError(apiMessage(cause)));
    return () => { alive = false; };
  }, [loggedIn, loginRefresh]);

  useEffect(() => {
    if (!loggedIn) return;
    let alive = true;
    const refreshResources = async () => {
      try {
        const [accountResult, instanceResult] = await Promise.all([
          apiFetch<unknown>('/api/gpu/accounts'),
          listGpuInstances(),
        ]);
        if (!alive) return;
        const nextAccounts = (Array.isArray(accountResult) ? accountResult : record(accountResult) && Array.isArray(accountResult.data) ? accountResult.data : []) as GpuAccount[];
        const nextInstances = instanceResult as unknown as GpuInstance[];
        setAccounts(uniqueBy(nextAccounts, (account) => account.account_id));
        setInstances(uniqueBy(nextInstances, (instance) => instanceIdOf(instance)));
        setResourceError('');
      } catch (cause) {
        if (alive) setResourceError(`账号/实例：${apiMessage(cause)}`);
      }
    };
    resourceRefresh.current = () => void refreshResources();
    void refreshResources();
    const timer = window.setInterval(() => void refreshResources(), 5000);
    return () => { alive = false; resourceRefresh.current = null; window.clearInterval(timer); };
  }, [loggedIn]);

  useEffect(() => {
    if (!modelId) return;
    const allowed = workflows.filter((w) => w.model_id === modelId);
    if (workflowId && !allowed.some((w) => w.id === workflowId)) {
      if (projectId && draftReady.current === `${projectId}:${workflowId}`) writeWorkflowDraft(projectId, workflowId, paramsRef.current);
      draftReady.current = '';
      setWorkflowId('');
      setManifest(null);
      setParams({});
    }
  }, [modelId, workflows, workflowId, projectId]);

  useEffect(() => {
    if (!workflowId || !projectId) return;
    const previous = draftLocation.current;
    const sameProject = previous.projectId === projectId;
    if (previous.workflowId && previous.projectId && draftReady.current === `${previous.projectId}:${previous.workflowId}`) {
      writeWorkflowDraft(previous.projectId, previous.workflowId, paramsRef.current);
    }
    const carried = sameProject && previous.workflowId && previous.workflowId !== workflowId ? carriedDraft(paramsRef.current) : null;
    draftLocation.current = { projectId, workflowId };
    let alive = true;
    const requestWorkflow = workflowId;
    const requestProject = projectId;
    setLoading('schema');
    getGpuWorkflowSchema(workflowId).then((next) => {
      if (!alive) return;
      const forced = incomingDraft.current;
      incomingDraft.current = null;
      const saved = forced ?? readWorkflowDraft(requestProject, requestWorkflow);
      setManifest(next);
      draftReady.current = `${requestProject}:${requestWorkflow}`;
      setParams(mergeWorkflowDraft(next, saved, forced ? null : carried));
    }).catch((cause) => alive && setError(`参数 schema：${apiMessage(cause)}`)).finally(() => alive && setLoading(''));
    return () => { alive = false; };
  }, [workflowId, projectId]);

  useEffect(() => {
    if (draftReady.current !== `${projectId}:${workflowId}` || !workflowId || !projectId) return;
    writeWorkflowDraft(projectId, workflowId, params);
  }, [params, projectId, workflowId]);

  useEffect(() => {
    if (!projectId || !loggedIn) return;
    let alive = true;
    const refresh = async () => {
      const seq = ++jobSequence.current;
      setRefreshingJobs(true);
      try {
        const next = await getGpuJobs(projectId);
        if (!alive || seq !== jobSequence.current) return;
        setJobs(next.filter((j) => j.project_id === projectId));
        setJobError('');
      } catch (cause) {
        if (alive && seq === jobSequence.current) setJobError(apiMessage(cause));
      } finally {
        if (alive && seq === jobSequence.current) setRefreshingJobs(false);
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 4000);
    return () => {
      alive = false;
      window.clearInterval(timer);
      ++jobSequence.current;
    };
  }, [projectId, loggedIn]);

  useEffect(() => {
    setReferenceImages([]);
    setSelectedAssets([]);
    if (!projectId || !loggedIn) return;
    let alive = true;
    getReferenceAssets(projectId).then((next) => {
      if (alive) setReferenceImages(referenceAssets(next));
    }).catch((cause) => {
      if (alive) setError(`素材：${apiMessage(cause)}`);
    });
    return () => { alive = false; };
  }, [projectId, loggedIn]);

  useEffect(() => {
    setSelectedAssets([]);
  }, [workflowId,projectId]);



  useEffect(() => {
    if (!projectId || !loggedIn) return;
    let alive = true;
    getGpuGallery(projectId).then((next) => alive && setGallery(next)).catch((cause) => alive && setError(`作品：${apiMessage(cause)}`));
    return () => { alive = false; };
  }, [projectId, loggedIn, tab]);

  useEffect(() => {
    if (againJob && (againJob.resolved_params || againJob.params)) {
      const restored = { ...(againJob.resolved_params || againJob.params) };
      setModelId(String(againJob.model_id || ''));
      const workflow = workflows.find((item) => item.workflow_key === againJob.workflow_key && String(item.workflow_version) === String(againJob.workflow_version));
      if (workflow && workflow.id !== workflowIdRef.current) incomingDraft.current = restored;
      if (workflow) setWorkflowId(workflow.id);
      setParams(restored);
      setAgainJob(null);
      setTab('创作');
    }
  }, [againJob, workflows]);

  function chooseProject(id: string) {
    setProjectId(id);
    saveProjectId(id);
  }

  async function createLocalProject(title: string): Promise<boolean> {
    try {
      const created = await createProject(title);
      setProjects((curr) => [created, ...curr]);
      chooseProject(created.id);
      return true;
    } catch (cause) {
      setError(apiMessage(cause));
      return false;
    }
  }

  async function refreshJobs() {
    if (!projectId) return;
    setRefreshingJobs(true);
    try {
      setJobs((await getGpuJobs(projectId)).filter((j) => j.project_id === projectId));
      setJobError('');
    } catch (cause) {
      setJobError(apiMessage(cause));
    } finally {
      setRefreshingJobs(false);
    }
  }

  async function handleUpload(file: File) {
    const uploadProjectId = projectId;
    if (!uploadProjectId) return;
    setUploadProgress((curr) => ({ ...curr, [file.name]: 0 }));
    try {
      const asset = await uploadAsset(uploadProjectId, file, (progress) => {
        if (currentProjectId.current === uploadProjectId) setUploadProgress((curr) => ({ ...curr, [file.name]: progress }));
      });
      if (currentProjectId.current !== uploadProjectId || asset.project_id !== uploadProjectId) return;
      setReferenceImages((curr) => [asset, ...curr.filter((item) => item.id !== asset.id)]);
      setSelectedAssets((curr) => [...new Set([...curr, asset.id])]);
    } catch (cause) {
      if (currentProjectId.current === uploadProjectId) setError(`上传 ${file.name}：${apiMessage(cause)}`);
    } finally {
      if (currentProjectId.current === uploadProjectId) setUploadProgress((curr) => {
        const next = { ...curr };
        delete next[file.name];
        return next;
      });
    }
  }

  async function submitJob() {
    if (submitting.current || !projectId || !modelId || !workflowId || !activeWorkflow) return;
    submitting.current = true;
    if (!idempotencyKey.current) idempotencyKey.current = keyId();
    setLoading('submit');
    setError('');
    try {
      const decision = resolveInstanceSubmission(runtime === 'kaggle' ? {accountId: null, instanceId: null} : runtimeChoice, instances);
      if (!decision.ok) {
        setRuntimeChoice({ ...runtimeChoice, accountId: decision.accountId, instanceId: decision.instanceId });
        setError(decision.message);
        return;
      }
      const validAssetIds = new Set(referenceAssets(referenceImages).filter((a) => a.project_id === projectId).map((a) => a.id));
      const body = {
        project_id: projectId,
        model_id: modelId,
        workflow_key: activeWorkflow.workflow_key,
        workflow_version: activeWorkflow.workflow_version,
        params,
        asset_ids: supportsReferences(manifest)
          ? [...new Set([...selectedAssets.filter((id) => validAssetIds.has(id)), ...assetIdsFromParams(params, validAssetIds)])]
          : [],
        execution_policy: {
          provider: runtime,
          max_runtime_minutes: maxRuntime,
          allocation: decision.allocation,
          account_id: decision.accountId,
          instance_id: decision.instanceId,
          lifecycle: runtime === 'kaggle' ? 'save' : runtimeChoice.lifecycle,
          retain_idle_minutes: runtimeChoice.retainIdleMinutes,
        },
      };
      if (runtime === 'mock' || batchCount > 1) {
        // 批次沿用用户选定的账号、实例与生命周期，不再静默改成 batch。
        await createGpuBatch(Array.from({length:batchCount},(_,index)=>({...body,
          params:{...body.params,...(typeof body.params.seed === 'number' && body.params.seed >= 0 ? {seed:body.params.seed+index} : {})}})),
          runtime === 'kaggle' ? 'save' : runtimeChoice.lifecycle, runtimeChoice.retainIdleMinutes * 60, idempotencyKey.current);
      } else {
        await createGpuJob(body, idempotencyKey.current);
      }
      setConfirm(null);
      setTab('任务');
      idempotencyKey.current = '';
      await refreshJobs();
    } catch (cause) {
      const code = cause instanceof ApiError ? cause.code : '';
      if (code === 'INSTANCE_UNAVAILABLE' || code === 'INSTANCE_NOT_FOUND') {
        const decision = resolveInstanceSubmission(runtime === 'kaggle' ? {accountId: null, instanceId: null} : runtimeChoice, instances);
        setRuntimeChoice({ ...runtimeChoice, accountId: decision.accountId || runtimeChoice.accountId, instanceId: null });
        setError('所选 GPU 机器已释放或不能复用。这次没有提交任务，也没有换账号。请再点一次确认，改为自动分配，或另选一台可用机器。');
      } else if (code === 'NETWORK_TIMEOUT') {
        setError(`提交任务：${apiMessage(cause)}。网络中断时请先到「任务」页确认是否已经提交，不要连续改参数重试。`);
      } else {
        setError(`提交任务：${apiMessage(cause)}。这次没有确认任务已创建。`);
      }
    } finally {
      submitting.current = false;
      setLoading('');
    }
  }

  async function action(name: string, operation: () => Promise<unknown>) {
    setLoading(name);
    setError('');
    try {
      await operation();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setLoading('');
    }
  }

  async function importWorkflow(raw: string) {
    try {
      const value = JSON.parse(raw) as unknown;
      const source = record(value) && record(value.workflow) ? value.workflow : value;
      if (!record(source) || !record(source.manifest) || !record(source.api_graph) || !record(source.object_info)) {
        throw new Error('JSON 必须包含 manifest、api_graph、object_info');
      }
      await action('import', async () => {
        const workflow = await importGpuWorkflow(value as never);
        setWorkflows((curr) => [workflow, ...curr.filter((item) => item.id !== workflow.id)]);
        setError('工作流已作为草稿导入；请校验后发布。');
      });
    } catch (cause) {
      setError(`导入失败：${apiMessage(cause)}`);
    }
  }

  async function reuseAsset(asset: GalleryAsset) {
    const sourceJobId = asset.source_job_id || asset.gpu_job_id;
    if (!sourceJobId) {
      setError('该素材没有来源任务记录。');
      return;
    }
    try {
      setAgainJob(await getGpuJob(sourceJobId));
    } catch (cause) {
      setError(`再次生成：${apiMessage(cause)}`);
    }
  }

  async function favorite(asset: GalleryAsset) {
    try {
      const next = await updateAsset(asset.id, !asset.favorite);
      setGallery((curr) => curr.map((i) => i.id === next.id ? next : i));
    } catch (cause) {
      setError(apiMessage(cause));
    }
  }

  async function remove(asset: GalleryAsset) {
    try {
      await deleteAsset(asset.id);
      setGallery((curr) => curr.filter((i) => i.id !== asset.id));
      setViewer(null);
    } catch (cause) {
      setError(`删除失败：${apiMessage(cause)}`);
    }
  }

  async function savePreset(name: string) {
    if (!workflowId) return;
    await action('preset', async () => {
      const next = await saveGpuPreset({ name, workflow_version_id: workflowId, overrides: params });
      setPresets((curr) => [next, ...curr]);
    });
  }

  function applyPreset(preset: GpuPreset) {
    if (preset.workflow_version_id !== workflowId) {
      setError('预设版本不兼容。');
      return;
    }
    setParams({ ...params, ...preset.overrides });
  }

  async function checkGpu() {
    setSpecs([]);
    setSpecsNote('');
    await action('check', async () => {
      // Fetch after the click so login polling and recent authorizations cannot
      // turn a not-yet-loaded account directory into a disconnected result.
      const [currentAccounts, currentSettings] = await Promise.all([
        apiFetch<GpuAccount[]>('/api/gpu/accounts'),
        getGpuSettings(),
      ]);
      setAccounts(uniqueBy(currentAccounts, (account) => account.account_id));
      setSettings(currentSettings);
      if (currentSettings?.compute_provider === 'kaggle') {
        setSpecs(await getGpuComputeSpecs('kaggle'));
        setSpecsNote('Kaggle 请求规格；是否可分配以平台配额和容量为准。');
        return;
      }
      const checkRequest = resolveGpuCheckRequest(currentSettings, currentAccounts, runtimeChoice.accountId || null);
      if (checkRequest.provider === 'mock') {
        // 真实 GPU 没启用、离线或没有可用账号：明确改查模拟，不假装查了真实机器
        setSpecs(await getGpuComputeSpecs('mock'));
        setSpecsNote(checkRequest.reason === 'offline'
          ? '站点处于离线测试模式，下面是「模拟」GPU，不是真实 Colab 机器。'
          : checkRequest.reason === 'real_disabled'
            ? '真实 GPU 未启用，下面是「模拟」GPU，不是真实 Colab 机器。'
            : runtimeChoice.accountId
              ? '所选 Google 账号未连接或已停用，下面是「模拟」GPU。请先到「设置」完成授权。'
              : '账号池没有已连接且启用的 Google 账号，下面是「模拟」GPU。请先到「设置」完成授权。');
        return;
      }
      setSpecs(await getGpuComputeSpecs(currentSettings?.compute_provider === 'kaggle' ? 'kaggle' : 'colab', checkRequest.accountId));
      setSpecsNote(runtimeChoice.accountId
        ? '已查询所选 Google 账号的 Colab：有申请资格不等于现在一定有空闲 GPU。'
        : '已按账号池选择 Google 账号并查询 Colab：有申请资格不等于现在一定有空闲 GPU。');
    });
  }

  async function stopCompute() {
    await action('stop', async () => {
      await stopGpuCompute();
      setConfirm(null);
      setError('已发出取消任务请求；未自动释放没有任务的保留实例。');
    });
  }

  function changeRuntimeChoice(next: RuntimeChoice) {
    setRuntimeChoice(next);
    setBatchMode(next.lifecycle);
  }

  function openInstanceWorkbench(instance: GpuInstance) {
    const id = instanceIdOf(instance);
    if (!id) return;
    changeRuntimeChoice({ accountId: instance.account_id || null, instanceId: id, lifecycle: runtimeChoice.lifecycle, retainIdleMinutes: runtimeChoice.retainIdleMinutes });
    setRuntime(instance.provider === 'mock' ? 'mock' : 'colab');
    setTab('算力');
  }

  function continueInstance(instance: GpuInstance) {
    const id = instanceIdOf(instance);
    if (!id) return;
    changeRuntimeChoice({ accountId: instance.account_id || null, instanceId: id, lifecycle: 'save', retainIdleMinutes: runtimeChoice.retainIdleMinutes });
    setRuntime(instance.provider === 'mock' ? 'mock' : 'colab');
    setTab('创作');
  }

  async function retainJob(job: GpuJob) {
    const instanceId = String(job.instance_id || '');
    if (!instanceId) { setJobError('该任务尚未分配实例，暂不能保留。'); return; }
    await action('retain', async () => {
      const operation = `retain:${instanceId}`;
      const writeKey = instanceRetentionKeys.current[operation] ||= `retain:${instanceId}:${keyId()}`;
      const updatedInstance = await apiFetch<GpuInstance>(`/api/gpu/instances/${encodeURIComponent(instanceId)}/retention`, {
        method: 'PATCH', headers: { 'Idempotency-Key': writeKey },
        body: JSON.stringify({ retention_mode: 'retain', idle_minutes: runtimeChoice.retainIdleMinutes }),
      });
      delete instanceRetentionKeys.current[operation];
      setInstances((current) => {
        const index = current.findIndex((item) => item.instance_id === updatedInstance.instance_id);
        const resolvedIndex = index >= 0 ? index : current.findIndex((item) => instanceIdOf(item) === instanceIdOf(updatedInstance));
        if (resolvedIndex < 0) return uniqueBy([updatedInstance, ...current], (instance) => instanceIdOf(instance));
        const next = [...current];
        next[resolvedIndex] = { ...next[resolvedIndex], ...updatedInstance, id: instanceIdOf(updatedInstance), instance_id: instanceIdOf(updatedInstance) };
        return next;
      });
      setJobs((current) => current.map((item) => item.instance_id === instanceId
        ? { ...item, lifecycle: 'retain', retained_until: updatedInstance.retained_until, instance_state: updatedInstance.state }
        : item));
      await refreshJobs();
    });
  }

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        if (viewer) setViewer(null);
        else if (detail) setDetail(null);
        else if (confirm) setConfirm(null);
        else if (showNewProject) setShowNewProject(false);
        else if (showProjectSheet) setShowProjectSheet(false);
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [viewer, detail, confirm, showProjectSheet, showNewProject]);

  if (!loggedIn) {
    return <Login onLoggedIn={() => { setLoggedIn(true); setLoginRefresh((c) => c + 1); }} />;
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brandMark"><Triangle size={16} strokeWidth={1.6} aria-hidden /></span>
          <span className="brandText">
            <strong>Colab GPU Studio</strong>
            <small>{settings?.offline_mode ? '离线演示' : '个人创作工作台'}</small>
          </span>
        </div>
        <div className="topbarActions">
          <button className="projectChip" type="button" aria-label="切换项目" onClick={() => setShowProjectSheet(true)}>
            <Folder size={15} strokeWidth={1.6} aria-hidden />
            <span>{activeProject ? projectOptionLabel(activeProject, projects) : '选择项目'}</span>
            <ChevronDown size={14} strokeWidth={1.6} aria-hidden />
          </button>
          <button className="iconButton" type="button" aria-label="新建项目" title="新建项目" onClick={() => setShowNewProject(true)}>
            <Plus size={18} strokeWidth={1.6} aria-hidden />
          </button>
          <button className="iconButton" aria-label="打开设置" onClick={() => setTab('设置')}>
            <SettingsIcon size={18} strokeWidth={1.6} aria-hidden />
          </button>
        </div>
      </header>

      <nav className="bottomNav" aria-label="工作区导航">
        {tabs.map((item) => {
          const Icon = item === '创作' ? Sparkles : item === '任务' ? Zap : item === '作品' ? ImageIcon : Cpu;
          return (
            <button key={item} type="button" aria-current={tab === item ? 'page' : undefined} onClick={() => setTab(item)}>
              <Icon size={21} strokeWidth={1.5} aria-hidden />
              <span>{item}</span>
              {item === '任务' && activeJobsCount > 0 && <span className="tabBadge">{activeJobsCount}</span>}
            </button>
          );
        })}
      </nav>

      {/* 项目切换弹层 */}
      <ProjectSheet
        open={showProjectSheet}
        projects={projects}
        projectId={projectId}
        onClose={() => setShowProjectSheet(false)}
        onSelect={chooseProject}
      />
      <NewProjectDialog open={showNewProject} onClose={() => setShowNewProject(false)} onCreate={createLocalProject} />

      {/* 主工作区 */}
      <main className={styles.page}>
        {tab === '创作' && (
          <CreatePanel
            projectId={projectId}
            models={models}
            workflows={workflows}
            presets={presets}
            settings={settings}
            manifest={manifest}
            params={params}
            setParams={setParams}
            gallery={referenceImages}
            selectedAssets={selectedAssets}
            setSelectedAssets={setSelectedAssets}
            uploadProgress={uploadProgress}
            onUpload={handleUpload}
            modelId={modelId}
            setModelId={setModelId}
            workflowId={workflowId}
            setWorkflowId={setWorkflowId}
            runtime={runtime}
            setRuntime={setRuntime}
            maxRuntime={maxRuntime}
            setMaxRuntime={setMaxRuntime}
            batchCount={batchCount}
            setBatchCount={setBatchCount}
            batchMode={batchMode}
            setBatchMode={setBatchMode}
            runtimeChoice={runtimeChoice}
            accounts={accounts}
            instances={instances}
            onRuntimeChoiceChange={changeRuntimeChoice}
            loading={loading}
            error={error}
            onSubmit={() => setConfirm('submit')}
            onApplyPreset={applyPreset}
            onSavePreset={(name) => void savePreset(name)}
          />
        )}

        {tab === '任务' && (
          <TasksPanel
            jobs={jobs}
            error={jobError}
            refreshing={refreshingJobs}
            onRefresh={() => void refreshJobs()}
            onCancel={(id) => void action('cancel', async () => { await cancelGpuJob(id); await refreshJobs(); })}
            onRetry={(id) => void action('retry', async () => {
              const job = jobs.find((item) => item.id === id);
              const key = retryKeys.current[id] || `attempt:${id}:${keyId()}`;
              retryKeys.current[id] = key;
              if (job?.provider === 'mock') await retryGpuAttempt(id, key);
              else await retryGpuJob(id, key);
              delete retryKeys.current[id];
              await refreshJobs();
            })}
            onDetail={(job) => void action('inspect', async () => setDetail(await getGpuJob(job.id)))}
            onRetain={(job) => void retainJob(job)}
            onInstance={(job) => { const instance = instances.find((item) => instanceIdOf(item) === String(job.instance_id || '')); if (instance) { const id = instanceIdOf(instance); setRuntimeChoice({ accountId: instance.account_id || null, instanceId: id, lifecycle: runtimeChoice.lifecycle, retainIdleMinutes: runtimeChoice.retainIdleMinutes }); setTab('算力'); } }}
          />
        )}

        {tab === '作品' && (
          <GalleryPanel
            gallery={gallery}
            onFavorite={(asset) => void favorite(asset)}
            onDelete={(asset) => void remove(asset)}
            onOpen={setViewer}
            onReuse={(asset) => void reuseAsset(asset)}
            onGoCreate={() => setTab('创作')}
          />
        )}

        {tab === '算力' && (
          <ComputePage
            settings={settings}
            usage={usage}
            instances={instances}
            onContinue={continueInstance}
            onOpenWorkbench={openInstanceWorkbench}
            onRefreshInstances={() => resourceRefresh.current?.()}
            onStop={() => setConfirm('stop')}
            onCheck={() => void checkGpu()}
            loadingStop={loading === 'stop'}
            loadingCheck={loading === 'check'}
            specs={specs}
            specsNote={specsNote}
            error={error}
          />
        )}
        {tab === '设置' && (
          <SettingsPanel
            accounts={accounts}
            onAccountsChange={setAccounts}
            instances={instances}
            onRefreshInstances={() => resourceRefresh.current?.()}
            settings={settings}
            usage={usage}
            specs={specs}
            specsNote={specsNote}
            workflows={workflows}
            loading={loading}
            error={error}
            onConnect={(purpose) => void action('connect', async () => {
              const url = await startGoogleAuth(purpose);
              if (url) window.location.href = url;
            })}
            onDisconnect={(purpose) => void action('disconnect', async () => {
              await disconnectGoogle(purpose);
              setSettings(await getGpuSettings());
            })}
            onCheck={() => void checkGpu()}
            onStop={() => setConfirm('stop')}
            onImport={(bundle) => void importWorkflow(bundle)}
            onValidate={(wf) => void action('validate', async () => {
              await validateGpuWorkflow(wf.id);
              setWorkflows(await getGpuWorkflows());
            })}
            onPublish={(wf) => void action('publish', async () => {
              await publishGpuWorkflow(wf.id);
              setWorkflows(await getGpuWorkflows());
            })}
            onRollback={(wf) => void action('rollback', async () => {
              await rollbackGpuWorkflow(wf.workflow_key, wf.workflow_version);
              setWorkflows(await getGpuWorkflows());
            })}
            onDeactivate={(wf) => void action('deactivate', async () => {
              await deactivateGpuWorkflow(wf.id);
              setWorkflows(await getGpuWorkflows());
            })}
            projectId={projectId}
            projectTitle={activeProject ? projectOptionLabel(activeProject, projects) : ''}
            onOpenJob={(jobId, jobProjectId) => {
              if (jobProjectId) chooseProject(jobProjectId);
              setTab('任务');
              void getGpuJob(jobId).then(setDetail).catch((cause) => setError(apiMessage(cause)));
            }}
            onUseProject={(id) => { chooseProject(id); setTab('任务'); }}
            onActivated={() => { void getGpuWorkflows().then(setWorkflows); void getGpuModels().then(setModels).catch(() => undefined); }}
            onLogout={() => {
              sessionStorage.removeItem('colab-gpu-studio-token');
              setLoggedIn(false);
            }}
            onBack={() => setTab('创作')}
          />
        )}
      </main>

      {/* 确认生成弹窗 */}
      {confirm === 'submit' && activeWorkflow && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setConfirm(null)}>
          <section
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-label="确认提交任务"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>确认开始生成</h2>
            <dl className={styles.kv}>
              <dt>工作流</dt><dd>{activeWorkflow.workflow_key} · v{activeWorkflow.workflow_version}</dd>
              <dt>运行方式</dt><dd>{runtime === 'kaggle' ? 'Kaggle GPU（使用免费配额）' : runtime === 'colab' ? 'Colab GPU（会扣 CU）' : '离线模拟（不扣 CU）'}</dd>
              <dt>机器</dt><dd>{resolveInstanceSubmission(runtimeChoice, instances).machineLabel}</dd>
              <dt>数量</dt><dd>{batchCount}</dd>
              {runtime === 'colab' && <><dt>最长运行</dt><dd>{maxRuntime} 分钟</dd></>}
            </dl>
            {error && <p className="status error" role="alert">{error}</p>}
            <div className="actions">
              <button className="button secondary" type="button" onClick={() => setConfirm(null)}>取消</button>
              <button className="button" disabled={loading === 'submit'} type="button" onClick={() => void submitJob()}>
                {loading === 'submit' ? '提交中…' : '确认启动'}
              </button>
            </div>
          </section>
        </div>
      )}

      {/* 确认急停弹窗 */}
      {confirm === 'stop' && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setConfirm(null)}>
          <section
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-label="确认停止"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>停止全部任务？</h2>
            <p className="muted">会取消所有正在跑的任务，并关闭对应的 Colab GPU 机器。已经花掉的 CU 不会退回。</p>
            <div className="actions">
              <button className="button secondary" type="button" onClick={() => setConfirm(null)}>取消</button>
              <button className="button danger" type="button" onClick={() => void stopCompute()}>确认停止</button>
            </div>
          </section>
        </div>
      )}

      {/* 任务详情 */}
      {detail && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setDetail(null)}>
          <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="任务详情" onClick={(e) => e.stopPropagation()}>
            <div className={styles.cardHead}>
              <h2>任务详情 · {detail.id.slice(0, 8)}</h2>
              <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={() => setDetail(null)}><X size={18} strokeWidth={1.6} aria-hidden /></button>
            </div>
            <dl className={styles.kv}>
              <dt>状态</dt><dd>{jobStateLabel(detail.state)}</dd>
              <dt>运行方式</dt><dd>{detail.provider === 'mock' ? '离线模拟' : detail.provider === 'colab' ? 'Colab GPU' : shown(detail.provider)}</dd>
              <dt>创建时间</dt><dd>{formatInstanceTime(detail.created_at)}</dd>
            </dl>
            <p className="muted small">{taskStatusNote(detail)}</p>
            {Array.isArray(detail.attempts) && <details className="foldable" open>
              <summary>执行记录</summary>
              <div className="foldContent">
                {detail.attempts.length ? detail.attempts.map((attempt: Record<string, unknown>) => <p key={String(attempt.id)}>
                  第 {String(attempt.attempt)} 次 · {jobStateLabel(String(attempt.state))}
                  {attempt.error_code ? ` · ${String(attempt.error_code)}` : ''}
                </p>) : <p className="muted">还没有开始执行</p>}
              </div>
            </details>}
            {(detail.timings || detail.timing_details || detail.logs || detail.events) ? <JobHistoryDetails job={detail} /> : null}
            {stageSecondsTotal(jobDetailTimings(detail)) ? <p className="muted small">各阶段相加约 {stageSecondsTotal(jobDetailTimings(detail))}。阶段按秒独立计时，可能重叠，不是工作流发布又额外等了这么久。</p> : null}
            {detail.resolved_params ? (
              <details className="foldable">
                <summary>本次使用的参数</summary>
                <div className="foldContent">
                  <dl className={styles.kv}>
                    {paramRows(detail.resolved_params).map((row) => <Fragment key={row.key}><dt>{row.label}</dt><dd>{row.value}</dd></Fragment>)}
                  </dl>
                </div>
              </details>
            ) : null}
          </section>
        </div>
      )}

      {/* 作品查看 */}
      {viewer && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setViewer(null)}>
          <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="查看作品" onClick={(e) => e.stopPropagation()}>
            <div className={`${styles.cardHead} ${styles.viewerHead}`}>
              <h2>{viewer.filename || viewer.id}</h2>
              <div className={styles.viewerHeadActions}>
                <button className="button secondary small" type="button" onClick={async (event) => {
                  const panel = event.currentTarget.closest('section');
                  try {
                    if (document.fullscreenElement) await document.exitFullscreen();
                    else if (panel?.requestFullscreen) await panel.requestFullscreen();
                    else throw new Error('当前浏览器不支持全屏');
                  } catch (error) { setError(apiMessage(error)); }
                }}><Maximize2 size={14} strokeWidth={1.6} aria-hidden />全屏查看</button>
                <button className="button secondary small" type="button" onClick={async () => {
                  if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
                  setViewer(null);
                }}>关闭</button>
              </div>
            </div>
            {viewer.mime.startsWith('video/') ? (
              <video className={styles.viewerVideo} src={assetUrl(viewer)} controls autoPlay playsInline />
            ) : viewer.mime.startsWith('image/') ? (
              <img className={styles.viewerImage} src={assetUrl(viewer)} alt="" />
            ) : (
              <p className="muted">这是诊断文件，不能直接预览，请下载查看。</p>
            )}
            <div className={styles.viewerActions}>
              <button className="button secondary small" type="button" onClick={() => void favorite(viewer)}>
                <Star size={14} strokeWidth={1.6} aria-hidden />{viewer.favorite ? '取消收藏' : '收藏'}
              </button>
              <button className="button secondary small" type="button" onClick={() => void reuseAsset(viewer)}>
                <RotateCcw size={14} strokeWidth={1.6} aria-hidden />用同样参数再来一次
              </button>
              <a className="button secondary small" href={assetUrl(viewer)} download={viewer.filename || viewer.id} target="_blank" rel="noreferrer">
                <Download size={14} strokeWidth={1.6} aria-hidden />下载
              </a>
              <button className="button danger small" type="button" onClick={() => void remove(viewer)}>
                <Trash2 size={14} strokeWidth={1.6} aria-hidden />删除
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
