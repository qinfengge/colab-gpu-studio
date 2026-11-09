import { useCallback, useEffect, useRef, useState } from 'react';
import {
  acceptGpuWorkflowSmoke, apiMessage, getGpuComputeSpecs, getGoogleAccounts, getReferenceAssets, getWorkflowActivation,
  publishGpuWorkflow, realSmokeGpuWorkflow, reviewGpuModel, reviewGpuWorkflow, uploadAsset, validateGpuWorkflow, getGpuJob,
  type ActivationInput, type ComputeSpec, type GalleryAsset, type GpuWorkflow, type WorkflowActivation,
} from '../lib/gpu-api';
import {
  activationSmokeParams, appendActivationLog, getActivationStep, getSmokeRuntimeMinutes, hasExistingSmoke,
  isSmokeCompleted, isSmokePolling, missingActivationInputs, shouldAcceptSmoke, shouldCreateSmoke, smokeIdempotencyKey,
  terminalSmokeJobId, type ActivationWatch,
} from '../lib/activation-resume';
import { resolveGpuCheckRequest, selectEligibleColabGpuSpec } from '../lib/colab-check';
import styles from '../gpu.module.css';
import { JobHistoryDetails, type JobHistoryDetailRecord } from './JobHistoryDetails';

const STEPS_POLL_MS = 6000;

const sleep = (ms: number) => new Promise((resolve) => { window.setTimeout(resolve, ms); });

function runtimeText(minutes: number | null): string {
  if (minutes === null) return '运行时长以服务端任务状态为准';
  const value = Number.isInteger(minutes) ? String(minutes) : minutes.toFixed(1);
  return `服务端运行上限 ${value} 分钟`;
}

function smokeStateText(state: string, minutes: number | null): string {
  const limit = minutes === null ? '' : `，最长 ${Number.isInteger(minutes) ? minutes : minutes.toFixed(1)} 分钟`;
  if (state === 'queued' || state === 'pending') return `冒烟任务已在队列中${limit}`;
  if (state === 'running') return `冒烟任务正在运行${limit}`;
  if (state === 'completed' || state === 'succeeded' || state === 'success') return '冒烟任务已完成，可直接验收';
  return `冒烟任务状态：${state}`;
}

export function WorkflowActivationPanel({
  workflow, projectId, projectTitle, onClose, onDone, onPreview, onOpenJob, onUseProject, onWatch,
}: {
  workflow: GpuWorkflow;
  projectId: string;
  projectTitle: string;
  onClose: () => void;
  onDone: () => void;
  onPreview?: () => void;
  onOpenJob?: (jobId: string, projectId?: string) => void;
  onUseProject?: (projectId: string) => void;
  onWatch?: (watch: ActivationWatch | null) => void;
}) {
  const [data, setData] = useState<WorkflowActivation | null>(null);
  const [history, setHistory] = useState<JobHistoryDetailRecord | null>(null);
  const [historyError, setHistoryError] = useState('');
  const [historyBusy, setHistoryBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [consent, setConsent] = useState(false);
  const [accountId, setAccountId] = useState('');
  const [spec, setSpec] = useState<ComputeSpec | null>(null);
  const [specsLoaded, setSpecsLoaded] = useState(false);
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [assets, setAssets] = useState<GalleryAsset[]>([]);
  const [uploading, setUploading] = useState('');
  const aliveRef = useRef(false);
  const executionRef = useRef(0);
  const runningRef = useRef(false);
  const fileRefs = useRef<Record<string, HTMLInputElement | null>>({});
  const onWatchRef = useRef(onWatch);
  onWatchRef.current = onWatch;

  const isCurrent = useCallback((token: number) => aliveRef.current && executionRef.current === token, []);

  const say = useCallback((line: string, token?: number) => {
    if (!aliveRef.current || (token !== undefined && executionRef.current !== token)) return;
    setLog((current) => appendActivationLog(current, line));
  }, []);

  const refresh = useCallback(async (token: number): Promise<WorkflowActivation> => {
    const next = await getWorkflowActivation(workflow.id);
    if (isCurrent(token)) setData(next);
    return next;
  }, [isCurrent, workflow.id]);

  const waitForSmoke = useCallback(async (
    initial: WorkflowActivation,
    token: number,
  ): Promise<WorkflowActivation | null> => {
    let snapshot = initial;
    let announcedState = '';
    const minutes = getSmokeRuntimeMinutes(snapshot);
    while (isCurrent(token)) {
      const state = String(snapshot.smoke?.state || '').toLowerCase();
      if (state && state !== announcedState) {
        announcedState = state;
        say(smokeStateText(state, minutes), token);
      }
      if (isSmokeCompleted(snapshot) || ['failed', 'cancelled', 'canceled', 'error'].includes(state)) {
        return snapshot;
      }
      await sleep(STEPS_POLL_MS);
      if (!isCurrent(token)) return null;
      snapshot = await refresh(token);
      if (!isCurrent(token)) return null;
    }
    return null;
  }, [isCurrent, refresh, say]);

  const runAll = useCallback(async () => {
    if (runningRef.current || busy || !aliveRef.current) return;
    runningRef.current = true;
    const token = executionRef.current + 1;
    executionRef.current = token;
    setError('');
    setBusy(true);
    try {
      let snapshot = await refresh(token);
      if (!isCurrent(token)) return;

      if (shouldCreateSmoke(snapshot) && snapshot.blockers.length) {
        throw new Error(snapshot.blockers[0]);
      }
      if (getActivationStep(snapshot, 'validate')?.done !== true) {
        say('正在校验模板…', token);
        await validateGpuWorkflow(workflow.id);
        if (!isCurrent(token)) return;
        snapshot = await refresh(token);
        if (!isCurrent(token)) return;
      }
      if (getActivationStep(snapshot, 'model_review')?.done !== true) {
        if (!snapshot.model) throw new Error('找不到绑定的模型记录');
        say('正在记录模型来源与许可…', token);
        await reviewGpuModel(snapshot.model.model_id);
        if (!isCurrent(token)) return;
        snapshot = await refresh(token);
        if (!isCurrent(token)) return;
      }
      if (getActivationStep(snapshot, 'workflow_review')?.done !== true) {
        say('正在锁定权重与节点…', token);
        await reviewGpuWorkflow(workflow.id);
        if (!isCurrent(token)) return;
        snapshot = await refresh(token);
        if (!isCurrent(token)) return;
      }

      if (getActivationStep(snapshot, 'real_smoke')?.done !== true) {
        if (hasExistingSmoke(snapshot)) {
          say('发现已有冒烟任务，继续读取服务端进度，不会重新提交。', token);
          if (!isSmokeCompleted(snapshot)) {
            const resumed = await waitForSmoke(snapshot, token);
            if (!resumed) return;
            snapshot = resumed;
          }
        } else {
          const required = snapshot.required_inputs || [];
          const missing = missingActivationInputs(required, inputs);
          if (missing.length) throw new Error(`请先上传：${missing.join('、')}`);
          if (!consent) throw new Error('请先确认会真实开 GPU 并产生费用');
          if (!projectId) throw new Error('请先在顶栏选择项目。冒烟只会记在当前项目里，不会自动改用别的项目。');
          if (!accountId || !spec) throw new Error('当前账号没有可用的 GPU 规格，请先授权 Colab');
          const minutes = getSmokeRuntimeMinutes(snapshot);
          say(minutes === null ? '正在提交真实 GPU 冒烟…' : `正在提交真实 GPU 冒烟，最长 ${Number.isInteger(minutes) ? minutes : minutes.toFixed(1)} 分钟`, token);
          const smokeBody = activationSmokeParams(required, inputs);
          await realSmokeGpuWorkflow(
            workflow.id,
            {
              confirmed: true,
              project_id: projectId,
              params: smokeBody.params,
              asset_ids: smokeBody.assetIds,
              account_id: accountId,
              spec,
            },
            smokeIdempotencyKey(workflow.id, projectId, smokeBody.assetIds),
          );
          if (!isCurrent(token)) return;
          snapshot = await refresh(token);
          if (!isCurrent(token)) return;
          snapshot = (await waitForSmoke(snapshot, token)) || snapshot;
          if (!isCurrent(token)) return;
        }
      }

      if (!isSmokeCompleted(snapshot)) {
        throw new Error(snapshot.smoke?.error_message || '冒烟未通过，详见任务详情');
      }
      if (shouldAcceptSmoke(snapshot)) {
        if (!snapshot.smoke?.job_id) throw new Error('找不到可验收的冒烟任务');
        say('正在验收冒烟输出…', token);
        await acceptGpuWorkflowSmoke(workflow.id, snapshot.smoke.job_id);
        if (!isCurrent(token)) return;
        snapshot = await refresh(token);
        if (!isCurrent(token)) return;
      }
      if (!snapshot.done) {
        say('正在发布并设为当前版本…', token);
        await publishGpuWorkflow(workflow.id);
        if (!isCurrent(token)) return;
        snapshot = await refresh(token);
      }
      if (!isCurrent(token)) return;
      say('完成，可以回创作页用了', token);
      onDone();
    } catch (err) {
      if (isCurrent(token)) setError(apiMessage(err));
    } finally {
      runningRef.current = false;
      if (isCurrent(token)) setBusy(false);
    }
  }, [accountId, busy, consent, inputs, isCurrent, onDone, projectId, refresh, say, spec, waitForSmoke, workflow.id]);

  useEffect(() => {
    aliveRef.current = true;
    const token = executionRef.current + 1;
    executionRef.current = token;
    const load = async () => {
      try {
        const snapshot = await refresh(token);
        if (!isCurrent(token)) return;
        if (hasExistingSmoke(snapshot)) {
          setSpecsLoaded(true);
          if (isSmokePolling(snapshot)) {
            setBusy(true);
            say('已恢复服务端冒烟进度…', token);
            try { await waitForSmoke(snapshot, token); }
            finally { if (isCurrent(token)) setBusy(false); }
          } else if (isSmokeCompleted(snapshot)) {
            say('已找到已完成的冒烟任务，可直接验收', token);
          }
          return;
        }
        if (snapshot.compute_provider === 'kaggle') {
          const eligible = selectEligibleColabGpuSpec(await getGpuComputeSpecs('kaggle'));
          if (!isCurrent(token)) return;
          if (!eligible) throw new Error('Kaggle GPU 尚未配置');
          setAccountId('kaggle');
          setSpec(eligible);
          return;
        }
        const accounts = await getGoogleAccounts();
        if (!isCurrent(token)) return;
        const request = resolveGpuCheckRequest(
          { real_enabled: snapshot.real_enabled, offline_mode: false },
          accounts,
          null,
        );
        if (request.provider !== 'colab') {
          throw new Error('没有已启用且已连接 Colab 的账号，无法读取真实 GPU 规格');
        }
        const eligible = selectEligibleColabGpuSpec(await getGpuComputeSpecs('colab', request.accountId));
        if (!isCurrent(token)) return;
        if (!eligible) throw new Error('当前账号没有可用的真实 GPU 规格（已过滤 CPU/TPU）');
        setAccountId(request.accountId);
        setSpec(eligible);
      } catch (err) {
        if (isCurrent(token)) setError(apiMessage(err));
      } finally {
        if (isCurrent(token)) setSpecsLoaded(true);
      }
    };
    void load();
    return () => {
      aliveRef.current = false;
      executionRef.current += 1;
    };
  }, [isCurrent, refresh, say, waitForSmoke]);

  useEffect(() => {
    if (!projectId) {
      setAssets([]);
      return;
    }
    let alive = true;
    getReferenceAssets(projectId).then((rows) => {
      if (alive) setAssets(rows.filter((asset) => String(asset.mime || '').startsWith('image/')));
    }).catch(() => undefined);
    return () => { alive = false; };
  }, [projectId]);

  useEffect(() => {
    const notify = onWatchRef.current;
    if (!notify) return;
    const smoke = data?.smoke;
    if (!smoke?.job_id) {
      notify(null);
      return;
    }
    notify({
      workflowId: workflow.id,
      projectId: smoke.project_id || projectId,
      jobId: smoke.job_id,
      state: String(smoke.state || ''),
      maxRuntimeMinutes: getSmokeRuntimeMinutes(data),
    });
  }, [data, projectId, workflow.id]);

  const existingSmoke = hasExistingSmoke(data);
  const needsNewSmoke = data ? !data.done && shouldCreateSmoke(data) : true;
  const runtimeLimit = getSmokeRuntimeMinutes(data);
  const requiredInputs = data?.required_inputs || [];
  const missing = missingActivationInputs(requiredInputs, inputs);
  const smokeProjectId = data?.smoke?.project_id || '';
  const smokeElsewhere = Boolean(existingSmoke && smokeProjectId && projectId && smokeProjectId !== projectId);

  async function uploadFor(input: ActivationInput, file: File) {
    if (!projectId) {
      setError('请先在顶栏选择项目，再上传图片。');
      return;
    }
    setUploading(input.key);
    setError('');
    try {
      const asset = await uploadAsset(projectId, file);
      setAssets((current) => [asset, ...current.filter((item) => item.id !== asset.id)]);
      setInputs((current) => ({ ...current, [input.key]: asset.id }));
    } catch (err) {
      setError(apiMessage(err));
    } finally {
      setUploading('');
    }
  }

  return (
    <div className={styles.activationPanel}>
      <p className="muted small">
        这一步会真实分配 GPU 跑一次冒烟（{runtimeText(runtimeLimit)}，Kaggle 使用免费配额，Colab 按实际用量计费）。通过后自动发布并设为当前版本。发布是单独的一步，不会在冒烟排队时一直空转。
      </p>
      <p className="muted small">
        {projectId
          ? `新的冒烟任务记在当前项目「${projectTitle || '当前项目'}」里，可以在这个项目的任务页看到。`
          : '还没有选中项目。请先用顶栏切换到要归档这次启用的项目，不会自动改用列表里的第一个项目。'}
      </p>
      {smokeElsewhere ? (
        <div className={styles.notice}>
          <p className="small">已有冒烟任务记在另一个项目里，所以当前项目的任务页看不到它。继续验收不会再提交一次。</p>
          {onUseProject ? <button className="button secondary small" type="button" onClick={() => onUseProject(smokeProjectId)}>打开那个项目的任务</button> : null}
        </div>
      ) : null}
      {data?.blockers.length && !existingSmoke ? (
        <div className={styles.notice}>
          {data.blockers.map((item) => <p key={item} className="small">{item}</p>)}
        </div>
      ) : null}
      {needsNewSmoke && requiredInputs.length ? (
        <div className={styles.stack}>
          <p className="small">这个工作流的冒烟需要图片。没有填必填项时不会提交，也不会跳过校验。</p>
          {requiredInputs.map((input) => (
            <div className={styles.paramField} key={input.key}>
              <div className={styles.paramLabel}>
                <label htmlFor={`activation-${input.key}`}>{input.label}{input.required ? '（必填）' : '（可选）'}</label>
              </div>
              <div className={styles.fileControl}>
                <select id={`activation-${input.key}`} value={inputs[input.key] || ''} onChange={(event) => setInputs((current) => ({ ...current, [input.key]: event.target.value }))}>
                  <option value="">{input.required ? '请选择或上传图片' : '不使用'}</option>
                  {assets.map((asset) => <option key={asset.id} value={asset.id}>{asset.filename || asset.id}</option>)}
                </select>
                <button className="button secondary small" type="button" disabled={uploading === input.key} onClick={() => fileRefs.current[input.key]?.click()}>
                  {uploading === input.key ? '上传中…' : '上传'}
                </button>
                <input
                  ref={(node) => { fileRefs.current[input.key] = node; }}
                  hidden
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void uploadFor(input, file);
                    event.target.value = '';
                  }}
                />
              </div>
              {input.description ? <span className={styles.hint}>{input.description}</span> : null}
            </div>
          ))}
          {missing.length ? <p className={styles.errorText}>还缺：{missing.join('、')}</p> : null}
        </div>
      ) : null}
      <ol className={styles.activationSteps}>
        {(data?.steps || []).map((step) => (
          <li key={step.key} data-state={step.done ? 'done' : step.failed ? 'failed' : step.running ? 'running' : step.next ? 'next' : 'todo'}>
            <span>{step.label}</span>
            <span className="pill" data-state={step.done ? 'active' : undefined}>
              {step.done ? '已完成' : step.failed ? '失败' : step.running ? '进行中' : '待执行'}
            </span>
          </li>
        ))}
      </ol>
      {error ? <p className={styles.errorText}>{error}</p> : null}
      {log.length ? (
        <details className="foldable" open>
          <summary>执行记录</summary>
          <div className="foldContent">
            {log.map((line, index) => <p key={`${line}-${index}`} className="muted small">{line}</p>)}
          </div>
        </details>
      ) : null}
      {needsNewSmoke ? (
        <label className={styles.checkline}>
          <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
          <span>我知道会真实开 GPU 并产生费用</span>
        </label>
      ) : null}
      {historyError ? <p role="alert" className={styles.errorText}>{historyError}</p> : null}
      {history ? <section aria-label="持久任务详情"><JobHistoryDetails job={history} /></section> : null}
      <div className="actions">
        {terminalSmokeJobId(data) ? <button className="button secondary" type="button" disabled={historyBusy} onClick={() => {
          const jobId = terminalSmokeJobId(data);
          if (!jobId) return;
          if (onOpenJob) { onOpenJob(jobId, smokeProjectId || projectId); return; }
          const token = executionRef.current;
          setHistoryError('');
          setHistoryBusy(true);
          void getGpuJob(jobId).then((job) => { if (isCurrent(token)) setHistory(job); })
            .catch((err) => { if (isCurrent(token)) setHistoryError(apiMessage(err)); })
            .finally(() => { if (isCurrent(token)) setHistoryBusy(false); });
        }}>查看任务详情/日志</button> : null}
        {onPreview ? <button className="button secondary" type="button" onClick={onPreview}>预览草稿</button> : null}
        <button
          className="button"
          type="button"
          disabled={busy || !specsLoaded || (needsNewSmoke && (!consent || missing.length > 0 || !projectId))}
          onClick={() => void runAll()}
        >
          {busy ? '执行中…' : data?.done ? '重新检查' : existingSmoke ? '继续验收并启用' : '一键跑通并启用'}
        </button>
        <button className="button secondary" type="button" onClick={() => {
          executionRef.current += 1;
          onClose();
        }}>关闭</button>
      </div>
    </div>
  );
}
