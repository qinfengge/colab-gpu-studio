import { useEffect, useMemo, useState } from 'react';
import {
  Sparkles, Image as ImageIcon, Film, Layers, Pencil, SlidersHorizontal, Play, Monitor, Cloud,
  ChevronRight, Minus, Plus, X, Check,
} from 'lucide-react';
import type {
  GalleryAsset, GpuManifest, GpuModel, GpuPreset, GpuSettings, GpuWorkflow, Provider,
} from '../lib/gpu-api';
import { GpuParamForm, splitParams } from '../components/GpuParamForm';
import { ReferencePicker, supportsReferences } from '../components/ReferencePicker';
import { RuntimeSelection } from '../components/RuntimeSelection';
import type { RuntimeSelectionValue } from '../components/RuntimeSelection';
import type { GpuAccount, GpuInstance } from '../components/AccountDirectory';
import { Card, InfoTip, Segmented } from '../components/ui';
import { RUNTIME_HELP } from '../lib/labels';
import { workflowTitle, workflowUsage } from '../lib/workflow-copy';
import { parameterList, parameterKey, shown } from '../lib/ui-helpers';
import styles from '../gpu.module.css';

type Category = 'image' | 'video';
export type RuntimeChoice = RuntimeSelectionValue;

const isVideoKey = (key: string) => /i2v|video|ref2v|t2v/i.test(key);

/** 常用画幅。图片使用固定尺寸表；视频等小分辨率工作流按工作流默认像素量等比换算。 */
const RATIOS = [
  { label: '1:1', rw: 1, rh: 1, size: [992, 992] },
  { label: '3:4', rw: 3, rh: 4, size: [768, 1024] },
  { label: '9:16', rw: 9, rh: 16, size: [576, 1024] },
  { label: '4:3', rw: 4, rh: 3, size: [1024, 768] },
  { label: '16:9', rw: 16, rh: 9, size: [1024, 576] },
] as const;

const roundTo = (value: number, step: number) => Math.max(step, Math.round(value / step) * step);

function buildSizes(defW: number, defH: number, step: number, min: number, max: number) {
  const area = defW * defH;
  const scale = area > 0 && area < 700_000;
  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  return RATIOS.map((ratio) => {
    let [w, h]: number[] = [...ratio.size];
    if (scale) {
      h = Math.sqrt(area * ratio.rh / ratio.rw);
      w = h * ratio.rw / ratio.rh;
      w = roundTo(w, step);
      h = roundTo(h, step);
    }
    return { ...ratio, w: clamp(w), h: clamp(h) };
  });
}

export function CreatePanel({
  projectId, models, workflows, presets, settings, manifest, params, setParams, gallery,
  selectedAssets, setSelectedAssets, uploadProgress, onUpload, setModelId, workflowId, setWorkflowId,
  runtime, setRuntime, maxRuntime, setMaxRuntime, batchCount, setBatchCount, batchMode,
  runtimeChoice, accounts, instances, onRuntimeChoiceChange,
  loading, error, onSubmit, onApplyPreset, onSavePreset,
}: {
  projectId: string;
  models: GpuModel[];
  workflows: GpuWorkflow[];
  presets: GpuPreset[];
  settings: GpuSettings | null;
  manifest: GpuManifest | null;
  params: Record<string, unknown>;
  setParams: (value: Record<string, unknown>) => void;
  gallery: GalleryAsset[];
  selectedAssets: string[];
  setSelectedAssets: (value: string[]) => void;
  uploadProgress: Record<string, number>;
  onUpload: (file: File) => Promise<void>;
  modelId: string;
  setModelId: (value: string) => void;
  workflowId: string;
  setWorkflowId: (value: string) => void;
  runtime: Provider;
  setRuntime: (value: Provider) => void;
  maxRuntime: number;
  setMaxRuntime: (value: number) => void;
  batchCount: number;
  setBatchCount: (n: number) => void;
  batchMode: 'save' | 'smart' | 'batch' | 'retain';
  setBatchMode: (m: 'save' | 'smart' | 'batch' | 'retain') => void;
  runtimeChoice: RuntimeChoice;
  accounts: GpuAccount[];
  instances: GpuInstance[];
  onRuntimeChoiceChange: (value: RuntimeChoice) => void;
  loading: string;
  error: string;
  onSubmit: () => void;
  onApplyPreset: (preset: GpuPreset) => void;
  onSavePreset: (name: string) => void;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const [customSize, setCustomSize] = useState(false);
  const [presetName, setPresetName] = useState('');

  /* ───── 工作流分组：同一个 workflow_key 的所有版本归为一族 ───── */
  const families = useMemo(() => {
    const map = new Map<string, { key: string; modelId: string; items: GpuWorkflow[]; category: Category }>();
    workflows.forEach((wf) => {
      const fam = map.get(wf.workflow_key) || { key: wf.workflow_key, modelId: wf.model_id, items: [], category: isVideoKey(wf.workflow_key) ? 'video' as Category : 'image' as Category };
      fam.items.push(wf);
      map.set(wf.workflow_key, fam);
    });
    map.forEach((fam) => fam.items.sort((a, b) => String(b.workflow_version).localeCompare(String(a.workflow_version), undefined, { numeric: true })));
    return [...map.values()];
  }, [workflows]);

  const activeWorkflow = workflows.find((item) => item.id === workflowId);
  const currentFamily = activeWorkflow ? families.find((f) => f.key === activeWorkflow.workflow_key) : undefined;
  const category: Category = currentFamily?.category || 'image';
  const modelName = (id?: string) => models.find((m) => m.model_id === id)?.display_name || id || '';
  const pickable = (fam: typeof families[number]) => fam.items.some((i) => i.state === 'published') ? fam.items.filter((i) => i.state === 'published') : fam.items;

  function chooseWorkflow(wf: GpuWorkflow) {
    setModelId(wf.model_id);
    setWorkflowId(wf.id);
  }
  function chooseFamily(fam: typeof families[number]) {
    const list = pickable(fam);
    const target = list.find((i) => i.active) || list[0];
    if (target) chooseWorkflow(target);
  }
  function switchCategory(next: Category) {
    if (next === category) return;
    const fam = families.find((f) => f.category === next);
    if (fam) chooseFamily(fam);
  }

  const subtitle = activeWorkflow
    ? String(activeWorkflow.description || activeWorkflow.manifest?.description || manifest?.description || modelName(activeWorkflow.model_id))
    : '';

  /* ───── 提示词、尺寸 ───── */
  const plist = useMemo(() => parameterList(manifest), [manifest]);
  const keys = useMemo(() => plist.map((p, i) => parameterKey(p, i)), [plist]);
  const promptKey = keys.find((k) => k === 'prompt') || keys.find((k) => /prompt|caption/i.test(k) && !/negative/i.test(k)) || 'prompt';
  const hasSize = keys.includes('width') && keys.includes('height');
  const wParam = plist.find((p, i) => parameterKey(p, i) === 'width');
  const hParam = plist.find((p, i) => parameterKey(p, i) === 'height');
  const step = Number(wParam?.step) > 1 ? Number(wParam?.step) : 16;
  const minSide = Math.max(Number(wParam?.min) || 0, Number(hParam?.min) || 0, 256);
  const maxSide = Math.min(Number(wParam?.max) || 2048, Number(hParam?.max) || 2048);
  const sizes = useMemo(
    () => buildSizes(Number(wParam?.default) || 0, Number(hParam?.default) || 0, step, minSide, maxSide),
    [wParam, hParam, step, minSide, maxSide],
  );
  const curW = Number(params.width) || 0;
  const curH = Number(params.height) || 0;
  const matched = sizes.find((s) => s.w === curW && s.h === curH);
  const showCustom = customSize || !matched;

  // 换工作流后重新判断是否为自定义尺寸
  useEffect(() => { setCustomSize(false); }, [workflowId]);

  const excluded = useMemo(() => [promptKey, ...(hasSize ? ['width', 'height'] : [])], [promptKey, hasSize]);
  const split = useMemo(() => splitParams(manifest, excluded), [manifest, excluded]);
  const promptValue = String(params[promptKey] ?? '');
  const maxMinutesCap = typeof settings?.budget?.max_runtime_minutes === 'number' ? settings.budget.max_runtime_minutes : 60;
  const patch = (next: Record<string, unknown>) => setParams({ ...params, ...next });

  if (!projectId) {
    return (
      <div className="empty">
        <strong>请先选择或新建项目</strong>
        <span>点右上角的项目名称，项目用来归档你的参数预设、素材和作品。</span>
      </div>
    );
  }

  return (
    <div className={styles.stack}>
      {/* 1 创作类型 */}
      <Card icon={<Sparkles size={17} strokeWidth={1.6} aria-hidden />} title="创作类型">
        <Segmented<Category>
          ariaLabel="创作类型"
          value={category}
          onChange={switchCategory}
          options={[
            { value: 'image', label: '图片生成', icon: <ImageIcon size={17} strokeWidth={1.6} aria-hidden />, disabled: !families.some((f) => f.category === 'image') },
            { value: 'video', label: '视频生成', icon: <Film size={17} strokeWidth={1.6} aria-hidden />, disabled: !families.some((f) => f.category === 'video') },
          ]}
        />
      </Card>

      {/* 2 工作流 */}
      <Card
        icon={<Layers size={17} strokeWidth={1.6} aria-hidden />}
        title="工作流"
        aside={<button type="button" className={styles.linkButton} onClick={() => setPickerOpen(true)}>切换<ChevronRight size={15} strokeWidth={1.6} aria-hidden /></button>}
      >
        {activeWorkflow ? (
          <button type="button" className={styles.wfCard} onClick={() => setPickerOpen(true)} aria-label="切换工作流">
            <span className={styles.wfThumb}><Layers size={20} strokeWidth={1.4} aria-hidden /></span>
            <span className={styles.wfBody}>
              <span className={styles.wfName}>
                <strong>{workflowTitle(activeWorkflow)}</strong>
                <span className="pill">v{activeWorkflow.workflow_version}</span>
              </span>
              <small className="muted">{activeWorkflow.workflow_key}{subtitle ? ` · ${subtitle}` : ''}</small>
            </span>
          </button>
        ) : <p className="muted small">还没有可用的工作流，请先到「设置 › 工作流与模型」发布一个。</p>}
        <details className={`foldable ${styles.presetBox}`}>
          <summary>参数预设{presets.filter((p) => p.workflow_version_id === workflowId).length ? `（${presets.filter((p) => p.workflow_version_id === workflowId).length}）` : ''}</summary>
          <div className="foldContent">
            <div className="field">
              <label htmlFor="preset-select">套用已存的参数</label>
              <select id="preset-select" defaultValue="" onChange={(e) => {
                const preset = presets.find((p) => p.id === e.target.value);
                if (preset) onApplyPreset(preset);
              }}>
                <option value="">选择预设…</option>
                {presets.filter((p) => p.workflow_version_id === workflowId).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </div>
            <div className={styles.inlineForm}>
              <input aria-label="新预设名称" placeholder="给当前参数起个名字" value={presetName} onChange={(e) => setPresetName(e.target.value)} />
              <button className="button secondary" type="button" disabled={!presetName.trim() || !workflowId} onClick={() => { onSavePreset(presetName.trim()); setPresetName(''); }}>保存</button>
            </div>
          </div>
        </details>
      </Card>

      {/* 3 提示词 */}
      <Card icon={<Pencil size={17} strokeWidth={1.6} aria-hidden />} title="提示词">
        <textarea
          aria-label="提示词"
          value={promptValue}
          placeholder="描述想要的画面：主体、风格、光线、细节……"
          rows={3}
          onChange={(e) => setParams({ ...params, [promptKey]: e.target.value })}
        />
      </Card>

      {/* 4 画面尺寸 —— 全页只有这一套尺寸选择 */}
      {hasSize && (
        <Card
          icon={<SlidersHorizontal size={17} strokeWidth={1.6} aria-hidden />}
          title="画面尺寸"
          aside={<span className={styles.sizeBadge}>{curW || '—'} × {curH || '—'}</span>}
        >
          <div className={styles.ratioGrid} role="group" aria-label="画幅比例">
            {sizes.map((s) => {
              const on = !showCustom && matched?.label === s.label;
              const box = 22;
              const bw = s.rw >= s.rh ? box : Math.round(box * s.rw / s.rh);
              const bh = s.rh >= s.rw ? box : Math.round(box * s.rh / s.rw);
              return (
                <button key={s.label} type="button" className={styles.ratioChip} data-active={on} aria-pressed={on}
                  onClick={() => { setCustomSize(false); patch({ width: s.w, height: s.h }); }}>
                  <span className={styles.ratioShape} aria-hidden><i style={{ width: bw, height: bh }} /></span>
                  <strong>{s.label}</strong>
                  <small>{s.w}×{s.h}</small>
                </button>
              );
            })}
            <button type="button" className={styles.ratioChip} data-active={showCustom} aria-pressed={showCustom} onClick={() => setCustomSize(true)}>
              <span className={styles.ratioShape} aria-hidden><i className={styles.ratioDashed} style={{ width: 22, height: 16 }} /></span>
              <strong>自定义</strong>
              <small>手动输入</small>
            </button>
          </div>
          {showCustom && (
            <div className={styles.customSize}>
              <label className={styles.sizeInput}>
                <span>宽</span>
                <input type="number" inputMode="numeric" min={minSide} max={maxSide} step={step} value={String(params.width ?? '')}
                  onChange={(e) => patch({ width: Number(e.target.value) || 0 })} />
                <em>px</em>
              </label>
              <span className="muted">×</span>
              <label className={styles.sizeInput}>
                <span>高</span>
                <input type="number" inputMode="numeric" min={minSide} max={maxSide} step={step} value={String(params.height ?? '')}
                  onChange={(e) => patch({ height: Number(e.target.value) || 0 })} />
                <em>px</em>
              </label>
            </div>
          )}
          {showCustom && <p className="muted small">建议取 {step} 的倍数，范围 {minSide}–{maxSide}。尺寸越大越慢、越费 CU。</p>}
        </Card>
      )}

      {/* 5 常用选项（时长 / 模式）直接展示 */}
      {split.options.length > 0 && (
        <Card icon={<SlidersHorizontal size={17} strokeWidth={1.6} aria-hidden />} title="生成选项">
          <GpuParamForm section="options" manifest={manifest} values={params} assets={gallery} uploadProgress={uploadProgress}
            onChange={(k, v) => setParams({ ...params, [k]: v })} onUpload={onUpload} excludeKeys={excluded} />
        </Card>
      )}

      {/* 6 素材 */}
      {manifest && (split.assets.length > 0 || supportsReferences(manifest)) && (
        <Card icon={<ImageIcon size={17} strokeWidth={1.6} aria-hidden />} title="参考素材">
          <GpuParamForm section="assets" manifest={manifest} values={params} assets={gallery} uploadProgress={uploadProgress}
            onChange={(k, v) => setParams({ ...params, [k]: v })} onUpload={onUpload} excludeKeys={excluded} />
          <ReferencePicker manifest={manifest} assets={gallery} selected={selectedAssets} onChange={setSelectedAssets} onUpload={onUpload} />
        </Card>
      )}

      {/* 7 数量 */}
      <Card>
        <div className={styles.countRow}>
          <div>
            <strong>生成数量</strong>
            <p className="muted small">一次提交几张/几段，多个会按顺序排队。</p>
          </div>
          <div className={styles.stepper}>
            <button type="button" aria-label="减少数量" disabled={batchCount <= 1} onClick={() => setBatchCount(Math.max(1, batchCount - 1))}><Minus size={16} strokeWidth={1.8} aria-hidden /></button>
            <input aria-label="生成数量" type="number" inputMode="numeric" min={1} max={100} value={batchCount}
              onChange={(e) => setBatchCount(Math.max(1, Math.min(100, Number(e.target.value) || 1)))} />
            <button type="button" aria-label="增加数量" disabled={batchCount >= 100} onClick={() => setBatchCount(Math.min(100, batchCount + 1))}><Plus size={16} strokeWidth={1.8} aria-hidden /></button>
          </div>
        </div>
      </Card>

      {/* 8 高级参数：默认折叠 */}
      {split.advanced.length > 0 && (
        <details className={`foldable ${styles.advanced}`}>
          <summary>
            <span className={styles.advTitle}><SlidersHorizontal size={17} strokeWidth={1.6} aria-hidden />高级参数<span className="pill">{split.advanced.length} 项</span></span>
          </summary>
          <div className="foldContent">
            <p className="muted small">一般不用改。不确定时保持默认即可。</p>
            <GpuParamForm section="advanced" manifest={manifest} values={params} assets={gallery} uploadProgress={uploadProgress}
              onChange={(k, v) => setParams({ ...params, [k]: v })} onUpload={onUpload} excludeKeys={excluded} />
          </div>
        </details>
      )}

      {/* 9 运行设置 */}
      <Card
        icon={<Play size={17} strokeWidth={1.6} aria-hidden />}
        title="运行设置"
        aside={<InfoTip label="运行方式说明" text={<><span className={styles.tipLine}><b>离线模拟</b>：{RUNTIME_HELP.mock}</span><span className={styles.tipLine}><b>Colab GPU</b>：{RUNTIME_HELP.colab}</span></>} />}
      >
        <Segmented<Provider>
          ariaLabel="运行方式"
          value={runtime}
          onChange={setRuntime}
          options={[
            { value: 'mock', label: '离线模拟', sub: '不扣 CU', icon: <Monitor size={17} strokeWidth={1.6} aria-hidden /> },
            { value: 'kaggle', label: 'Kaggle GPU', sub: settings?.real_enabled ? '私有 Dataset 模型' : '未启用', disabled: !settings?.real_enabled, icon: <Cloud size={17} strokeWidth={1.6} aria-hidden /> },
            { value: 'colab', label: 'Colab GPU', sub: settings?.real_enabled ? '真实出图' : '未启用', disabled: !settings?.real_enabled, icon: <Cloud size={17} strokeWidth={1.6} aria-hidden /> },
          ]}
        />
        {runtime !== 'kaggle' && <RuntimeSelection value={runtimeChoice} accounts={accounts} instances={instances} provider={runtime}
          onChange={onRuntimeChoiceChange} />}
        {batchMode === 'retain' && <p className="muted small">保留机器会继续产生 CU；本站只承诺保留到设定截止，Google 仍可能提前回收。</p>}
        {runtime !== 'mock' && (
          <div className="field">
            <label htmlFor="max-runtime">单个任务最长运行（分钟）</label>
            <input id="max-runtime" type="number" inputMode="numeric" min={5} max={maxMinutesCap} value={maxRuntime} onChange={(e) => setMaxRuntime(Number(e.target.value))} />
            <span className={styles.hint}>Kaggle 由 Notebook 时限结束运行；配额及硬件可用性以平台为准。</span>
          </div>
        )}
        <p className="muted small">
          {runtime === 'kaggle' ? '模型从私有 Kaggle Dataset 挂载读取；每个任务独立运行，使用账号免费 GPU 配额。取消后需等待远端结束。' : runtime === 'mock' ? RUNTIME_HELP.mock : `${RUNTIME_HELP.colab}今日限额 ${shown(settings?.budget?.daily_units)} CU。`}
        </p>
        <button className="button big" disabled={!workflowId || loading === 'submit'} type="button" onClick={onSubmit}>
          <Sparkles size={18} strokeWidth={1.8} aria-hidden />{loading === 'submit' ? '提交中…' : '立即生成'}
        </button>
        {error && <p className="status error" role="alert">{error}</p>}
      </Card>

      {/* 工作流选择面板 */}
      {pickerOpen && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setPickerOpen(false)}>
          <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="选择工作流" onClick={(e) => e.stopPropagation()}>
            <div className={styles.cardHead}>
              <h2>选择工作流</h2>
              <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={() => setPickerOpen(false)}><X size={18} strokeWidth={1.6} aria-hidden /></button>
            </div>
            <p className="muted small">同一个工作流的不同版本收在一起，点版本即可切换。{category === 'image' ? '当前显示图片类。' : '当前显示视频类。'}</p>
            <div className={styles.pickList}>
              {families.filter((f) => f.category === category).map((fam) => {
                const list = pickable(fam);
                const here = currentFamily?.key === fam.key;
                return (
                  <div key={fam.key} className={styles.pickItem} data-on={here}>
                    <button type="button" className={styles.pickMain} onClick={() => { chooseFamily(fam); setPickerOpen(false); }}>
                      <span>
                        <strong>{workflowTitle(list[0] || { workflow_key: fam.key })}</strong>
                        <small className="muted">{workflowUsage(list[0] || { workflow_key: fam.key }) || modelName(fam.modelId)}</small>
                        <small className="muted">{fam.key}</small>
                      </span>
                      {here && <Check size={18} strokeWidth={1.8} aria-hidden />}
                    </button>
                    {list.length > 1 && (
                      <div className={styles.versionRow} role="group" aria-label={`${fam.key} 的版本`}>
                        {list.map((wf) => (
                          <button key={wf.id} type="button" data-active={wf.id === workflowId} onClick={() => { chooseWorkflow(wf); setPickerOpen(false); }}>
                            v{wf.workflow_version}{wf.active ? ' · 当前' : ''}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
              {!families.some((f) => f.category === category) && <p className="muted small">这个类型下还没有工作流。</p>}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
