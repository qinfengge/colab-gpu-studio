import { useEffect, useMemo, useRef, useState } from 'react';
import type { GpuWorkflow } from '../lib/gpu-api';
import { normalisePreviewGraph, previewFileName, type JsonRecord, type PreviewGraph } from './graphPreviewData';
import styles from './WorkflowGraphPreview.module.css';

export interface WorkflowGraphPreviewData {
  workflow_key: string;
  workflow_version: string | number;
  model_id?: string;
  description?: string;
  reviewed?: boolean;
  published?: boolean;
  state?: string;
  ui_graph?: unknown;
  api_graph?: unknown;
  parameters?: Array<JsonRecord>;
  models?: Array<JsonRecord>;
  [key: string]: unknown;
}

export interface WorkflowGraphPreviewProps {
  workflow: Pick<GpuWorkflow, 'workflow_key' | 'workflow_version'>;
  loadPreview: () => Promise<WorkflowGraphPreviewData>;
  onClose: () => void;
}

type Tab = 'ui' | 'api' | 'json';

type Pan = { x: number; y: number; scale: number };

function valueText(value: unknown): string {
  if (value === undefined || value === null || value === '') return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function record(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.filter((item): item is JsonRecord => Boolean(item && typeof item === 'object' && !Array.isArray(item))) : [];
}

function evidenceLabel(value: unknown): string {
  const item = record(value);
  if (item.label) return String(item.label);
  switch (String(item.status || value)) {
    case 'confirmed': return '已确认';
    case 'confirmed_missing': return '确认缺失';
    case 'not_in_allowlist': return '未列入允许清单';
    case 'not_confirmed': return '未确认';
    default: return '未检测';
  }
}

function downloadPreview(data: WorkflowGraphPreviewData): void {
  const body = JSON.stringify(data, null, 2);
  const blob = new Blob([body], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = previewFileName(data.workflow_key, data.workflow_version);
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function GraphCanvas({ graph, selected, onSelect, pan, setPan }: {
  graph: PreviewGraph;
  selected: string;
  onSelect: (id: string) => void;
  pan: Pan;
  setPan: (next: Pan) => void;
}) {
  const drag = useRef<{ pointerId: number; x: number; y: number; pan: Pan } | null>(null);
  const onPointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, pan };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!drag.current || drag.current.pointerId !== event.pointerId) return;
    setPan({ ...pan, x: drag.current.pan.x + event.clientX - drag.current.x, y: drag.current.pan.y + event.clientY - drag.current.y });
  };
  const stopPointer = () => { drag.current = null; };
  const onWheel = (event: React.WheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    const scale = Math.min(2.4, Math.max(0.45, pan.scale * (event.deltaY > 0 ? 0.9 : 1.1)));
    setPan({ ...pan, scale });
  };
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  return (
    <svg
      className={styles.canvas}
      role="img"
      aria-label="只读工作流图，可缩放和拖动"
      viewBox={`0 0 ${graph.width} ${graph.height}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={stopPointer}
      onPointerCancel={stopPointer}
      onWheel={onWheel}
    >
      <defs><marker id="workflow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" /></marker></defs>
      <g transform={`translate(${pan.x} ${pan.y}) scale(${pan.scale})`}>
        {graph.edges.map((edge) => {
          const source = byId.get(edge.source);
          const target = byId.get(edge.target);
          if (!source || !target) return null;
          const x1 = source.x + source.width;
          const y1 = source.y + source.height / 2;
          const x2 = target.x;
          const y2 = target.y + target.height / 2;
          const curve = Math.max(40, Math.abs(x2 - x1) / 2);
          return <path key={edge.id} className={styles.edge} d={`M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`} markerEnd="url(#workflow-arrow)" />;
        })}
        {graph.nodes.map((node) => (
          <g
            key={node.id}
            className={`${styles.node} ${selected === node.id ? styles.selected : ''}`}
            transform={`translate(${node.x} ${node.y})`}
            role="button"
            tabIndex={0}
            aria-label={`${node.label}，${node.type}`}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => onSelect(node.id)}
            onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onSelect(node.id); }}
          >
            <rect width={node.width} height={node.height} rx="10" />
            <text className={styles.nodeTitle} x="14" y="25">{node.label.slice(0, 29)}</text>
            <text className={styles.nodeType} x="14" y="48">{node.type.slice(0, 31)}</text>
            <text className={styles.nodeId} x={node.width - 12} y="20" textAnchor="end">#{node.id}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

export function WorkflowGraphPreview({ workflow, loadPreview, onClose }: WorkflowGraphPreviewProps) {
  const [data, setData] = useState<WorkflowGraphPreviewData | null>(null);
  const [tab, setTab] = useState<Tab>('ui');
  const [pan, setPan] = useState<Pan>({ x: 0, y: 0, scale: 1 });
  const [selected, setSelected] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    let alive = true;
    setData(null); setError(''); setTab('ui'); setPan({ x: 0, y: 0, scale: 1 }); setSelected('');
    loadPreview().then((next) => { if (alive) setData(next); }).catch((cause: unknown) => {
      if (alive) setError(cause instanceof Error ? cause.message : '读取工作流预览失败');
    });
    return () => { alive = false; };
  }, [loadPreview]);

  const uiGraph = useMemo(() => data && data.ui_graph ? normalisePreviewGraph(data.ui_graph, 'ui') : null, [data]);
  const apiGraph = useMemo(() => data && data.api_graph ? normalisePreviewGraph(data.api_graph, 'api') : null, [data]);
  const activeGraph = tab === 'ui' ? uiGraph : apiGraph;
  const activeNode = activeGraph?.nodes.find((node) => node.id === selected);
  const shownGraph = activeGraph || (tab === 'ui' ? apiGraph : uiGraph);
  const manifest = record(data?.manifest);
  const review = record(data?.review);
  const runtime = record(data?.runtime);
  const gpuDetection = record(runtime.gpu_detection);
  const weightFiles = records(data?.weight_files || manifest.weight_files);
  const nodeReferences = records(data?.node_references);

  return (
    <div className={styles.backdrop} role="presentation" onClick={onClose}>
      <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="工作流图预览" onClick={(event) => event.stopPropagation()}>
        <header className={styles.header}>
          <div><p className={styles.eyebrow}>离线只读预览 · 非原生 ComfyUI 在线 UI · {data?.published === false ? '草稿' : '已发布'}</p><h2>{workflow.workflow_key} · v{workflow.workflow_version}</h2><p className={styles.muted}>{data?.description || '不连接实例，不安装脚本，不创建 GPU。'}</p></div>
          <div className={styles.headerActions}><button type="button" className="button secondary small" onClick={() => data && downloadPreview(data)} disabled={!data}>下载 JSON</button><button type="button" className="button ghost small" onClick={onClose}>关闭</button></div>
        </header>
        {error && <p className={styles.error} role="alert">{error}</p>}
        {!data && !error && <div className={styles.loading}>正在读取受审查图…</div>}
        {data && <>
          <div className={styles.tabs} role="tablist" aria-label="预览类型">
            <button type="button" role="tab" aria-selected={tab === 'ui'} onClick={() => setTab('ui')}>UI workflow</button>
            <button type="button" role="tab" aria-selected={tab === 'api'} onClick={() => setTab('api')}>API graph</button>
            <button type="button" role="tab" aria-selected={tab === 'json'} onClick={() => setTab('json')}>只读 JSON</button>
          </div>
          {tab === 'json' ? <pre className={styles.json}>{JSON.stringify(data, null, 2)}</pre> : <>
            <div className={styles.toolbar}><span className="pill solid">只读</span><span className={styles.muted}>{tab === 'api' ? 'API graph 自动布局 · 结构预览' : uiGraph ? 'UI 节点布局 · 结构预览' : '没有 UI workflow，已回退 API graph'}</span><span className={styles.spacer} /><button type="button" className="button ghost small" onClick={() => setPan({ ...pan, scale: Math.min(2.4, pan.scale + 0.15) })}>放大</button><button type="button" className="button ghost small" onClick={() => setPan({ ...pan, scale: Math.max(0.45, pan.scale - 0.15) })}>缩小</button><button type="button" className="button ghost small" onClick={() => setPan({ x: 0, y: 0, scale: 1 })}>重置</button></div>
            {shownGraph ? <GraphCanvas graph={shownGraph} selected={selected} onSelect={setSelected} pan={pan} setPan={setPan} /> : <div className={styles.empty}>该版本没有可展示的图数据。</div>}
            <div className={styles.detailGrid}>
              <div><h3>参数</h3>{data.parameters?.length ? <dl className={styles.list}>{data.parameters.map((parameter, index) => <div key={String(parameter.key || parameter.id || index)}><dt>{String(parameter.label || parameter.name || parameter.key || `参数 ${index + 1}`)}</dt><dd>{valueText(parameter.default)}{parameter.type ? ` · ${String(parameter.type)}` : ''}{parameter.editable === false ? ' · 只读' : ' · 可编辑（受控工作台）'}</dd></div>)}</dl> : <p className={styles.muted}>没有公开参数。</p>}</div>
              <div><h3>模型</h3>{data.models?.length ? <ul className={styles.modelList}>{data.models.map((model, index) => <li key={String(model.model_id || index)}><strong>{String(model.name || model.display_name || model.model_id || '未命名模型')}</strong><span>{String(model.model_id || '')}{model.is_mock ? ' · Mock' : ''}</span></li>)}</ul> : <p className={styles.muted}>{data.model_id || '没有模型清单。'}</p>}</div>
              <div><h3>Manifest 权重清单</h3>{weightFiles.length ? <ul className={styles.modelList}>{weightFiles.map((weight, index) => { const metadata = record(weight.metadata); const availability = record(weight.availability); return <li key={String(weight.target || weight.filename || index)}><strong>{String(weight.filename || weight.target || `权重 ${index + 1}`)}</strong><span>{String(weight.repo_id || '来源未记录')} · {String(weight.target || '目标未记录')}</span><small>revision：{String(weight.revision || '未记录')}</small><small>sha256：{String(weight.sha256 || '未记录')}</small><small>{evidenceLabel(metadata)} · {evidenceLabel(availability)}</small></li>; })}</ul> : <p className={styles.muted}>Manifest 未保存权重清单。</p>}<p className={styles.muted}>ComfyUI revision：{String(manifest.comfyui_commit || '未记录')}</p></div>
              <div><h3>节点引用</h3>{nodeReferences.length ? <ul className={styles.modelList}>{nodeReferences.map((node, index) => <li key={String(node.node_id || index)}><strong>#{String(node.node_id || index + 1)} · {String(node.class_type || '未知节点')}</strong><span>inputs：{valueText(node.inputs)}</span></li>)}</ul> : <p className={styles.muted}>没有保存的 API 节点引用。</p>}</div>
              <div><h3>审查结果</h3><dl className={styles.list}><div><dt>工作流</dt><dd>{evidenceLabel(review.workflow)}</dd></div><div><dt>模型与许可</dt><dd>{evidenceLabel(review.model)}</dd></div><div><dt>节点允许清单</dt><dd>{evidenceLabel(review.nodes)}</dd></div><div><dt>GPU 冒烟</dt><dd>{evidenceLabel(review.smoke)}</dd></div></dl><p className={styles.muted}>GPU 检测：{evidenceLabel(gpuDetection)}；离线预览不启动实例或执行节点。</p></div>
              <div><h3>运行边界</h3><p className={styles.muted}>{String(runtime.native_comfy_ui_label || '非原生 ComfyUI 在线 UI')}</p><p className={styles.muted}>此页面只读保存的审查快照，不代表在线 ComfyUI、真实 GPU 或权重已下载。</p></div>
              <div><h3>节点详情</h3>{activeNode ? <><strong>{activeNode.label}</strong><p className={styles.muted}>{activeNode.type} · #{activeNode.id}</p><pre className={styles.nodeInputs}>{JSON.stringify(activeNode.inputs, null, 2)}</pre></> : <p className={styles.muted}>点选节点查看参数。</p>}</div>
            </div>
          </>}
        </>}
      </section>
    </div>
  );
}
