export type JsonRecord = Record<string, unknown>;

export interface PreviewNode {
  id: string;
  label: string;
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  inputs: JsonRecord;
}

export interface PreviewEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface PreviewGraph {
  nodes: PreviewNode[];
  edges: PreviewEdge[];
  width: number;
  height: number;
  source: 'ui' | 'api';
}

function record(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {};
}

function text(value: unknown, fallback: string): string {
  return value === undefined || value === null || value === '' ? fallback : String(value);
}

function nodeId(value: unknown, fallback: string): string {
  return text(value, fallback);
}

function linkSource(value: unknown): string | null {
  if (Array.isArray(value) && value.length > 0 && (typeof value[0] === 'string' || typeof value[0] === 'number')) return String(value[0]);
  if (value && typeof value === 'object') {
    const item = record(value);
    const source = item.node_id ?? item.node ?? item.source;
    if (typeof source === 'string' || typeof source === 'number') return String(source);
  }
  return null;
}

function rawNodes(graph: unknown): { nodes: JsonRecord[]; source: 'ui' | 'api' } {
  const root = record(graph);
  if (Array.isArray(root.nodes)) return { nodes: root.nodes.map(record), source: 'ui' };
  return {
    nodes: Object.entries(root).map(([id, value]) => ({ id, ...record(value) })),
    source: 'api',
  };
}

function rawPosition(node: JsonRecord): [number, number] | null {
  const value = node.pos ?? node.position;
  if (Array.isArray(value) && value.length >= 2 && Number.isFinite(Number(value[0])) && Number.isFinite(Number(value[1]))) {
    return [Number(value[0]), Number(value[1])];
  }
  return null;
}

function autoLayout(nodes: PreviewNode[], edges: PreviewEdge[]): void {
  const ids = new Set(nodes.map((item) => item.id));
  const incoming = new Map<string, number>();
  const outgoing = new Map<string, string[]>();
  ids.forEach((id) => { incoming.set(id, 0); outgoing.set(id, []); });
  edges.forEach((edge) => {
    if (!ids.has(edge.source) || !ids.has(edge.target)) return;
    incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1);
    outgoing.get(edge.source)?.push(edge.target);
  });
  const levels = new Map<string, number>();
  let frontier = nodes.filter((item) => incoming.get(item.id) === 0).map((item) => item.id);
  if (!frontier.length && nodes.length) frontier = [nodes[0].id];
  const seen = new Set<string>();
  while (frontier.length) {
    const next: string[] = [];
    frontier.forEach((id) => {
      if (seen.has(id)) return;
      seen.add(id);
      const level = levels.get(id) || 0;
      outgoing.get(id)?.forEach((target) => {
        levels.set(target, Math.max(levels.get(target) || 0, level + 1));
        next.push(target);
      });
    });
    frontier = next;
  }
  nodes.forEach((item, index) => {
    const level = levels.get(item.id) || 0;
    const row = nodes.filter((candidate) => (levels.get(candidate.id) || 0) === level).findIndex((candidate) => candidate.id === item.id);
    const fallbackRow = row < 0 ? index : row;
    item.x = 40 + level * 260;
    item.y = 40 + fallbackRow * 128;
  });
}

export function normalisePreviewGraph(graph: unknown, source?: 'ui' | 'api'): PreviewGraph {
  const raw = rawNodes(graph);
  const nodes: PreviewNode[] = raw.nodes.map((item, index) => {
    const id = nodeId(item.id ?? item._id ?? item.node_id, String(index + 1));
    const position = rawPosition(item);
    return {
      id,
      label: text(item.title ?? item.name ?? item.class_type ?? item.type, `节点 ${id}`),
      type: text(item.class_type ?? item.type ?? item.title, '未知节点'),
      x: position?.[0] ?? 0,
      y: position?.[1] ?? 0,
      width: Number(item.width) > 40 ? Number(item.width) : 200,
      height: Number(item.height) > 50 ? Number(item.height) : 76,
      inputs: record(item.inputs),
    };
  });
  const ids = new Set(nodes.map((item) => item.id));
  const edges: PreviewEdge[] = [];
  nodes.forEach((node) => {
    Object.entries(node.inputs).forEach(([key, value]) => {
      const sourceId = linkSource(value);
      if (sourceId && ids.has(sourceId)) edges.push({ id: `${sourceId}-${node.id}-${key}`, source: sourceId, target: node.id, label: key });
    });
  });
  if (raw.source === 'ui') {
    const root = record(graph);
    const links = Array.isArray(root.links) ? root.links : [];
    links.forEach((link, index) => {
      const item: JsonRecord = Array.isArray(link)
        ? { source: link[1], target: link[3] }
        : record(link);
      const sourceId = text(item.source ?? item.origin_id, '');
      const targetId = text(item.target ?? item.target_id, '');
      if (ids.has(sourceId) && ids.has(targetId) && !edges.some((edge) => edge.source === sourceId && edge.target === targetId)) {
        edges.push({ id: `ui-${index}`, source: sourceId, target: targetId });
      }
    });
  }
  if (raw.source === 'api' || nodes.some((item) => item.x === 0 && item.y === 0)) autoLayout(nodes, edges);
  const maxX = Math.max(700, ...nodes.map((item) => item.x + item.width + 80));
  const maxY = Math.max(360, ...nodes.map((item) => item.y + item.height + 80));
  return { nodes, edges, width: maxX, height: maxY, source: source || raw.source };
}

export function previewFileName(workflowKey: string, version: string | number): string {
  const safe = workflowKey.replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '') || 'workflow';
  return `${safe}-v${version}-preview.json`;
}
