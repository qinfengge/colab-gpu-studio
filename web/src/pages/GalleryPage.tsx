import { useState, useMemo } from 'react';
import { Image as ImageIcon, Film, Search, Star, Check, Eye, RotateCcw, Trash2, Play } from 'lucide-react';
import { apiFetch, apiMessage, assetUrl, type GalleryAsset } from '../lib/gpu-api';
import styles from '../gpu.module.css';

type Filter = 'all' | 'image' | 'video' | 'favorite';

function dims(asset: GalleryAsset): string {
  const meta = (asset.metadata || {}) as Record<string, unknown>;
  const w = Number(meta.width ?? asset.width);
  const h = Number(meta.height ?? asset.height);
  return w > 0 && h > 0 ? `${w} × ${h}` : '';
}

export function GalleryPanel({
  gallery, onFavorite, onDelete, onOpen, onReuse, onGoCreate,
}: {
  gallery: GalleryAsset[];
  onFavorite: (asset: GalleryAsset) => void;
  onDelete: (asset: GalleryAsset) => void;
  onOpen: (asset: GalleryAsset) => void;
  onReuse?: (asset: GalleryAsset) => void;
  onGoCreate?: () => void;
}) {
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(24);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [syncing, setSyncing] = useState('');
  const [synced, setSynced] = useState<string[]>([]);
  const [syncError, setSyncError] = useState('');
  async function retrySync(id: string) {
    setSyncing(id); setSyncError('');
    try {
      await apiFetch(`/api/assets/${encodeURIComponent(id)}/sync-drive`, {method: 'POST'});
      setSynced((current) => [...current, id]);
    } catch (error) { setSyncError(apiMessage(error)); }
    finally { setSyncing(''); }
  }


  const filtered = useMemo(() => gallery.filter((asset) => {
    const isVideo = asset.mime.startsWith('video/');
    const isImage = asset.mime.startsWith('image/');
    if (filter === 'image' && !isImage) return false;
    if (filter === 'video' && !isVideo) return false;
    if (filter === 'favorite' && !asset.favorite) return false;
    return `${asset.filename || ''} ${asset.source_job_id || ''}`.toLowerCase().includes(query.toLowerCase());
  }), [gallery, filter, query]);

  const selected = useMemo(() => gallery.find((a) => a.id === selectedId) || null, [gallery, selectedId]);

  const chip = (value: Filter, label: string, icon?: React.ReactNode) => (
    <button type="button" className={styles.chip} data-active={filter === value} onClick={() => setFilter(value)}>{icon}{label}</button>
  );

  return (
    <div className={styles.stack}>
      {syncError && <p role="alert" className="status error">{syncError}</p>}
      <div className={styles.pageTitle}>
        <h2><ImageIcon size={20} strokeWidth={1.6} aria-hidden />作品<span className="pill">{gallery.length} 件作品</span></h2>
      </div>

      <label className={styles.searchBox}>
        <Search size={17} strokeWidth={1.6} aria-hidden />
        <input aria-label="搜索作品" placeholder="搜索作品（文件名、任务编号）" value={query} onChange={(e) => { setQuery(e.target.value); setLimit(24); }} />
      </label>

      <div className={styles.chipRow}>
        {chip('all', '全部')}
        {chip('image', '图片')}
        {chip('video', '视频')}
        {chip('favorite', '收藏', <Star size={13} strokeWidth={1.8} aria-hidden />)}
      </div>

      {filtered.length ? (
        <div className={styles.galleryGrid}>
          {filtered.slice(0, limit).map((asset) => {
            const isVideo = asset.mime.startsWith('video/');
            const isJson = asset.mime.includes('json') || asset.filename?.toLowerCase().endsWith('.json');
            const on = selectedId === asset.id;
            const size = dims(asset);
            return (
              <article className={styles.tile} key={asset.id} data-selected={on}>
                <div className={styles.tileMedia}>
                  <button className={styles.tileCover} type="button" aria-label={`查看 ${asset.filename || '作品'}`}
                    onClick={() => { setSelectedId(asset.id); setConfirmDelete(false); onOpen(asset); }}>
                    {isVideo ? (
                      <video src={assetUrl(asset)} preload="metadata" muted playsInline aria-label={asset.filename || '视频'} />
                    ) : isJson ? (
                      <span className={styles.tileJson}>JSON<small>诊断文件</small></span>
                    ) : (
                      <img loading="lazy" src={assetUrl(asset, true)} alt={asset.filename || '作品'} />
                    )}
                  </button>
                  <span className={styles.tileType}>{isVideo ? <Film size={12} strokeWidth={1.8} aria-hidden /> : <ImageIcon size={12} strokeWidth={1.8} aria-hidden />}{isVideo ? '视频' : '图片'}</span>
                  <button type="button" className={styles.tileCheck} data-on={on} aria-label={on ? '取消选中' : '选中'} aria-pressed={on}
                    onClick={() => { setSelectedId(on ? null : asset.id); setConfirmDelete(false); }}>
                    {on && <Check size={13} strokeWidth={2.6} aria-hidden />}
                  </button>
                  {(size || isVideo) && <span className={styles.tileHud}>{isVideo && <Play size={10} strokeWidth={2} aria-hidden />}{size || '视频'}</span>}
                </div>
                <div className={styles.tileMeta}>
                  <div>
                    <strong title={asset.filename || asset.id}>{asset.filename || `作品 ${asset.id.slice(0, 6)}`}</strong>
                    <small className="muted">{asset.source_job_id ? `任务 ${asset.source_job_id.slice(0, 6)}` : '上传素材'}</small>
                  </div>
                  <button type="button" className={styles.starBtn} data-on={Boolean(asset.favorite)} aria-label={asset.favorite ? '取消收藏' : '收藏'} onClick={() => onFavorite(asset)}>
                    <Star size={17} strokeWidth={1.6} fill={asset.favorite ? 'currentColor' : 'none'} aria-hidden />
                  </button>
                </div>
                {asset.drive_sync_status === 'failed' && !synced.includes(asset.id) && (
                  <button className="button small" disabled={Boolean(syncing)} onClick={() => void retrySync(asset.id)}>
                    {syncing === asset.id ? '正在同步…' : 'Drive 未同步 · 重试'}
                  </button>
                )}
              </article>
            );
          })}
        </div>
      ) : (
        <div className="empty">
          <strong>暂无生成作品</strong>
          <span>生成成功的图片和视频会自动保存在这里。</span>
          {onGoCreate && <button className="button small" type="button" onClick={onGoCreate}>去创作</button>}
        </div>
      )}

      {filtered.length > limit && (
        <button className="button secondary block" type="button" onClick={() => setLimit(limit + 24)}>加载更多作品</button>
      )}

      {selected && (
        <div className={styles.floatBar} role="toolbar" aria-label="作品操作">
          <button type="button" onClick={() => onOpen(selected)}><Eye size={17} strokeWidth={1.6} aria-hidden />预览</button>
          <i />
          <button type="button" disabled={!onReuse || !selected.source_job_id} onClick={() => onReuse && onReuse(selected)}><RotateCcw size={17} strokeWidth={1.6} aria-hidden />再次生成</button>
          <i />
          <button type="button" data-danger={confirmDelete} onClick={() => {
            if (!confirmDelete) { setConfirmDelete(true); return; }
            onDelete(selected); setSelectedId(null); setConfirmDelete(false);
          }}><Trash2 size={17} strokeWidth={1.6} aria-hidden />{confirmDelete ? '再点确认删除' : '删除'}</button>
        </div>
      )}
    </div>
  );
}
