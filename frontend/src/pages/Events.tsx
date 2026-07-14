import { useState } from "react";
import { Bell, ScanFace, HelpCircle, Zap, Trash2, Filter, X, ImageOff } from "lucide-react";
import { useCameras, useEventTypes, useEvents, useInvalidate } from "../hooks/useData";
import { Card, Badge, EmptyState, Skeleton, Dot } from "../components/ui/primitives";
import { ConfirmDialog, Dialog } from "../components/ui/Dialog";
import { api, mediaUrl } from "../lib/api";
import type { AIEvent } from "../lib/types";
import { dateTime, timeAgo, EVENT_LABEL, cx } from "../lib/format";
import { useToast } from "../hooks/useToast";

const TYPE_ICON: Record<string, typeof Bell> = {
  face_recognized: ScanFace, unknown_person: HelpCircle, wiring_error: Zap, component_detected: Zap,
};
function tone(type: string) {
  if (type === "face_recognized") return "green" as const;
  if (type === "unknown_person") return "red" as const;
  if (type === "wiring_error") return "amber" as const;
  return "blue" as const;
}

export default function Events() {
  const [type, setType] = useState("");
  const [cameraId, setCameraId] = useState("");
  const q = `?limit=200${type ? `&type=${type}` : ""}${cameraId ? `&camera_id=${cameraId}` : ""}`;
  const { data, isLoading } = useEvents(q);
  const { data: types } = useEventTypes();
  const { data: cams } = useCameras();
  const invalidate = useInvalidate();
  const { push } = useToast();
  const [clearing, setClearing] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [preview, setPreview] = useState<AIEvent | null>(null);

  const events = data?.events ?? [];
  const doClear = async () => {
    setClearing(true);
    try { await api.clearEvents(); push("Events cleared", "success"); invalidate("events", "event-types", "dashboard"); }
    catch (e: any) { push(e.message, "error"); }
    finally { setClearing(false); setConfirmClear(false); }
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-1.5 text-sm font-semibold text-muted"><Filter className="h-4 w-4" /> Filters</span>
          <select className="input max-w-[200px]" value={type} onChange={(e) => setType(e.target.value)}>
            <option value="">All event types</option>
            {(types?.types ?? []).map((t) => <option key={t.type} value={t.type}>{EVENT_LABEL[t.type] || t.type} ({t.c})</option>)}
          </select>
          <select className="input max-w-[200px]" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
            <option value="">All cameras</option>
            {(cams?.cameras ?? []).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
          {(type || cameraId) && <button className="btn-ghost btn-sm" onClick={() => { setType(""); setCameraId(""); }}><X className="h-3.5 w-3.5" /> Clear</button>}
        </div>
        <button className="btn-outline text-rose-500" onClick={() => setConfirmClear(true)} disabled={!events.length}><Trash2 className="h-4 w-4" /> Clear all events</button>
      </div>

      <Card className="overflow-hidden p-0">
        {isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-16" />)}</div>
        ) : !events.length ? (
          <EmptyState icon={<Bell className="h-10 w-10" />} title="No events" hint="Recognition and system events will appear here as they happen." />
        ) : (
          <div className="divide-y divide-[rgb(var(--border))]">
            {events.map((e) => {
              const Icon = TYPE_ICON[e.type] || Bell; const t = tone(e.type);
              return (
                <div key={e.id} className="flex items-center gap-4 px-4 py-3.5 transition-colors hover:bg-[rgb(var(--surface-2))]">
                  {e.snapshot ? (
                    <button onClick={() => setPreview(e)} className="h-12 w-16 shrink-0 overflow-hidden rounded-lg bg-slate-950">
                      <img src={mediaUrl(e.snapshot)} className="h-full w-full object-cover" />
                    </button>
                  ) : (
                    <div className={cx("grid h-12 w-12 shrink-0 place-items-center rounded-lg",
                      t === "green" ? "bg-emerald-500/15 text-emerald-500" : t === "red" ? "bg-rose-500/15 text-rose-500" : t === "amber" ? "bg-amber-500/15 text-amber-500" : "bg-brand-500/15 text-brand-400")}>
                      <Icon className="h-5 w-5" />
                    </div>
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="truncate font-semibold">{e.label || EVENT_LABEL[e.type] || e.type}</p>
                      <Badge tone={t}>{EVENT_LABEL[e.type] || e.type}</Badge>
                    </div>
                    <p className="mt-0.5 text-xs text-muted">{e.camera_name || e.camera_id || "system"} · {dateTime(e.created_at)}</p>
                  </div>
                  <div className="text-right">
                    {e.confidence != null && <p className="text-sm font-bold tabular-nums">{(e.confidence * 100).toFixed(1)}%</p>}
                    <p className="text-[11px] text-muted">{timeAgo(e.created_at)}</p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      <Dialog open={Boolean(preview)} onClose={() => setPreview(null)} size="md"
        title={preview?.label || (preview ? EVENT_LABEL[preview.type] : "")}
        subtitle={preview ? `${preview.camera_name || preview.camera_id} · ${dateTime(preview.created_at)}` : ""}>
        {preview?.snapshot ? <img src={mediaUrl(preview.snapshot)} className="w-full rounded-xl" /> :
          <div className="grid h-48 place-items-center text-muted"><ImageOff className="h-8 w-8" /></div>}
      </Dialog>

      <ConfirmDialog open={confirmClear} onClose={() => setConfirmClear(false)} onConfirm={doClear} loading={clearing}
        title="Clear all events" confirmLabel="Clear all" message="This permanently removes every logged event. This cannot be undone." />
    </div>
  );
}
