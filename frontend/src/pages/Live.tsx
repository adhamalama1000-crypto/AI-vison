import { useEffect, useState } from "react";
import { Camera as CamIcon, Download, Gauge, Clock, Wifi, WifiOff, RefreshCw } from "lucide-react";
import { useCameras } from "../hooks/useData";
import { CameraStream } from "../components/CameraStream";
import { Card, Segmented, Badge, Dot, EmptyState, Skeleton } from "../components/ui/primitives";
import { snapshotUrl } from "../lib/api";
import { fmt, duration, STATE_TONE, cx } from "../lib/format";
import { useToast } from "../hooks/useToast";

export default function Live() {
  const { data, isLoading } = useCameras();
  const { push } = useToast();
  const [selected, setSelected] = useState<string | null>(null);
  const [mode, setMode] = useState<"raw" | "ai">("ai");

  const cameras = data?.cameras ?? [];
  useEffect(() => {
    if (!selected && cameras.length) setSelected(data?.active_camera || cameras[0].id);
  }, [cameras, data, selected]);

  const cam = cameras.find((c) => c.id === selected) || cameras[0] || null;

  const snapshot = () => {
    if (!cam) return;
    const a = document.createElement("a");
    a.href = snapshotUrl(cam.id); a.download = `${cam.id}-${Date.now()}.jpg`;
    document.body.appendChild(a); a.click(); a.remove();
    push("Snapshot downloaded", "success");
  };

  if (isLoading) return <div className="grid gap-6 lg:grid-cols-3"><Skeleton className="h-[420px] lg:col-span-2" /><Skeleton className="h-[420px]" /></div>;
  if (!cameras.length) return <Card className="p-6"><EmptyState icon={<CamIcon className="h-10 w-10" />} title="No cameras configured" hint="Add an RTSP camera on the Settings page or in config.yaml to start streaming." /></Card>;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <div className="space-y-4 lg:col-span-2">
        <Card className="overflow-hidden p-0">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b p-4">
            <div className="flex items-center gap-3">
              <select className="input max-w-[220px]" value={cam?.id} onChange={(e) => setSelected(e.target.value)}>
                {cameras.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
              {cam && <Badge tone={STATE_TONE[cam.state] || "gray"}><Dot tone={STATE_TONE[cam.state] || "gray"} pulse={cam.state === "connected"} />{cam.state}</Badge>}
            </div>
            <div className="flex items-center gap-2">
              <Segmented value={mode} onChange={setMode} options={[{ value: "ai", label: "AI Overlay" }, { value: "raw", label: "Raw" }]} />
              <button className="btn-outline btn-sm" onClick={snapshot}><Download className="h-4 w-4" /> Snapshot</button>
            </div>
          </div>
          <div className="p-4">
            <CameraStream cameraId={cam?.id ?? null} ai={mode === "ai"} badge={cam?.name} className="w-full" />
          </div>
        </Card>

        {cam && (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <MiniStat icon={<Gauge className="h-4 w-4" />} label="FPS" value={fmt(cam.fps, 1)} />
            <MiniStat icon={<Clock className="h-4 w-4" />} label="Latency" value={cam.latency.avg_ms != null ? `${fmt(cam.latency.avg_ms, 0)} ms` : "—"} />
            <MiniStat icon={cam.healthy ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />} label="Frame age" value={cam.frame_age_ms != null ? `${fmt(cam.frame_age_ms, 0)} ms` : "—"} />
            <MiniStat icon={<RefreshCw className="h-4 w-4" />} label="Uptime" value={duration(cam.connected_for_seconds)} />
          </div>
        )}
      </div>

      <div className="space-y-4">
        <Card className="p-5">
          <h3 className="mb-3 text-sm font-bold uppercase tracking-wide text-muted">All cameras</h3>
          <div className="space-y-2">
            {cameras.map((c) => (
              <button key={c.id} onClick={() => setSelected(c.id)}
                className={cx("flex w-full items-center gap-3 rounded-xl border p-2.5 text-left transition-all",
                  c.id === cam?.id ? "border-brand-500 bg-brand-500/5" : "border-transparent surface-2 hover:border-[rgb(var(--border))]")}>
                <div className="relative h-14 w-24 shrink-0 overflow-hidden rounded-lg bg-slate-950">
                  <img src={snapshotUrl(c.id)} alt={c.name} className="h-full w-full object-cover"
                    onError={(e) => { (e.target as HTMLImageElement).style.opacity = "0"; }} />
                  <span className="absolute left-1 top-1"><Dot tone={STATE_TONE[c.state] || "gray"} pulse={c.state === "connected"} /></span>
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold">{c.name}</p>
                  <p className="truncate text-[11px] text-muted">{c.fps.toFixed(1)} fps · {c.state}</p>
                </div>
              </button>
            ))}
          </div>
        </Card>

        {cam && (
          <Card className="p-5">
            <h3 className="mb-3 text-sm font-bold uppercase tracking-wide text-muted">Connection details</h3>
            <dl className="space-y-2 text-sm">
              <Row k="Transport" v={`${cam.transport}${cam.transport_in_use ? ` (${cam.transport_in_use})` : ""}`} />
              <Row k="Frames captured" v={fmt(cam.statistics.frames_captured)} />
              <Row k="Frames dropped" v={fmt(cam.statistics.frames_dropped)} />
              <Row k="Reconnects" v={fmt(cam.statistics.reconnect_count)} />
              <Row k="Read failures" v={fmt(cam.statistics.read_failures_total)} />
              <Row k="URL" v={<span className="font-mono text-xs">{cam.url}</span>} />
            </dl>
            {cam.statistics.last_error && <div className="mt-3 rounded-lg bg-rose-500/10 p-2.5 text-xs text-rose-400">{cam.statistics.last_error}</div>}
          </Card>
        )}
      </div>
    </div>
  );
}

function MiniStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <Card className="p-4">
      <div className="flex items-center gap-2 text-muted"><span className="text-brand-400">{icon}</span><span className="text-xs font-semibold uppercase tracking-wide">{label}</span></div>
      <p className="mt-1.5 text-xl font-bold tabular-nums">{value}</p>
    </Card>
  );
}
function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return <div className="flex items-center justify-between gap-4"><dt className="text-muted">{k}</dt><dd className="truncate text-right font-medium">{v}</dd></div>;
}
