import { useState } from "react";
import { Cpu, Gauge, Timer, HardDrive, ScanFace, Boxes, Zap, Waypoints, AlertCircle } from "lucide-react";
import { useAIStatus, useAIMetrics, useEnableModel, useSelectModel, useSetParams } from "../hooks/useData";
import { Card, Badge, Dot, Toggle, Skeleton, SectionTitle } from "../components/ui/primitives";
import { RadialGauge } from "../components/ui/charts";
import type { TaskStatus } from "../lib/types";
import { fmt, STATE_TONE, TASK_META, cx } from "../lib/format";
import { useToast } from "../hooks/useToast";

const TASK_ICON: Record<string, typeof Cpu> = { face: ScanFace, detection: Boxes, components: Zap, wires: Waypoints };

export default function Models() {
  const { data: status, isLoading } = useAIStatus();
  const { data: metrics } = useAIMetrics();

  if (isLoading || !status) {
    return <div className="grid gap-5 lg:grid-cols-2">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-64" />)}</div>;
  }

  const res = status.resources;
  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle title="Compute resources" action={<Badge tone={res.gpu_available ? "green" : "gray"}><HardDrive className="h-3 w-3" />{res.gpu_available ? "GPU available" : "CPU only"}</Badge>} />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <RadialGauge value={res.cpu_percent} label="CPU" color="#3366ff" />
          <RadialGauge value={res.ram_percent} label="RAM" color="#8b5cf6" />
          <div className="flex flex-col items-center justify-center rounded-xl surface-2 p-4">
            <p className="text-2xl font-bold tabular-nums">{fmt(res.ram_used_mb, 0)}</p><p className="text-xs text-muted">MB RAM used</p>
          </div>
          <div className="flex flex-col items-center justify-center rounded-xl surface-2 p-4">
            <p className="text-2xl font-bold tabular-nums">{res.gpu_available ? fmt(res.gpu_percent, 0) + "%" : "N/A"}</p><p className="text-xs text-muted">GPU load</p>
          </div>
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        {Object.entries(status.tasks).map(([task, t]) => (
          <ModelCard key={task} task={task} t={t} metrics={metrics?.tasks?.[task]} />
        ))}
      </div>
    </div>
  );
}

function ModelCard({ task, t, metrics }: { task: string; t: TaskStatus; metrics?: any }) {
  const { push } = useToast();
  const enable = useEnableModel(); const select = useSelectModel(); const setParams = useSetParams();
  const Icon = TASK_ICON[task] || Cpu;
  const meta = TASK_META[task] || { label: task, desc: "" };
  const [threshold, setThreshold] = useState<number | null>(
    t.backend?.params?.threshold != null ? Number(t.backend.params.threshold) : null);

  const fps = metrics?.fps ?? t.metrics.fps;
  const infer = metrics?.avg_inference_ms ?? t.metrics.avg_inference_ms;
  const device = (t.backend?.params?.device || "cpu").toString().toUpperCase();

  const toggle = (v: boolean) =>
    enable.mutate({ task, enabled: v }, {
      onSuccess: (s) => push(`${meta.label} ${s.enabled ? "enabled" : "disabled"}`, "success"),
      onError: (e: any) => push(e.message, "error"),
    });
  const changeBackend = (backend_id: string) =>
    select.mutate({ task, backend_id }, {
      onSuccess: (s) => push(s.state === "error" ? `${meta.label}: ${s.reason || "backend error"}` : `Backend switched to ${backend_id}`, s.state === "error" ? "warning" : "success"),
      onError: (e: any) => push(e.message, "error"),
    });
  const applyThreshold = () => {
    if (threshold == null) return;
    setParams.mutate({ task, params: { threshold } }, {
      onSuccess: () => push("Threshold updated", "success"), onError: (e: any) => push(e.message, "error"),
    });
  };

  const tt = STATE_TONE[t.state] || "gray";
  return (
    <Card className={cx("p-5 animate-slide-up", t.state === "error" && "ring-1 ring-rose-500/40")}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className={cx("grid h-11 w-11 place-items-center rounded-xl", t.enabled ? "bg-brand-500/15 text-brand-400" : "surface-2 text-muted")}>
            <Icon className="h-5 w-5" />
          </div>
          <div>
            <p className="font-bold">{meta.label}</p>
            <p className="text-xs text-muted">{meta.desc}</p>
          </div>
        </div>
        <Toggle checked={t.enabled} onChange={toggle} disabled={enable.isPending} />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Badge tone={tt}><Dot tone={tt} pulse={t.state === "running"} />{t.state}</Badge>
        <Badge tone="gray"><HardDrive className="h-3 w-3" />{device}</Badge>
        {t.backend?.requires_weights && <Badge tone={t.backend.ready ? "green" : "amber"}>{t.backend.ready ? "weights loaded" : "weights required"}</Badge>}
      </div>

      {t.state === "error" && t.detail && (
        <div className="mt-3 flex items-start gap-2 rounded-lg bg-rose-500/10 p-2.5 text-xs text-rose-400">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span>{t.detail}</span>
        </div>
      )}

      <div className="mt-4 grid grid-cols-2 gap-3">
        <Metric icon={<Gauge className="h-4 w-4" />} label="Throughput" value={`${fmt(fps, 1)} fps`} />
        <Metric icon={<Timer className="h-4 w-4" />} label="Inference" value={infer != null ? `${fmt(infer, 1)} ms` : "—"} />
      </div>

      <div className="mt-4 space-y-3 border-t pt-4">
        <div>
          <label className="label">Backend</label>
          <select className="input" value={t.selected_backend} onChange={(e) => changeBackend(e.target.value)} disabled={select.isPending}>
            {t.available_backends.map((b) => (
              <option key={b.backend_id} value={b.backend_id}>{b.display_name}{b.requires_weights ? " (needs weights)" : ""}</option>
            ))}
          </select>
        </div>
        {threshold != null && (
          <div>
            <label className="label">Match threshold — <span className="font-mono text-brand-400">{threshold.toFixed(2)}</span></label>
            <div className="flex items-center gap-3">
              <input type="range" min={0.1} max={0.95} step={0.01} value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-[rgb(var(--surface-2))] accent-brand-600" />
              <button className="btn-outline btn-sm" onClick={applyThreshold} disabled={setParams.isPending}>Apply</button>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-xl surface-2 p-3">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-muted"><span className="text-brand-400">{icon}</span>{label}</div>
      <p className="mt-1 text-lg font-bold tabular-nums">{value}</p>
    </div>
  );
}
