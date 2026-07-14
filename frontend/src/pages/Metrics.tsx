import { useQuery } from "@tanstack/react-query";
import { GaugeCircle, Cpu, Timer, Video, HardDrive, Activity } from "lucide-react";
import { Card, Badge, Skeleton, SectionTitle, Dot, EmptyState } from "../components/ui/primitives";
import { RadialGauge } from "../components/ui/charts";
import { api } from "../lib/api";
import { fmt, STATE_TONE, TASK_META } from "../lib/format";

export default function Metrics() {
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: api.metrics, refetchInterval: 2000 });
  const aiMetrics = useQuery({ queryKey: ["ai-metrics"], queryFn: api.aiMetrics, refetchInterval: 2000 });

  if (metrics.isLoading || !metrics.data) {
    return <div className="space-y-4"><Skeleton className="h-40" /><Skeleton className="h-64" /></div>;
  }

  const m = metrics.data as any;
  const res = m.resources ?? {};
  const aiTasks: Record<string, any> = m.ai ?? aiMetrics.data?.tasks ?? {};
  const cameras: any[] = Array.isArray(m.cameras) ? m.cameras : [];

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle title="Compute resources"
          action={<Badge tone={res.gpu_available ? "green" : "gray"}><HardDrive className="h-3 w-3" />{res.gpu_available ? "GPU available" : "GPU unavailable"}</Badge>} />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <RadialGauge value={res.cpu_percent ?? null} label="CPU" color="#3366ff" />
          <RadialGauge value={res.ram_percent ?? null} label="RAM" color="#8b5cf6" />
          <div className="flex flex-col items-center justify-center rounded-xl surface-2 p-4">
            <p className="text-2xl font-bold tabular-nums">{fmt(res.ram_used_mb, 0)}</p><p className="text-xs text-muted">MB RAM used</p>
          </div>
          <div className="flex flex-col items-center justify-center rounded-xl surface-2 p-4">
            <p className="text-2xl font-bold tabular-nums">{res.gpu_available ? `${fmt(res.gpu_percent, 0)}%` : "N/A"}</p><p className="text-xs text-muted">GPU load</p>
          </div>
        </div>
      </Card>

      <div>
        <SectionTitle title="AI tasks" />
        {!Object.keys(aiTasks).length ? (
          <Card className="p-5"><EmptyState icon={<GaugeCircle className="h-10 w-10" />} title="No AI task metrics" /></Card>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {Object.entries(aiTasks).map(([task, t]) => {
              const meta = TASK_META[task] || { label: task, desc: "" };
              const state = t?.state ?? (t?.enabled ? "running" : "disabled");
              const fps = t?.ai_fps ?? t?.fps;
              const infer = t?.inference_ms ?? t?.avg_inference_ms;
              return (
                <Card key={task} className="p-4 animate-slide-up">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-bold">{meta.label}</p>
                    <Badge tone={STATE_TONE[state] || "gray"}><Dot tone={STATE_TONE[state] || "gray"} pulse={state === "running"} />{state}</Badge>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <div className="rounded-lg surface-2 p-2.5">
                      <div className="flex items-center gap-1 text-[11px] font-semibold text-muted"><Activity className="h-3 w-3 text-brand-400" /> FPS</div>
                      <p className="mt-0.5 text-lg font-bold tabular-nums">{fps != null ? fmt(fps, 1) : "—"}</p>
                    </div>
                    <div className="rounded-lg surface-2 p-2.5">
                      <div className="flex items-center gap-1 text-[11px] font-semibold text-muted"><Timer className="h-3 w-3 text-brand-400" /> Inference</div>
                      <p className="mt-0.5 text-lg font-bold tabular-nums">{infer != null ? `${fmt(infer, 1)}ms` : "—"}</p>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>

      <div>
        <SectionTitle title="Camera capture" />
        {!cameras.length ? (
          <Card className="p-5"><EmptyState icon={<Video className="h-10 w-10" />} title="No cameras" /></Card>
        ) : (
          <Card className="overflow-hidden p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs font-semibold uppercase tracking-wide text-muted">
                    <th className="px-4 py-3">Camera</th>
                    <th className="px-4 py-3">State</th>
                    <th className="px-4 py-3 text-right">FPS</th>
                    <th className="hidden px-4 py-3 text-right sm:table-cell">Captured</th>
                    <th className="hidden px-4 py-3 text-right sm:table-cell">Dropped</th>
                    <th className="hidden px-4 py-3 text-right md:table-cell">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {cameras.map((c, i) => {
                    const stats = c.statistics ?? {};
                    const state = c.state ?? "—";
                    const lat = c.latency?.avg_ms ?? c.latency?.last_ms;
                    return (
                      <tr key={c.id ?? i} className="border-b border-[rgb(var(--border))] last:border-0 hover:bg-[rgb(var(--surface-2))]">
                        <td className="px-4 py-3 font-semibold">{c.name ?? c.id ?? `Camera ${i + 1}`}</td>
                        <td className="px-4 py-3"><Badge tone={STATE_TONE[state] || "gray"}><Dot tone={STATE_TONE[state] || "gray"} pulse={state === "connected"} />{state}</Badge></td>
                        <td className="px-4 py-3 text-right tabular-nums">{fmt(c.fps, 1)}</td>
                        <td className="hidden px-4 py-3 text-right tabular-nums sm:table-cell">{fmt(stats.frames_captured)}</td>
                        <td className="hidden px-4 py-3 text-right tabular-nums sm:table-cell">{fmt(stats.frames_dropped)}</td>
                        <td className="hidden px-4 py-3 text-right tabular-nums md:table-cell">{lat != null ? `${fmt(lat, 0)}ms` : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>

      <p className="flex items-center gap-1.5 text-xs text-faint"><Cpu className="h-3.5 w-3.5" /> Auto-refreshing every 2 seconds.</p>
    </div>
  );
}
