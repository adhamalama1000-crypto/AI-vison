import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft, Pause, Play, Square, GitCompare, Timer, TrendingUp, Zap, Cpu,
} from "lucide-react";
import { Card, Badge, Skeleton, SectionTitle, Dot, EmptyState } from "../components/ui/primitives";
import { RadialGauge, MultiLineChart } from "../components/ui/charts";
import { api } from "../lib/api";
import type { TrainingMetrics } from "../lib/types";
import { fmt, duration, STATE_TONE } from "../lib/format";
import { useToast } from "../hooks/useToast";

export default function TrainingProgress() {
  const { id = "" } = useParams();
  const { push } = useToast();
  const job = useQuery({
    queryKey: ["training-job", id],
    queryFn: () => api.trainingJob(id),
    // Poll while the job is in a non-terminal state (queued/running/paused);
    // stop once it reaches a terminal state so we don't poll forever.
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s && ["completed", "stopped", "failed"].includes(s) ? false : 1500;
    },
  });

  if (job.isLoading || !job.data) {
    return <div className="space-y-4"><Skeleton className="h-24" /><Skeleton className="h-72" /></div>;
  }

  const d = job.data;
  const running = d.status === "running";
  const paused = d.status === "paused";
  const done = d.status === "completed" || d.status === "stopped" || d.status === "finished";
  const m: TrainingMetrics | null = d.metrics;
  const res = m?.resources;
  const pct = Math.round((d.progress ?? 0) * (d.progress <= 1 ? 100 : 1));

  const history = (d.history ?? []).map((h, i) => ({
    t: h.epoch ?? i + 1,
    train_loss: h.train_loss, val_loss: h.val_loss,
    precision: h.precision, recall: h.recall, f1: h.f1,
  }));

  const act = async (fn: () => Promise<unknown>, label: string) => {
    try { await fn(); push(label, "success"); job.refetch(); } catch (e: any) { push(e.message, "error"); }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link to="/training" className="btn-icon btn-outline"><ArrowLeft className="h-4 w-4" /></Link>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-lg font-bold">{d.name}</h1>
            <Badge tone={STATE_TONE[d.status] || "gray"}><Dot tone={STATE_TONE[d.status] || "gray"} pulse={running} />{d.status}</Badge>
          </div>
          <p className="text-xs text-muted">{d.task}{m?.model ? ` · training ${m.model}` : ""}</p>
        </div>
        <div className="flex items-center gap-1.5">
          {running && <button className="btn-outline btn-sm" onClick={() => act(() => api.pauseTraining(id), "Paused")}><Pause className="h-4 w-4" /> Pause</button>}
          {paused && <button className="btn-outline btn-sm" onClick={() => act(() => api.resumeTraining(id), "Resumed")}><Play className="h-4 w-4" /> Resume</button>}
          {(running || paused) && <button className="btn-outline btn-sm text-rose-500" onClick={() => act(() => api.stopTraining(id), "Stopped")}><Square className="h-4 w-4" /> Stop</button>}
          {done && <Link className="btn-primary btn-sm" to={`/training/${id}/comparison`}><GitCompare className="h-4 w-4" /> Comparison</Link>}
        </div>
      </div>

      <Card className="p-5">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold">Epoch {fmt(m?.epoch)} / {fmt(m?.epochs)}</p>
          <span className="text-sm font-bold tabular-nums text-brand-400">{fmt(pct)}%</span>
        </div>
        <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-[rgb(var(--surface-2))]">
          <div className="h-full rounded-full bg-brand-600 transition-all" style={{ width: `${Math.min(100, pct)}%` }} />
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric icon={<Zap className="h-4 w-4" />} label="Learning rate" value={m?.learning_rate != null ? m.learning_rate.toExponential(2) : "—"} />
          <Metric icon={<Timer className="h-4 w-4" />} label="Elapsed" value={duration(m?.elapsed_s ?? null)} />
          <Metric icon={<Timer className="h-4 w-4" />} label="ETA" value={running ? duration(m?.eta_s ?? null) : "—"} />
          <Metric icon={<TrendingUp className="h-4 w-4" />} label="Accuracy" value={m?.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : "—"} />
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <SectionTitle title="Loss" />
          {history.length ? (
            <MultiLineChart data={history} series={[
              { key: "train_loss", color: "#3366ff", label: "Train loss" },
              { key: "val_loss", color: "#f97316", label: "Val loss" },
            ]} />
          ) : <EmptyState title="Waiting for first epoch…" />}
        </Card>
        <Card className="p-5">
          <SectionTitle title="Precision / Recall / F1" />
          {history.length ? (
            <MultiLineChart data={history} series={[
              { key: "precision", color: "#22c55e", label: "Precision" },
              { key: "recall", color: "#8b5cf6", label: "Recall" },
              { key: "f1", color: "#3366ff", label: "F1" },
            ]} />
          ) : <EmptyState title="Waiting for first epoch…" />}
        </Card>
      </div>

      <Card className="p-5">
        <SectionTitle title="Resources" action={<Badge tone={res?.gpu_available ? "green" : "gray"}><Cpu className="h-3 w-3" />{res?.gpu_available ? "GPU available" : "GPU unavailable"}</Badge>} />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <RadialGauge value={res?.cpu_percent ?? null} label="CPU" color="#3366ff" />
          <RadialGauge value={res?.ram_percent ?? null} label="RAM" color="#8b5cf6" />
          <div className="flex flex-col items-center justify-center rounded-xl surface-2 p-4">
            <p className="text-2xl font-bold tabular-nums">{fmt(res?.ram_used_mb, 0)}</p><p className="text-xs text-muted">MB RAM used</p>
          </div>
          <div className="flex flex-col items-center justify-center rounded-xl surface-2 p-4">
            <p className="text-2xl font-bold tabular-nums">N/A</p><p className="text-xs text-muted">GPU load</p>
          </div>
        </div>
      </Card>
    </div>
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
