import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  Cpu, Play, Pause, Square, Rocket, GitCompare, ChevronRight, Sparkles,
} from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle, Toggle, Dot } from "../components/ui/primitives";
import { api } from "../lib/api";
import type { TrainingConfig, TrainingJob, TrainingTask } from "../lib/types";
import { fmt, timeAgo, STATE_TONE } from "../lib/format";
import { useInvalidate } from "../hooks/useData";
import { useToast } from "../hooks/useToast";

const DEFAULT_CONFIG: TrainingConfig = {
  epochs: 20, augment: true, hpo: false, hpo_trials: 10,
  learning_rate: 0.001, weight_decay: 0.0005, early_stopping_patience: 5,
  image_size: 224, batch_size: 16,
};

export default function Training() {
  const invalidate = useInvalidate();
  const { push } = useToast();
  const catalog = useQuery({ queryKey: ["training-catalog"], queryFn: api.trainingCatalog });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const jobs = useQuery({ queryKey: ["training-jobs"], queryFn: api.training, refetchInterval: 3000 });

  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState<string>("demo");
  const [task, setTask] = useState<TrainingTask>("classification");
  const [models, setModels] = useState<string[]>([]);
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [starting, setStarting] = useState(false);

  const modelOptions = task === "classification"
    ? (catalog.data?.classification_models ?? [])
    : (catalog.data?.detection_models ?? []);

  const toggleModel = (m: string) =>
    setModels((prev) => prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]);

  const cfg = <K extends keyof TrainingConfig>(k: K, v: TrainingConfig[K]) => setConfig((c) => ({ ...c, [k]: v }));

  const start = async () => {
    if (!name.trim()) return push("Enter a job name", "error");
    if (!models.length) return push("Select at least one model", "error");
    setStarting(true);
    try {
      const res = await api.startTraining({
        name: name.trim(),
        dataset_id: datasetId === "demo" ? null : Number(datasetId),
        task, models, config,
      });
      push(`Training started (job #${res.job_id})`, "success");
      setName(""); setModels([]); invalidate("training-jobs");
    } catch (e: any) { push(e.message, "error"); } finally { setStarting(false); }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
      <Card className="p-5">
        <SectionTitle title="Start a training job" />
        <div className="space-y-4">
          <div>
            <label className="label">Job name</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="components-baseline" />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label">Dataset</label>
              <select className="input" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                <option value="demo">Demo dataset (built-in)</option>
                {(datasets.data?.datasets ?? []).map((d) => <option key={d.id} value={String(d.id)}>{d.name}</option>)}
              </select>
            </div>
            <div>
              <label className="label">Task</label>
              <select className="input" value={task} onChange={(e) => { setTask(e.target.value as TrainingTask); setModels([]); }}>
                <option value="classification">Classification</option>
                <option value="detection">Detection</option>
              </select>
            </div>
          </div>

          <div>
            <label className="label">Models to train &amp; compare</label>
            {catalog.isLoading ? <Skeleton className="h-10" /> : !modelOptions.length ? (
              <p className="text-xs text-muted">No models available for this task.</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {modelOptions.map((m) => (
                  <button key={m} type="button" onClick={() => toggleModel(m)}
                    className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition-all ${models.includes(m) ? "border-brand-600 bg-brand-600 text-white" : "text-muted hover:bg-[rgb(var(--surface-2))]"}`}>
                    {m}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="grid gap-4 border-t pt-4 sm:grid-cols-2">
            <NumField label="Epochs" value={config.epochs} onChange={(v) => cfg("epochs", v)} min={1} />
            <NumField label="Batch size" value={config.batch_size} onChange={(v) => cfg("batch_size", v)} min={1} />
            <NumField label="Learning rate" value={config.learning_rate} onChange={(v) => cfg("learning_rate", v)} step={0.0001} />
            <NumField label="Weight decay" value={config.weight_decay} onChange={(v) => cfg("weight_decay", v)} step={0.0001} />
            <NumField label="Image size" value={config.image_size} onChange={(v) => cfg("image_size", v)} min={32} step={32} />
            <NumField label="Early stopping patience" value={config.early_stopping_patience} onChange={(v) => cfg("early_stopping_patience", v)} min={0} />
          </div>

          <div className="space-y-3 border-t pt-4">
            <ToggleRow label="Data augmentation" desc="Random flips, crops & colour jitter" checked={config.augment} onChange={(v) => cfg("augment", v)} />
            <ToggleRow label="Hyperparameter optimisation" desc="Search for the best hyperparameters" checked={config.hpo} onChange={(v) => cfg("hpo", v)} />
            {config.hpo && <NumField label="HPO trials" value={config.hpo_trials} onChange={(v) => cfg("hpo_trials", v)} min={1} />}
          </div>

          <button className="btn-primary w-full" onClick={start} disabled={starting}>
            <Rocket className="h-4 w-4" /> {starting ? "Starting…" : "Start training"}
          </button>
        </div>
      </Card>

      <Card className="overflow-hidden p-0">
        <div className="border-b p-4"><h2 className="text-sm font-bold uppercase tracking-wide text-muted">Jobs</h2></div>
        {jobs.isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20" />)}</div>
        ) : !jobs.data?.jobs.length ? (
          <EmptyState icon={<Cpu className="h-10 w-10" />} title="No training jobs" hint="Configure and start a job to train and compare models." />
        ) : (
          <div className="divide-y divide-[rgb(var(--border))]">
            {jobs.data.jobs.map((j) => <JobRow key={j.id} job={j} onChanged={() => invalidate("training-jobs")} />)}
          </div>
        )}
      </Card>
    </div>
  );
}

function JobRow({ job, onChanged }: { job: TrainingJob; onChanged: () => void }) {
  const { push } = useToast();
  const running = job.status === "running";
  const paused = job.status === "paused";
  const done = job.status === "completed" || job.status === "finished";
  const pct = Math.round((job.progress ?? 0) * (job.progress <= 1 ? 100 : 1));

  const act = async (fn: () => Promise<unknown>, label: string) => {
    try { await fn(); push(label, "success"); onChanged(); } catch (e: any) { push(e.message, "error"); }
  };

  return (
    <div className="p-4">
      <div className="flex items-center gap-3">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-brand-500/15 text-brand-400"><Sparkles className="h-5 w-5" /></div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Link to={`/training/${job.id}`} className="truncate font-semibold hover:text-brand-400">{job.name}</Link>
            <Badge tone={STATE_TONE[job.status] || "gray"}><Dot tone={STATE_TONE[job.status] || "gray"} pulse={running} />{job.status}</Badge>
          </div>
          <p className="text-xs text-muted">{job.task} · {timeAgo(job.created_at)}{job.best_model ? ` · best: ${job.best_model}` : ""}</p>
        </div>
        <div className="flex items-center gap-1.5">
          {running && <button className="btn-icon btn-ghost" title="Pause" onClick={() => act(() => api.pauseTraining(job.id), "Paused")}><Pause className="h-4 w-4" /></button>}
          {paused && <button className="btn-icon btn-ghost" title="Resume" onClick={() => act(() => api.resumeTraining(job.id), "Resumed")}><Play className="h-4 w-4" /></button>}
          {(running || paused) && <button className="btn-icon btn-ghost text-rose-500" title="Stop" onClick={() => act(() => api.stopTraining(job.id), "Stopped")}><Square className="h-4 w-4" /></button>}
          <Link className="btn-icon btn-ghost" to={`/training/${job.id}`} title="Live progress"><ChevronRight className="h-4 w-4" /></Link>
          {done && <Link className="btn-icon btn-ghost" to={`/training/${job.id}/comparison`} title="Comparison"><GitCompare className="h-4 w-4" /></Link>}
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-[rgb(var(--surface-2))]">
          <div className="h-full rounded-full bg-brand-600 transition-all" style={{ width: `${Math.min(100, pct)}%` }} />
        </div>
        <span className="w-10 text-right text-xs font-semibold tabular-nums text-muted">{fmt(pct)}%</span>
      </div>
    </div>
  );
}

function NumField({ label, value, onChange, min, step }: { label: string; value: number; onChange: (v: number) => void; min?: number; step?: number }) {
  return (
    <div>
      <label className="label">{label}</label>
      <input className="input" type="number" value={value} min={min} step={step ?? 1}
        onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))} />
    </div>
  );
}

function ToggleRow({ label, desc, checked, onChange }: { label: string; desc: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-semibold">{label}</p>
        <p className="text-xs text-muted">{desc}</p>
      </div>
      <Toggle checked={checked} onChange={onChange} />
    </div>
  );
}
