import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ScanLine, UploadCloud, FileText, Waypoints, Boxes, Cable, Play,
} from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle, Spinner, StatCard } from "../components/ui/primitives";
import { api, mediaUrl } from "../lib/api";
import type { PanelAnalyzeResult } from "../lib/types";
import { useCameras, useInvalidate } from "../hooks/useData";
import { timeAgo, fmt } from "../lib/format";
import { useToast } from "../hooks/useToast";

export default function PanelAnalysis() {
  const invalidate = useInvalidate();
  const { push } = useToast();
  const { data: cams } = useCameras();
  const panels = useQuery({ queryKey: ["panels"], queryFn: api.panels });

  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [cameraId, setCameraId] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PanelAnalyzeResult | null>(null);

  const run = async () => {
    if (!file && !cameraId) return push("Choose an image or a camera", "error");
    const form = new FormData();
    if (file) form.append("file", file); else form.append("camera_id", cameraId);
    form.append("make_pdf", "true");
    setRunning(true);
    try { const r = await api.analyzePanel(form); setResult(r); push("Panel analysed", "success"); invalidate("panels"); }
    catch (e: any) { push(e.message, "error"); } finally { setRunning(false); }
  };

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle title="Analyse a control panel" />
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="label">Upload image</label>
            <button className="input flex items-center gap-2 text-left" onClick={() => fileRef.current?.click()}>
              <UploadCloud className="h-4 w-4 text-faint" />
              <span className="truncate">{file ? file.name : "Choose a panel photo…"}</span>
            </button>
            <input ref={fileRef} type="file" accept="image/*" className="hidden"
              onChange={(e) => { setFile(e.target.files?.[0] ?? null); setCameraId(""); }} />
          </div>
          <div>
            <label className="label">…or capture from camera</label>
            <select className="input" value={cameraId} onChange={(e) => { setCameraId(e.target.value); setFile(null); }}>
              <option value="">Select a camera</option>
              {(cams?.cameras ?? []).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </div>
        </div>
        <button className="mt-4 btn-primary" onClick={run} disabled={running}>
          {running ? <><Spinner className="h-4 w-4" /> Analysing…</> : <><Play className="h-4 w-4" /> Run analysis</>}
        </button>
      </Card>

      {result && <PanelResultView res={result} />}

      <Card className="overflow-hidden p-0">
        <div className="border-b p-4"><h2 className="text-sm font-bold uppercase tracking-wide text-muted">Past panel reports</h2></div>
        {panels.isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-14" />)}</div>
        ) : !panels.data?.panels.length ? (
          <EmptyState icon={<ScanLine className="h-10 w-10" />} title="No panel reports yet" hint="Run an analysis to generate a report." />
        ) : (
          <div className="divide-y divide-[rgb(var(--border))]">
            {panels.data.panels.map((p) => (
              <div key={p.id} className="flex items-center gap-3 px-4 py-3.5 hover:bg-[rgb(var(--surface-2))]">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-brand-500/15 text-brand-400"><ScanLine className="h-5 w-5" /></div>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-semibold">{p.title}</p>
                  <p className="truncate text-xs text-muted">{p.summary || "—"} · {timeAgo(p.created_at)}</p>
                </div>
                <a className="btn-outline btn-sm" href={mediaUrl(p.path)} target="_blank" rel="noreferrer"><FileText className="h-3.5 w-3.5" /> Report</a>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function PanelResultView({ res }: { res: PanelAnalyzeResult }) {
  const r = res.result;
  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-bold uppercase tracking-wide text-muted">Analysis result</h2>
        {res.pdf && <a className="btn-outline btn-sm" href={mediaUrl(res.pdf)} target="_blank" rel="noreferrer"><FileText className="h-3.5 w-3.5" /> PDF report</a>}
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <div className="overflow-hidden rounded-xl bg-slate-950/40">
          {res.annotated ? <img src={mediaUrl(res.annotated)} className="w-full object-contain" /> :
            <div className="grid h-64 place-items-center text-muted">No annotated image</div>}
        </div>
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <StatCard icon={<Boxes className="h-5 w-5" />} label="Components" value={fmt(r.component_total)} tone="blue" />
            <StatCard icon={<Cable className="h-5 w-5" />} label="Wires" value={fmt(r.wire_total)} tone="violet" />
            <StatCard icon={<Waypoints className="h-5 w-5" />} label="Nodes" value={fmt(r.topology?.nodes)} sub={`${fmt(r.topology?.edges)} edges`} tone="green" />
          </div>
          <CountBlock title="Component counts" counts={r.component_counts} />
          <CountBlock title="Wire colour counts" counts={r.wire_color_counts} colored />
        </div>
      </div>

      {r.notes?.length > 0 && (
        <div className="mt-4 rounded-xl surface-2 p-3.5">
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted">Notes</p>
          <ul className="space-y-1 text-sm">{r.notes.map((n, i) => <li key={i} className="text-muted">• {n}</li>)}</ul>
        </div>
      )}
    </Card>
  );
}

const WIRE_COLOR: Record<string, string> = {
  red: "#ef4444", black: "#334155", blue: "#3366ff", green: "#22c55e", yellow: "#eab308",
  white: "#e2e8f0", orange: "#f97316", brown: "#92400e", grey: "#94a3b8", gray: "#94a3b8",
};

function CountBlock({ title, counts, colored }: { title: string; counts: Record<string, number>; colored?: boolean }) {
  const entries = Object.entries(counts || {});
  if (!entries.length) return null;
  return (
    <div className="rounded-xl surface-2 p-3.5">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">{title}</p>
      <div className="flex flex-wrap gap-2">
        {entries.map(([k, v]) => (
          <span key={k} className="badge-gray">
            {colored && <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: WIRE_COLOR[k.toLowerCase()] || "#94a3b8" }} />}
            {k} · {fmt(v)}
          </span>
        ))}
      </div>
    </div>
  );
}
