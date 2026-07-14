import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ClipboardCheck, UploadCloud, FileText, Play, CheckCircle2, AlertTriangle, XCircle,
} from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle, Spinner } from "../components/ui/primitives";
import { api, mediaUrl } from "../lib/api";
import type { InspectionRunResult, Mismatch } from "../lib/types";
import { useCameras, useInvalidate } from "../hooks/useData";
import { timeAgo, fmt } from "../lib/format";
import { useToast } from "../hooks/useToast";

function statusTone(s: string): "green" | "amber" | "red" | "gray" {
  const v = s.toLowerCase();
  if (v === "pass" || v === "ok" || v === "passed") return "green";
  if (v === "warning" || v === "warn") return "amber";
  if (v === "fail" || v === "failed") return "red";
  return "gray";
}
function StatusIcon({ s }: { s: string }) {
  const t = statusTone(s);
  if (t === "green") return <CheckCircle2 className="h-3.5 w-3.5" />;
  if (t === "amber") return <AlertTriangle className="h-3.5 w-3.5" />;
  if (t === "red") return <XCircle className="h-3.5 w-3.5" />;
  return <ClipboardCheck className="h-3.5 w-3.5" />;
}

export default function Inspection() {
  const invalidate = useInvalidate();
  const { push } = useToast();
  const { data: cams } = useCameras();
  const references = useQuery({ queryKey: ["references"], queryFn: api.references });
  const inspections = useQuery({ queryKey: ["inspections"], queryFn: api.inspections });

  const fileRef = useRef<HTMLInputElement>(null);
  const [referenceId, setReferenceId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [cameraId, setCameraId] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<InspectionRunResult | null>(null);

  const run = async () => {
    if (!referenceId) return push("Select a reference design", "error");
    if (!file && !cameraId) return push("Choose an image or a camera", "error");
    const form = new FormData();
    form.append("reference_id", referenceId);
    if (file) form.append("file", file); else form.append("camera_id", cameraId);
    form.append("make_pdf", "true");
    setRunning(true);
    try { const r = await api.runInspection(form); setResult(r); push(`Inspection: ${r.status}`, statusTone(r.status) === "green" ? "success" : "warning"); invalidate("inspections"); }
    catch (e: any) { push(e.message, "error"); } finally { setRunning(false); }
  };

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle title="Run an inspection" />
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <label className="label">Reference design</label>
            <select className="input" value={referenceId} onChange={(e) => setReferenceId(e.target.value)}>
              <option value="">Select a reference</option>
              {(references.data?.references ?? []).map((r) => <option key={r.id} value={String(r.id)}>{r.name}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Upload image</label>
            <button className="input flex items-center gap-2 text-left" onClick={() => fileRef.current?.click()}>
              <UploadCloud className="h-4 w-4 text-faint" />
              <span className="truncate">{file ? file.name : "Choose a photo…"}</span>
            </button>
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => { setFile(e.target.files?.[0] ?? null); setCameraId(""); }} />
          </div>
          <div>
            <label className="label">…or camera</label>
            <select className="input" value={cameraId} onChange={(e) => { setCameraId(e.target.value); setFile(null); }}>
              <option value="">Select a camera</option>
              {(cams?.cameras ?? []).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </div>
        </div>
        <button className="mt-4 btn-primary" onClick={run} disabled={running}>
          {running ? <><Spinner className="h-4 w-4" /> Inspecting…</> : <><Play className="h-4 w-4" /> Run inspection</>}
        </button>
      </Card>

      {result && (
        <Card className="p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold uppercase tracking-wide text-muted">Result</h2>
              <Badge tone={statusTone(result.status)}><StatusIcon s={result.status} />{result.status}</Badge>
              <Badge tone={result.n_mismatches ? "amber" : "green"}>{fmt(result.n_mismatches)} mismatch{result.n_mismatches === 1 ? "" : "es"}</Badge>
            </div>
            {result.pdf && <a className="btn-outline btn-sm" href={mediaUrl(result.pdf)} target="_blank" rel="noreferrer"><FileText className="h-3.5 w-3.5" /> PDF report</a>}
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
            <div className="overflow-hidden rounded-xl bg-slate-950/40">
              {result.annotated ? <img src={mediaUrl(result.annotated)} className="w-full object-contain" /> :
                <div className="grid h-64 place-items-center text-muted">No annotated image</div>}
            </div>
            <MismatchList items={result.mismatches ?? []} />
          </div>
        </Card>
      )}

      <Card className="overflow-hidden p-0">
        <div className="border-b p-4"><h2 className="text-sm font-bold uppercase tracking-wide text-muted">Past inspections</h2></div>
        {inspections.isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-14" />)}</div>
        ) : !inspections.data?.inspections.length ? (
          <EmptyState icon={<ClipboardCheck className="h-10 w-10" />} title="No inspections yet" hint="Run an inspection against a reference design." />
        ) : (
          <div className="divide-y divide-[rgb(var(--border))]">
            {inspections.data.inspections.map((it) => (
              <div key={it.id} className="flex items-center gap-3 px-4 py-3.5 hover:bg-[rgb(var(--surface-2))]">
                <Badge tone={statusTone(it.status)}><StatusIcon s={it.status} />{it.status}</Badge>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-semibold">Reference #{it.reference_id ?? "—"} · {it.source || it.camera_id || "upload"}</p>
                  <p className="text-xs text-muted">{fmt(it.n_mismatches)} mismatches · {timeAgo(it.created_at)}</p>
                </div>
                {it.report_path && <a className="btn-outline btn-sm" href={mediaUrl(it.report_path)} target="_blank" rel="noreferrer"><FileText className="h-3.5 w-3.5" /> Report</a>}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function MismatchList({ items }: { items: Mismatch[] }) {
  if (!items.length) {
    return (
      <div className="grid place-items-center rounded-xl surface-2 p-6 text-center">
        <CheckCircle2 className="h-8 w-8 text-emerald-500" />
        <p className="mt-2 text-sm font-semibold">No mismatches</p>
        <p className="text-xs text-muted">The panel matches the reference design.</p>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      {items.map((m, i) => (
        <div key={i} className="rounded-xl surface-2 p-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <span className="text-sm font-semibold">{m.type}</span>
          </div>
          {m.detail && <p className="mt-1 text-xs text-muted">{m.detail}</p>}
          <div className="mt-1.5 flex gap-4 text-xs">
            <span className="text-muted">Expected: <b className="text-[rgb(var(--text))]">{String(m.expected ?? "—")}</b></span>
            <span className="text-muted">Found: <b className="text-[rgb(var(--text))]">{String(m.found ?? "—")}</b></span>
          </div>
        </div>
      ))}
    </div>
  );
}
