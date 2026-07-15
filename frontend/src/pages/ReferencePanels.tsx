import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Layers, UploadCloud, Camera, GraduationCap, ScanSearch, Trash2, Plus,
  CheckCircle2, AlertTriangle, XCircle, Cpu, Cable, CircuitBoard,
} from "lucide-react";
import {
  Card, Badge, EmptyState, Skeleton, SectionTitle, Spinner, StatCard,
} from "../components/ui/primitives";
import { Dialog, ConfirmDialog } from "../components/ui/Dialog";
import { api, mediaUrl } from "../lib/api";
import type { RefPanelDetail, CompareResult, RefError } from "../lib/types";
import { useCameras, useInvalidate } from "../hooks/useData";
import { useToast } from "../hooks/useToast";

function statusBadge(s: string) {
  const map: Record<string, "green" | "amber" | "red" | "blue" | "gray"> = {
    ready: "green", learning: "amber", draft: "gray", error: "red",
    pass: "green", warning: "amber", fail: "red",
  };
  return <Badge tone={map[s] ?? "gray"}>{s}</Badge>;
}

function StatusIcon({ s }: { s: string }) {
  if (s === "pass") return <CheckCircle2 className="h-5 w-5 text-emerald-500" />;
  if (s === "warning") return <AlertTriangle className="h-5 w-5 text-amber-500" />;
  if (s === "fail") return <XCircle className="h-5 w-5 text-red-500" />;
  return null;
}

export default function ReferencePanels() {
  const { push } = useToast();
  const invalidate = useInvalidate();
  const panels = useQuery({ queryKey: ["refPanels"], queryFn: api.refPanels });
  const cameras = useCameras();

  const [selId, setSelId] = useState<number | null>(null);
  const detail = useQuery({
    queryKey: ["refPanel", selId],
    queryFn: () => api.refPanel(selId as number),
    enabled: selId != null,
  });

  const [showCreate, setShowCreate] = useState(false);
  const [delId, setDelId] = useState<number | null>(null);

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        {/* -------- panel list -------- */}
        <Card className="overflow-hidden p-0">
          <div className="flex items-center justify-between border-b p-4">
            <h2 className="text-sm font-bold uppercase tracking-wide text-muted">Reference Panels</h2>
            <button className="btn-primary btn-sm" onClick={() => setShowCreate(true)}>
              <Plus className="h-3.5 w-3.5" /> New
            </button>
          </div>
          {panels.isLoading ? (
            <div className="space-y-2 p-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-16" />)}</div>
          ) : !panels.data?.panels.length ? (
            <EmptyState icon={<Layers className="h-10 w-10" />} title="No reference panels"
              hint="Create one, capture or upload images of a correct panel, then Learn it." />
          ) : (
            <div className="divide-y divide-[rgb(var(--border))]">
              {panels.data.panels.map((p) => (
                <button key={p.id} onClick={() => setSelId(p.id)}
                  className={"flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-[rgb(var(--surface-2))] " +
                    (selId === p.id ? "bg-[rgb(var(--surface-2))]" : "")}>
                  {p.thumbnail
                    ? <img src={mediaUrl(p.thumbnail)} className="h-11 w-11 rounded-lg object-cover" />
                    : <div className="grid h-11 w-11 place-items-center rounded-lg surface-2"><CircuitBoard className="h-5 w-5 text-muted" /></div>}
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-semibold">{p.name} <span className="text-xs text-muted">{p.version}</span></p>
                    <p className="text-xs text-muted">{p.n_images} img · {p.n_components} comp · {p.n_wires} wires</p>
                  </div>
                  {statusBadge(p.status)}
                </button>
              ))}
            </div>
          )}
        </Card>

        {/* -------- selected panel -------- */}
        {selId == null ? (
          <Card className="grid place-items-center p-10">
            <EmptyState icon={<ScanSearch className="h-10 w-10" />} title="Select a reference panel"
              hint="Pick a panel on the left, or create a new one." />
          </Card>
        ) : detail.isLoading || !detail.data ? (
          <Card className="p-6"><Skeleton className="h-64" /></Card>
        ) : (
          <PanelDetail
            panel={detail.data}
            cameras={(cameras.data?.cameras ?? []).map((c) => ({ id: c.id, name: c.name }))}
            onChanged={() => { invalidate("refPanels"); detail.refetch(); }}
            onDelete={() => setDelId(selId)}
          />
        )}
      </div>

      <CreateDialog open={showCreate} onClose={() => setShowCreate(false)}
        onCreated={(id) => { invalidate("refPanels"); setSelId(id); setShowCreate(false); }} />
      <ConfirmDialog open={delId != null} onClose={() => setDelId(null)}
        title="Delete reference panel?"
        message="This permanently removes the panel, its images, learned template and inspection history."
        onConfirm={async () => {
          try { await api.deleteRefPanel(delId as number); push("Panel deleted", "success"); setSelId(null); invalidate("refPanels"); }
          catch (e: any) { push(e.message, "error"); } finally { setDelId(null); }
        }} />
    </div>
  );
}

function PanelDetail({ panel, cameras, onChanged, onDelete }: {
  panel: RefPanelDetail;
  cameras: { id: string; name: string }[];
  onChanged: () => void;
  onDelete: () => void;
}) {
  const { push } = useToast();
  const fileRef = useRef<HTMLInputElement>(null);
  const cmpFileRef = useRef<HTMLInputElement>(null);
  const [cameraId, setCameraId] = useState<string>("");
  const [busy, setBusy] = useState<string>("");
  const [result, setResult] = useState<CompareResult | null>(null);

  const capture = async () => {
    setBusy("capture");
    try { await api.captureRefPanel(panel.id, cameraId || undefined); push("Frame captured", "success"); onChanged(); }
    catch (e: any) { push(e.message, "error"); } finally { setBusy(""); }
  };
  const doUpload = async (files: FileList | null) => {
    if (!files?.length) return;
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    setBusy("upload");
    try { const r = await api.uploadRefPanel(panel.id, form); push(`${r.count} image(s) added`, "success"); onChanged(); }
    catch (e: any) { push(e.message, "error"); } finally { setBusy(""); if (fileRef.current) fileRef.current.value = ""; }
  };
  const learn = async () => {
    setBusy("learn");
    try { await api.learnRefPanel(panel.id); push("Template learned", "success"); onChanged(); }
    catch (e: any) { push(e.message, "error"); } finally { setBusy(""); }
  };
  const compare = async (fromCamera: boolean) => {
    const form = new FormData();
    if (fromCamera) { if (cameraId) form.append("camera_id", cameraId); }
    else { const f = cmpFileRef.current?.files?.[0]; if (!f) return push("Choose an image", "error"); form.append("file", f); }
    setBusy("compare");
    try { const r = await api.compareRefPanel(panel.id, form); setResult(r); push(`Inspection: ${r.status}`, r.status === "pass" ? "success" : r.status === "fail" ? "error" : "warning"); onChanged(); }
    catch (e: any) { push(e.message, "error"); } finally { setBusy(""); if (cmpFileRef.current) cmpFileRef.current.value = ""; }
  };

  const learned = panel.status === "ready";

  return (
    <div className="space-y-5">
      <Card className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-xl font-bold">{panel.name} <span className="text-sm font-normal text-muted">{panel.version}</span></h2>
            {panel.description && <p className="text-sm text-muted">{panel.description}</p>}
          </div>
          <div className="flex items-center gap-2">
            {statusBadge(panel.status)}
            <button className="btn-icon btn-outline" title="Delete" onClick={onDelete}><Trash2 className="h-4 w-4" /></button>
          </div>
        </div>
        {panel.note && <p className="mt-2 rounded-lg surface-2 p-2 text-xs text-muted">{panel.note}</p>}

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard icon={<Layers className="h-4 w-4" />} label="Images" value={panel.n_images} tone="blue" />
          <StatCard icon={<Cpu className="h-4 w-4" />} label="Components" value={panel.n_components} tone="violet" />
          <StatCard icon={<CircuitBoard className="h-4 w-4" />} label="Terminals" value={panel.n_terminals} tone="amber" />
          <StatCard icon={<Cable className="h-4 w-4" />} label="Wires" value={panel.n_wires} tone="green" />
        </div>
      </Card>

      {/* capture / upload / learn */}
      <Card className="p-5">
        <SectionTitle title="Build the reference" />
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <select className="input max-w-[200px]" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
            <option value="">Active camera</option>
            {cameras.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
          <button className="btn-outline" onClick={capture} disabled={!!busy}>
            {busy === "capture" ? <Spinner className="h-4 w-4" /> : <Camera className="h-4 w-4" />} Capture from camera
          </button>
          <button className="btn-outline" onClick={() => fileRef.current?.click()} disabled={!!busy}>
            {busy === "upload" ? <Spinner className="h-4 w-4" /> : <UploadCloud className="h-4 w-4" />} Upload images
          </button>
          <input ref={fileRef} type="file" accept="image/*" multiple className="hidden"
            onChange={(e) => doUpload(e.target.files)} />
          <button className="btn-primary ml-auto" onClick={learn} disabled={!!busy || panel.n_images === 0}>
            {busy === "learn" ? <Spinner className="h-4 w-4" /> : <GraduationCap className="h-4 w-4" />} Learn template
          </button>
        </div>

        {panel.images.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {panel.images.map((img: any) => (
              <img key={img.id} src={mediaUrl(img.path)} title={img.source}
                className="h-16 w-16 rounded-lg object-cover ring-1 ring-[rgb(var(--border))]" />
            ))}
          </div>
        )}
      </Card>

      {/* inspect */}
      <Card className="p-5">
        <SectionTitle title="Inspect against this reference" />
        {!learned && <p className="mt-2 text-sm text-amber-500">Learn the template first to enable inspection.</p>}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button className="btn-outline" onClick={() => compare(true)} disabled={!learned || !!busy}>
            {busy === "compare" ? <Spinner className="h-4 w-4" /> : <Camera className="h-4 w-4" />} Inspect from camera
          </button>
          <button className="btn-outline" onClick={() => cmpFileRef.current?.click()} disabled={!learned || !!busy}>
            <UploadCloud className="h-4 w-4" /> Inspect uploaded image
          </button>
          <input ref={cmpFileRef} type="file" accept="image/*" className="hidden" onChange={() => compare(false)} />
        </div>

        {result && <InspectionView result={result} />}
      </Card>
    </div>
  );
}

function InspectionView({ result }: { result: CompareResult }) {
  return (
    <div className="mt-4 space-y-4">
      <div className="flex items-center gap-3 rounded-xl surface-2 p-3">
        <StatusIcon s={result.status} />
        <div>
          <p className="font-bold">{result.status.toUpperCase()} · score {(result.score * 100).toFixed(0)}%</p>
          <p className="text-xs text-muted">{result.n_errors} errors · {result.n_warnings} warnings
            {result.alignment?.ok ? " · registered" : " · raw coords"}</p>
        </div>
      </div>
      {result.snapshot && (
        <img src={mediaUrl(result.snapshot)} className="w-full rounded-xl ring-1 ring-[rgb(var(--border))]" />
      )}
      {result.errors.length > 0 && (
        <div className="divide-y divide-[rgb(var(--border))] overflow-hidden rounded-xl ring-1 ring-[rgb(var(--border))]">
          {result.errors.map((e: RefError, i: number) => (
            <div key={i} className="flex items-center gap-3 px-3 py-2 text-sm">
              <Badge tone={e.severity === "error" ? "red" : e.severity === "warning" ? "amber" : "blue"}>{e.error_type}</Badge>
              <span className="min-w-0 flex-1 truncate">{e.detail}</span>
              <span className="text-xs text-muted">{(e.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CreateDialog({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: (id: number) => void }) {
  const { push } = useToast();
  const [name, setName] = useState("");
  const [version, setVersion] = useState("v1");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    if (!name.trim()) return push("Name is required", "error");
    setBusy(true);
    try { const p = await api.createRefPanel({ name: name.trim(), version, description }); onCreated(p.id); setName(""); setDescription(""); }
    catch (e: any) { push(e.message, "error"); } finally { setBusy(false); }
  };
  return (
    <Dialog open={open} onClose={onClose} title="New reference panel" size="sm"
      footer={<><button className="btn-outline" onClick={onClose}>Cancel</button>
        <button className="btn-primary" onClick={submit} disabled={busy}>{busy ? <Spinner className="h-4 w-4" /> : null} Create</button></>}>
      <div className="space-y-3">
        <div><label className="label">Panel name</label><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Main Distribution Panel" /></div>
        <div><label className="label">Version</label><input className="input" value={version} onChange={(e) => setVersion(e.target.value)} /></div>
        <div><label className="label">Description</label><textarea className="input" rows={3} value={description} onChange={(e) => setDescription(e.target.value)} /></div>
      </div>
    </Dialog>
  );
}
