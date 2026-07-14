import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Database, UploadCloud, FolderOpen, Trash2, RefreshCw, CheckCircle2, AlertTriangle,
  XCircle, FileArchive, Layers, ImageOff,
} from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle, Spinner } from "../components/ui/primitives";
import { ConfirmDialog } from "../components/ui/Dialog";
import { SimpleBarChart } from "../components/ui/charts";
import { api } from "../lib/api";
import type { Dataset, DatasetDetail } from "../lib/types";
import { fmt, timeAgo } from "../lib/format";
import { useInvalidate } from "../hooks/useData";
import { useToast } from "../hooks/useToast";

export default function ElectricalDataset() {
  const invalidate = useInvalidate();
  const { push } = useToast();
  const list = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });

  const filesRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [toDelete, setToDelete] = useState<Dataset | null>(null);
  const [deleting, setDeleting] = useState(false);

  const datasets = list.data?.datasets ?? [];

  const doUpload = async (files: FileList | null) => {
    if (!files || !files.length) return;
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    if (name.trim()) form.append("name", name.trim());
    setUploading(true);
    try {
      const res = await api.uploadDataset(form);
      push(`Dataset “${res.name}” uploaded — ${res.report.ok ? "valid" : "issues found"}`, res.report.ok ? "success" : "warning");
      setName(""); setSelectedId(res.id); invalidate("datasets");
    } catch (e: any) { push(e.message, "error"); }
    finally { setUploading(false); if (filesRef.current) filesRef.current.value = ""; if (folderRef.current) folderRef.current.value = ""; }
  };

  const del = async () => {
    if (!toDelete) return; setDeleting(true);
    try { await api.deleteDataset(toDelete.id); push("Dataset deleted", "success"); if (selectedId === toDelete.id) setSelectedId(null); invalidate("datasets"); }
    catch (e: any) { push(e.message, "error"); } finally { setDeleting(false); setToDelete(null); }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
      <div className="space-y-6">
        <Card className="p-5">
          <SectionTitle title="Upload dataset" />
          <div className="space-y-3">
            <div>
              <label className="label">Dataset name (optional)</label>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="my-components-v1" />
            </div>
            <div className="grid grid-dots rounded-xl border border-dashed p-6 text-center">
              <UploadCloud className="mx-auto mb-2 h-8 w-8 text-faint" />
              <p className="text-sm font-semibold">Upload images, labels or a .zip</p>
              <p className="mt-1 text-xs text-muted">Select individual files, a whole folder, or a single archive.</p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                <button className="btn-outline btn-sm" disabled={uploading} onClick={() => filesRef.current?.click()}>
                  <FileArchive className="h-4 w-4" /> Choose files
                </button>
                <button className="btn-outline btn-sm" disabled={uploading} onClick={() => folderRef.current?.click()}>
                  <FolderOpen className="h-4 w-4" /> Choose folder
                </button>
              </div>
              {uploading && <p className="mt-3 flex items-center justify-center gap-2 text-xs text-muted"><Spinner className="h-4 w-4" /> Uploading & validating…</p>}
            </div>
            <input ref={filesRef} type="file" multiple accept="image/*,.txt,.json,.csv,.yaml,.yml,.zip" className="hidden" onChange={(e) => doUpload(e.target.files)} />
            {/* webkitdirectory: folder upload */}
            <input ref={folderRef} type="file" multiple className="hidden"
              // @ts-expect-error non-standard directory upload attributes
              webkitdirectory="" directory="" onChange={(e) => doUpload(e.target.files)} />
          </div>
        </Card>

        <Card className="overflow-hidden p-0">
          <div className="border-b p-4"><h2 className="text-sm font-bold uppercase tracking-wide text-muted">Datasets</h2></div>
          {list.isLoading ? (
            <div className="space-y-2 p-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-16" />)}</div>
          ) : !datasets.length ? (
            <EmptyState icon={<Database className="h-10 w-10" />} title="No datasets yet" hint="Upload images and labels to validate them for training." />
          ) : (
            <div className="divide-y divide-[rgb(var(--border))]">
              {datasets.map((d) => (
                <button key={d.id} onClick={() => setSelectedId(d.id)}
                  className={`flex w-full items-center gap-3 px-4 py-3.5 text-left transition-colors hover:bg-[rgb(var(--surface-2))] ${selectedId === d.id ? "bg-[rgb(var(--surface-2))]" : ""}`}>
                  <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-brand-500/15 text-brand-400"><Layers className="h-5 w-5" /></div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-semibold">{d.name}</p>
                    <p className="text-xs text-muted">{d.kind} · {fmt(d.n_images)} imgs · {fmt(d.n_classes)} classes · {timeAgo(d.created_at)}</p>
                  </div>
                  <Badge tone={d.status === "valid" ? "green" : d.status === "invalid" ? "red" : "amber"}>{d.status}</Badge>
                  <button className="btn-icon btn-ghost text-rose-500" onClick={(e) => { e.stopPropagation(); setToDelete(d); }}><Trash2 className="h-4 w-4" /></button>
                </button>
              ))}
            </div>
          )}
        </Card>
      </div>

      <DatasetDetailPanel id={selectedId} onChanged={() => invalidate("datasets")} />

      <ConfirmDialog open={Boolean(toDelete)} onClose={() => setToDelete(null)} onConfirm={del} loading={deleting}
        title="Delete dataset" message={<>Delete <b>{toDelete?.name}</b>? This removes all uploaded files permanently.</>} />
    </div>
  );
}

function DatasetDetailPanel({ id, onChanged }: { id: number | null; onChanged: () => void }) {
  const { push } = useToast();
  const [revalidating, setRevalidating] = useState(false);
  const detail = useQuery({ queryKey: ["dataset", id], queryFn: () => api.dataset(id as number), enabled: id != null });

  if (id == null) {
    return <Card className="grid place-items-center p-5"><EmptyState icon={<Database className="h-10 w-10" />} title="Select a dataset" hint="Choose a dataset to view its validation report." /></Card>;
  }
  if (detail.isLoading || !detail.data) return <Card className="p-5"><Skeleton className="h-96" /></Card>;

  const d: DatasetDetail = detail.data;
  const r = d.report;
  const classData = Object.entries(r.class_counts || {}).map(([name, value]) => ({ name, value }));

  const revalidate = async () => {
    setRevalidating(true);
    try { await api.revalidateDataset(id); push("Revalidated", "success"); detail.refetch(); onChanged(); }
    catch (e: any) { push(e.message, "error"); } finally { setRevalidating(false); }
  };

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold">{d.name}</h2>
            <Badge tone={r.ok ? "green" : "red"}>{r.ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}{r.ok ? "valid" : "issues"}</Badge>
          </div>
          <p className="text-xs text-muted">{r.kind} dataset · created {timeAgo(d.created_at)}</p>
        </div>
        <button className="btn-outline btn-sm" onClick={revalidate} disabled={revalidating}>
          <RefreshCw className={`h-4 w-4 ${revalidating ? "animate-spin" : ""}`} /> Revalidate
        </button>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Images" value={fmt(r.n_images)} />
        <Stat label="Labels" value={fmt(r.n_labels)} />
        <Stat label="Classes" value={fmt(r.n_classes)} />
        <Stat label="Imbalance" value={r.imbalance_ratio != null ? `${fmt(r.imbalance_ratio, 2)}×` : "—"} />
      </div>

      {classData.length > 0 && (
        <div className="mt-5">
          <SectionTitle title="Class distribution" />
          <SimpleBarChart data={classData} color="#8b5cf6" height={220} />
        </div>
      )}

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <IssueList icon={<AlertTriangle className="h-4 w-4" />} title="Missing labels" items={r.missing_labels} tone="amber" />
        <IssueList icon={<ImageOff className="h-4 w-4" />} title="Corrupt images" items={r.corrupt_images} tone="red" />
        <IssueList icon={<AlertTriangle className="h-4 w-4" />} title="Warnings" items={r.warnings} tone="amber" />
        <IssueList icon={<XCircle className="h-4 w-4" />} title="Errors" items={r.errors} tone="red" />
      </div>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-xl surface-2 p-3.5">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</p>
      <p className="mt-1 text-xl font-bold tabular-nums">{value}</p>
    </div>
  );
}

function IssueList({ icon, title, items, tone }: { icon: React.ReactNode; title: string; items: string[]; tone: "amber" | "red" }) {
  return (
    <div className="rounded-xl surface-2 p-3.5">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
        <span className={tone === "red" ? "text-rose-500" : "text-amber-500"}>{icon}</span>{title}
        <Badge tone={items.length ? tone : "gray"} className="ml-auto">{items.length}</Badge>
      </div>
      {!items.length ? (
        <p className="text-xs text-muted">None</p>
      ) : (
        <ul className="max-h-32 space-y-1 overflow-y-auto text-xs text-muted">
          {items.slice(0, 50).map((it, i) => <li key={i} className="truncate font-mono">{it}</li>)}
        </ul>
      )}
    </div>
  );
}
