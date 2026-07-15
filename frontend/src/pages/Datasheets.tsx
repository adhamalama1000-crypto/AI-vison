import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FileText, UploadCloud, ScanText, Trash2, Cpu, CircuitBoard, Cable } from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle, Spinner, StatCard } from "../components/ui/primitives";
import { ConfirmDialog } from "../components/ui/Dialog";
import { api, mediaUrl } from "../lib/api";
import type { DatasheetExtract } from "../lib/types";
import { useInvalidate } from "../hooks/useData";
import { timeAgo } from "../lib/format";
import { useToast } from "../hooks/useToast";

export default function Datasheets() {
  const { push } = useToast();
  const invalidate = useInvalidate();
  const list = useQuery({ queryKey: ["datasheets"], queryFn: api.datasheets });
  const fileRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [extract, setExtract] = useState<DatasheetExtract | null>(null);
  const [delId, setDelId] = useState<number | null>(null);

  const doUpload = async (files: FileList | null) => {
    if (!files?.length) return;
    const form = new FormData();
    form.append("file", files[0]);
    setBusy(true);
    try { await api.uploadDatasheet(form); push("Datasheet uploaded", "success"); invalidate("datasheets"); }
    catch (e: any) { push(e.message, "error"); } finally { setBusy(false); if (fileRef.current) fileRef.current.value = ""; }
  };
  const run = async (id: number) => {
    setBusy(true);
    try { const r = await api.extractDatasheet(id); setExtract(r); push(`Extracted via ${r.ocr_engine}`, r.ocr_engine === "none" ? "warning" : "success"); invalidate("datasheets"); }
    catch (e: any) { push(e.message, "error"); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle title="Upload a schematic / datasheet" />
        <p className="mt-1 text-sm text-muted">PDF, PNG, JPG, DXF or SVG. Single-line diagrams and wiring schematics are parsed into component / terminal / wire IDs and an expected connection graph via OCR.</p>
        <button className="mt-3 btn-primary" onClick={() => fileRef.current?.click()} disabled={busy}>
          {busy ? <Spinner className="h-4 w-4" /> : <UploadCloud className="h-4 w-4" />} Upload document
        </button>
        <input ref={fileRef} type="file" accept=".pdf,.png,.jpg,.jpeg,.bmp,.webp,.dxf,.svg" className="hidden"
          onChange={(e) => doUpload(e.target.files)} />
      </Card>

      <Card className="overflow-hidden p-0">
        <div className="border-b p-4"><h2 className="text-sm font-bold uppercase tracking-wide text-muted">Datasheets</h2></div>
        {list.isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-14" />)}</div>
        ) : !list.data?.datasheets.length ? (
          <EmptyState icon={<FileText className="h-10 w-10" />} title="No datasheets" hint="Upload a schematic to extract its expected wiring." />
        ) : (
          <div className="divide-y divide-[rgb(var(--border))]">
            {list.data.datasheets.map((d) => (
              <div key={d.id} className="flex items-center gap-3 px-4 py-3 hover:bg-[rgb(var(--surface-2))]">
                <div className="grid h-10 w-10 place-items-center rounded-lg surface-2"><FileText className="h-5 w-5 text-muted" /></div>
                <div className="min-w-0 flex-1">
                  <a href={mediaUrl(d.path)} target="_blank" rel="noreferrer" className="truncate font-semibold hover:underline">{d.name}</a>
                  <p className="text-xs text-muted">{d.kind} · {timeAgo(d.created_at)}{d.ocr_engine ? ` · ${d.ocr_engine}` : ""}</p>
                </div>
                <Badge tone={d.status === "extracted" ? "green" : "gray"}>{d.status}</Badge>
                <button className="btn-outline btn-sm" onClick={() => run(d.id)} disabled={busy}><ScanText className="h-3.5 w-3.5" /> Extract</button>
                <button className="btn-icon btn-outline" onClick={() => setDelId(d.id)}><Trash2 className="h-4 w-4" /></button>
              </div>
            ))}
          </div>
        )}
      </Card>

      {extract && (
        <Card className="p-5">
          <SectionTitle title="Extraction result" />
          {extract.note && <p className="mt-2 rounded-lg surface-2 p-2 text-xs text-amber-500">{extract.note}</p>}
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard icon={<Cpu className="h-4 w-4" />} label="Components" value={extract.parsed.n_components} tone="violet" />
            <StatCard icon={<CircuitBoard className="h-4 w-4" />} label="Terminals" value={extract.parsed.n_terminals} tone="amber" />
            <StatCard icon={<Cable className="h-4 w-4" />} label="Connections" value={extract.parsed.n_connections} tone="green" />
            <StatCard icon={<ScanText className="h-4 w-4" />} label="Text chars" value={extract.text_chars} tone="blue" />
          </div>
          {extract.parsed.component_ids.length > 0 && (
            <div className="mt-4">
              <p className="mb-1.5 text-xs font-bold uppercase tracking-wide text-muted">Component IDs</p>
              <div className="flex flex-wrap gap-1.5">
                {extract.parsed.component_ids.map((c) => <Badge key={c} tone="blue">{c}</Badge>)}
              </div>
            </div>
          )}
          {extract.parsed.connections.length > 0 && (
            <div className="mt-4">
              <p className="mb-1.5 text-xs font-bold uppercase tracking-wide text-muted">Expected connections</p>
              <div className="flex flex-wrap gap-1.5">
                {extract.parsed.connections.map((c, i) => <Badge key={i} tone="gray">{c.from} → {c.to}</Badge>)}
              </div>
            </div>
          )}
        </Card>
      )}

      <ConfirmDialog open={delId != null} onClose={() => setDelId(null)} title="Delete datasheet?"
        message="This removes the uploaded document and its extraction."
        onConfirm={async () => {
          try { await api.deleteDatasheet(delId as number); push("Deleted", "success"); invalidate("datasheets"); }
          catch (e: any) { push(e.message, "error"); } finally { setDelId(null); }
        }} />
    </div>
  );
}
