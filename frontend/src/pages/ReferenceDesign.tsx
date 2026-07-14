import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Layers, UploadCloud, Trash2, FileText, Image as ImageIcon, ExternalLink, Save, Settings2,
} from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle, Spinner } from "../components/ui/primitives";
import { ConfirmDialog, Dialog } from "../components/ui/Dialog";
import { api, mediaUrl } from "../lib/api";
import type { ReferenceDesign as Ref, ReferenceSpec } from "../lib/types";
import { timeAgo } from "../lib/format";
import { useInvalidate } from "../hooks/useData";
import { useToast } from "../hooks/useToast";

const IMAGE_KINDS = ["png", "jpg", "jpeg", "image"];

export default function ReferenceDesign() {
  const invalidate = useInvalidate();
  const { push } = useToast();
  const list = useQuery({ queryKey: ["references"], queryFn: api.references });

  const fileRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [uploading, setUploading] = useState(false);
  const [toDelete, setToDelete] = useState<Ref | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [specFor, setSpecFor] = useState<Ref | null>(null);

  const refs = list.data?.references ?? [];

  const doUpload = async (files: FileList | null) => {
    if (!files || !files.length) return;
    const form = new FormData();
    form.append("file", files[0]);
    if (name.trim()) form.append("name", name.trim());
    if (description.trim()) form.append("description", description.trim());
    setUploading(true);
    try { const r = await api.uploadReference(form); push(`Reference “${r.name}” uploaded`, "success"); setName(""); setDescription(""); invalidate("references"); }
    catch (e: any) { push(e.message, "error"); }
    finally { setUploading(false); if (fileRef.current) fileRef.current.value = ""; }
  };

  const del = async () => {
    if (!toDelete) return; setDeleting(true);
    try { await api.deleteReference(toDelete.id); push("Reference deleted", "success"); invalidate("references"); }
    catch (e: any) { push(e.message, "error"); } finally { setDeleting(false); setToDelete(null); }
  };

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle title="Upload reference design" />
        <div className="grid gap-3 sm:grid-cols-2">
          <div><label className="label">Name (optional)</label><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="panel-rev-A" /></div>
          <div><label className="label">Description (optional)</label><input className="input" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Main distribution panel" /></div>
        </div>
        <div className="mt-3 grid grid-dots rounded-xl border border-dashed p-6 text-center">
          <UploadCloud className="mx-auto mb-2 h-8 w-8 text-faint" />
          <p className="text-sm font-semibold">PDF, PNG, JPG, DXF or DWG</p>
          <button className="mx-auto mt-3 btn-outline btn-sm" disabled={uploading} onClick={() => fileRef.current?.click()}>
            {uploading ? <><Spinner className="h-4 w-4" /> Uploading…</> : <><UploadCloud className="h-4 w-4" /> Choose file</>}
          </button>
          <input ref={fileRef} type="file" accept=".pdf,.png,.jpg,.jpeg,.dxf,.dwg,image/*" className="hidden" onChange={(e) => doUpload(e.target.files)} />
        </div>
      </Card>

      {list.isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-56" />)}</div>
      ) : !refs.length ? (
        <Card className="p-5"><EmptyState icon={<Layers className="h-10 w-10" />} title="No reference designs" hint="Upload a reference design to inspect panels against it." /></Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {refs.map((r) => {
            const isImg = IMAGE_KINDS.includes((r.kind || "").toLowerCase());
            return (
              <Card key={r.id} className="overflow-hidden p-0 animate-slide-up">
                <div className="grid h-40 place-items-center bg-slate-950/40">
                  {isImg ? <img src={mediaUrl(r.path)} className="h-full w-full object-contain" /> :
                    <div className="flex flex-col items-center text-muted"><FileText className="h-10 w-10" /><span className="mt-1 text-xs uppercase">{r.kind}</span></div>}
                </div>
                <div className="p-4">
                  <div className="flex items-center gap-2">
                    <p className="truncate font-semibold">{r.name}</p>
                    <Badge tone="blue">{r.kind}</Badge>
                  </div>
                  {r.description && <p className="mt-0.5 truncate text-xs text-muted">{r.description}</p>}
                  <p className="mt-0.5 text-xs text-faint">{timeAgo(r.created_at)}</p>
                  <div className="mt-3 flex items-center gap-1.5">
                    <a className="btn-outline btn-sm" href={mediaUrl(r.path)} target="_blank" rel="noreferrer">
                      {isImg ? <ImageIcon className="h-3.5 w-3.5" /> : <ExternalLink className="h-3.5 w-3.5" />} Open
                    </a>
                    <button className="btn-outline btn-sm" onClick={() => setSpecFor(r)}><Settings2 className="h-3.5 w-3.5" /> Spec</button>
                    <button className="btn-icon btn-ghost ml-auto text-rose-500" onClick={() => setToDelete(r)}><Trash2 className="h-4 w-4" /></button>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {specFor && <SpecDialog ref_={specFor} onClose={() => setSpecFor(null)} onSaved={() => invalidate("references")} />}
      <ConfirmDialog open={Boolean(toDelete)} onClose={() => setToDelete(null)} onConfirm={del} loading={deleting}
        title="Delete reference" message={<>Delete <b>{toDelete?.name}</b>? This cannot be undone.</>} />
    </div>
  );
}

function SpecDialog({ ref_, onClose, onSaved }: { ref_: Ref; onClose: () => void; onSaved: () => void }) {
  const { push } = useToast();
  const detail = useQuery({ queryKey: ["reference", ref_.id], queryFn: () => api.reference(ref_.id) });
  const initial = JSON.stringify(detail.data?.spec ?? { component_counts: {}, wire_color_counts: {} }, null, 2);
  const [text, setText] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const value = text ?? initial;

  const save = async () => {
    let spec: ReferenceSpec;
    try {
      const parsed = JSON.parse(value);
      spec = {
        component_counts: parsed.component_counts ?? {},
        wire_color_counts: parsed.wire_color_counts ?? {},
      };
    } catch { return push("Invalid JSON — check your syntax", "error"); }
    setSaving(true);
    try { await api.setReferenceSpec(ref_.id, spec); push("Spec saved", "success"); onSaved(); onClose(); }
    catch (e: any) { push(e.message, "error"); } finally { setSaving(false); }
  };

  return (
    <Dialog open onClose={onClose} size="md" title={`Expected spec — ${ref_.name}`}
      subtitle="Define expected component and wire-colour counts (JSON)"
      footer={<>
        <button className="btn-ghost" onClick={onClose}>Cancel</button>
        <button className="btn-primary" onClick={save} disabled={saving || detail.isLoading}><Save className="h-4 w-4" /> Save spec</button>
      </>}>
      {detail.isLoading ? <Skeleton className="h-64" /> : (
        <textarea className="input min-h-[280px] font-mono text-xs" spellCheck={false}
          value={value} onChange={(e) => setText(e.target.value)} />
      )}
    </Dialog>
  );
}
