import { useState } from "react";
import { Plus, Video, Pencil, Trash2, Sun, Moon, Server, Activity, RadioTower, Save } from "lucide-react";
import { useCameras, useHealth, useInvalidate, useSettings } from "../hooks/useData";
import { Card, Badge, Dot, EmptyState, SectionTitle, Segmented } from "../components/ui/primitives";
import { Dialog, ConfirmDialog } from "../components/ui/Dialog";
import { api } from "../lib/api";
import type { Camera } from "../lib/types";
import { STATE_TONE, fmt } from "../lib/format";
import { useToast } from "../hooks/useToast";
import { useTheme } from "../hooks/useTheme";

export default function Settings() {
  const { data: cams } = useCameras();
  const { data: health } = useHealth();
  const invalidate = useInvalidate();
  const { push } = useToast();
  const { theme, setTheme } = useTheme();

  const [dialogOpen, setDialogOpen] = useState(false);
  const [editing, setEditing] = useState<Camera | null>(null);
  const [toDelete, setToDelete] = useState<Camera | null>(null);
  const [busy, setBusy] = useState(false);

  const cameras = cams?.cameras ?? [];

  const del = async () => {
    if (!toDelete) return; setBusy(true);
    try { await api.deleteCamera(toDelete.id); push("Camera removed", "success"); invalidate("cameras", "dashboard"); }
    catch (e: any) { push(e.message, "error"); } finally { setBusy(false); setToDelete(null); }
  };

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle title="Appearance" />
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold">Theme</p>
            <p className="text-xs text-muted">Choose a light or dark interface</p>
          </div>
          <Segmented value={theme} onChange={(v) => setTheme(v as "dark" | "light")}
            options={[{ value: "light", label: <span className="flex items-center gap-1.5"><Sun className="h-3.5 w-3.5" /> Light</span> },
                      { value: "dark", label: <span className="flex items-center gap-1.5"><Moon className="h-3.5 w-3.5" /> Dark</span> }]} />
        </div>
      </Card>

      <Card className="p-5">
        <SectionTitle title="RTSP cameras" action={<button className="btn-primary btn-sm" onClick={() => { setEditing(null); setDialogOpen(true); }}><Plus className="h-4 w-4" /> Add camera</button>} />
        {!cameras.length ? (
          <EmptyState icon={<Video className="h-9 w-9" />} title="No cameras configured" hint="Add an RTSP camera to start streaming and recognition." />
        ) : (
          <div className="space-y-2.5">
            {cameras.map((c) => (
              <div key={c.id} className="flex flex-wrap items-center gap-3 rounded-xl surface-2 p-3.5">
                <div className="grid h-10 w-10 place-items-center rounded-lg bg-brand-500/15 text-brand-400"><RadioTower className="h-5 w-5" /></div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <p className="truncate font-semibold">{c.name}</p>
                    <Badge tone={STATE_TONE[c.state] || "gray"}><Dot tone={STATE_TONE[c.state] || "gray"} pulse={c.state === "connected"} />{c.state}</Badge>
                  </div>
                  <p className="truncate font-mono text-xs text-muted">{c.url}</p>
                </div>
                <div className="flex items-center gap-1.5">
                  <button className="btn-icon btn-ghost" onClick={() => { setEditing(c); setDialogOpen(true); }}><Pencil className="h-4 w-4" /></button>
                  <button className="btn-icon btn-ghost text-rose-500" onClick={() => setToDelete(c)}><Trash2 className="h-4 w-4" /></button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card className="p-5">
        <SectionTitle title="System" />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Info icon={<Server className="h-4 w-4" />} label="Service" value={health?.service || "—"} />
          <Info icon={<Activity className="h-4 w-4" />} label="Status" value={health?.status || "—"} />
          <Info icon={<Video className="h-4 w-4" />} label="Cameras" value={health ? `${health.cameras_connected}/${health.cameras_total}` : "—"} />
          <Info icon={<RadioTower className="h-4 w-4" />} label="Subscribers" value={fmt(health?.event_subscribers)} />
        </div>
        <p className="mt-3 text-xs text-muted">RTSP-only platform — sources must be <span className="kbd">rtsp://</span> or <span className="kbd">rtsps://</span> URLs. USB / local files are rejected.</p>
      </Card>

      {dialogOpen && <CameraDialog open={dialogOpen} onClose={() => setDialogOpen(false)} onSaved={() => invalidate("cameras", "dashboard")} camera={editing} />}
      <ConfirmDialog open={Boolean(toDelete)} onClose={() => setToDelete(null)} onConfirm={del} loading={busy}
        title="Remove camera" message={<>Remove <b>{toDelete?.name}</b>? The live stream will stop immediately.</>} />
    </div>
  );
}

function Info({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-xl surface-2 p-3.5">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-muted"><span className="text-brand-400">{icon}</span>{label}</div>
      <p className="mt-1 truncate text-sm font-bold capitalize">{value}</p>
    </div>
  );
}

function CameraDialog({ open, onClose, onSaved, camera }: {
  open: boolean; onClose: () => void; onSaved: () => void; camera?: Camera | null;
}) {
  const { push } = useToast();
  const editing = Boolean(camera);
  const [f, setF] = useState({
    id: camera?.id || "", name: camera?.name || "", url: camera?.url || "",
    transport: camera?.transport || "auto",
  });
  const [saving, setSaving] = useState(false);

  const save = async () => {
    if (!f.url.trim().match(/^rtsps?:\/\//i)) return push("URL must be an rtsp:// or rtsps:// address", "error");
    if (!editing && !f.id.trim()) return push("Camera ID is required", "error");
    setSaving(true);
    try {
      if (editing && camera) await api.updateCamera(camera.id, { url: f.url.trim(), name: f.name.trim() || camera.id, transport: f.transport });
      else await api.createCamera({ id: f.id.trim(), name: f.name.trim() || f.id.trim(), url: f.url.trim(), transport: f.transport });
      push(editing ? "Camera updated" : "Camera added", "success"); onSaved(); onClose();
    } catch (e: any) { push(e.message, "error"); } finally { setSaving(false); }
  };

  return (
    <Dialog open={open} onClose={onClose} size="md" title={editing ? `Edit — ${camera?.name}` : "Add RTSP camera"}
      subtitle="Point the platform at any rtsp:// stream"
      footer={<>
        <button className="btn-ghost" onClick={onClose}>Cancel</button>
        <button className="btn-primary" onClick={save} disabled={saving}><Save className="h-4 w-4" /> {editing ? "Save" : "Add camera"}</button>
      </>}>
      <div className="space-y-3.5">
        {!editing && <div><label className="label">Camera ID *</label><input className="input" value={f.id} onChange={(e) => setF({ ...f, id: e.target.value })} placeholder="front_door" /></div>}
        <div><label className="label">Display name</label><input className="input" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="Front Door" /></div>
        <div><label className="label">RTSP URL *</label><input className="input font-mono text-xs" value={f.url} onChange={(e) => setF({ ...f, url: e.target.value })} placeholder="rtsp://user:pass@192.168.1.10:554/stream" /></div>
        <div><label className="label">Transport</label>
          <select className="input" value={f.transport} onChange={(e) => setF({ ...f, transport: e.target.value })}>
            <option value="auto">Auto (TCP then UDP)</option><option value="tcp">TCP</option><option value="udp">UDP</option>
          </select>
        </div>
      </div>
    </Dialog>
  );
}
