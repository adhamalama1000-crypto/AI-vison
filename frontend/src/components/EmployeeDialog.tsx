import { useEffect, useRef, useState } from "react";
import { Camera as CamIcon, Trash2, CheckCircle2, AlertTriangle, XCircle, ScanFace, Loader2 } from "lucide-react";
import { Dialog } from "./ui/Dialog";
import { Badge, Dot } from "./ui/primitives";
import { CameraStream } from "./CameraStream";
import { api, mediaUrl } from "../lib/api";
import type { Camera, Employee, FaceVerdict } from "../lib/types";
import { REASON_TEXT, cx } from "../lib/format";
import { useToast } from "../hooks/useToast";

interface Sample { image: string; preview: string; blur: number | null; multi: boolean; }

export function EmployeeDialog({ open, onClose, onSaved, cameras, activeCamera, employee }: {
  open: boolean; onClose: () => void; onSaved: () => void;
  cameras: Camera[]; activeCamera: string | null; employee?: Employee | null;
}) {
  const { push } = useToast();
  const editing = Boolean(employee);
  const [camId, setCamId] = useState<string>(activeCamera || cameras[0]?.id || "");
  const [form, setForm] = useState({ full_name: "", employee_code: "", department: "", job_title: "" });
  const [samples, setSamples] = useState<Sample[]>([]);
  const [verdict, setVerdict] = useState<{ kind: "ok" | "warn" | "bad" | "info"; text: string } | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [phase, setPhase] = useState<"enroll" | "verify">("enroll");
  const [verifyText, setVerifyText] = useState<{ kind: "ok" | "warn" | "info"; text: string }>({ kind: "info", text: "Stand in front of the camera…" });
  const [existingImages, setExistingImages] = useState<Employee["images"]>([]);
  const verifyTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const savedName = useRef("");

  useEffect(() => {
    if (!open) return;
    setForm({
      full_name: employee?.full_name || "", employee_code: employee?.employee_code || "",
      department: employee?.department || "", job_title: employee?.job_title || "",
    });
    setSamples([]); setVerdict(null); setPhase("enroll");
    if (employee) api.employee(employee.id).then((e) => setExistingImages(e.images)).catch(() => {});
    return () => { if (verifyTimer.current) clearInterval(verifyTimer.current); };
    // Only re-run when the dialog opens or the target employee changes.
    // NOT on camera refetch — the cameras array ref changes every poll and
    // would otherwise wipe the form / captured samples mid-flow.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, employee?.id]);

  // Pick an initial camera once one is available, without resetting the form.
  useEffect(() => {
    if (open && !camId && cameras.length) setCamId(activeCamera || cameras[0].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, cameras, activeCamera]);

  const hasCam = cameras.length > 0;

  const capture = async () => {
    if (!hasCam) return;
    setCapturing(true); setVerdict({ kind: "info", text: "Capturing from camera…" });
    try {
      const v: FaceVerdict = await api.validateCapture({ camera_id: camId });
      if (v.ok) {
        if (editing && employee) {
          const res = await api.addImage(employee.id, { image: v.image });
          if (res.enrollment?.ok) {
            setVerdict({ kind: "ok", text: `Enrolled ✓ (sharpness ${v.blur_score})` });
            push("Face enrolled — recognition is live", "success");
            const e = await api.employee(employee.id); setExistingImages(e.images);
          } else setVerdict({ kind: "bad", text: REASON_TEXT[res.enrollment?.reason] || "Rejected" });
        } else {
          setSamples((s) => [...s, { image: v.image!, preview: v.face_preview || v.image!, blur: v.blur_score, multi: v.multiple_faces }]);
          setVerdict({ kind: v.multiple_faces ? "warn" : "ok", text: v.multiple_faces ? `Captured — multiple faces, largest used (sharpness ${v.blur_score})` : `Captured ✓ (sharpness ${v.blur_score})` });
        }
      } else setVerdict({ kind: "bad", text: REASON_TEXT[v.reason || ""] || v.reason || "Rejected" });
    } catch (e: any) {
      setVerdict({ kind: "bad", text: e.status === 503 ? "Camera has no frame yet — check the RTSP connection." : e.message });
    } finally { setCapturing(false); }
  };

  const removeSample = (i: number) => setSamples((s) => s.filter((_, idx) => idx !== i));

  const deleteExisting = async (imgId: number) => {
    if (!employee) return;
    try { await api.deleteImage(employee.id, imgId); const e = await api.employee(employee.id); setExistingImages(e.images); push("Image removed", "success"); }
    catch (e: any) { push(e.message, "error"); }
  };

  const startVerify = (name: string) => {
    savedName.current = name; setPhase("verify"); setVerifyText({ kind: "info", text: "Stand in front of the camera…" });
    if (verifyTimer.current) clearInterval(verifyTimer.current);
    verifyTimer.current = setInterval(async () => {
      try {
        const a = await api.analyze(camId);
        const faces = a.faces || [];
        const mine = faces.find((f: any) => f.identity === savedName.current);
        if (mine) setVerifyText({ kind: "ok", text: `Recognized ✓ ${mine.identity} · ${(mine.confidence * 100).toFixed(1)}% confidence` });
        else if (faces.length) { const b = faces[0]; setVerifyText({ kind: "warn", text: b.identity ? `Seen: ${b.identity} (${(b.confidence * 100).toFixed(1)}%)` : `Face detected — ${(b.confidence * 100).toFixed(1)}% (unknown)` }); }
        else setVerifyText({ kind: "info", text: "No face in view — step into frame." });
      } catch { /* transient */ }
    }, 900);
  };

  const save = async () => {
    if (saving) return;
    const name = form.full_name.trim();
    if (!name) return push("Full name is required", "error");
    setSaving(true);
    try {
      if (editing && employee) {
        await api.updateEmployee(employee.id, {
          full_name: name, employee_code: form.employee_code.trim() || null,
          department: form.department.trim() || null, job_title: form.job_title.trim() || null,
        });
        push("Employee updated", "success"); onSaved(); onClose();
      } else {
        if (!samples.length) { setSaving(false); return push("Capture at least one valid face from the camera first", "error"); }
        const res = await api.registerEmployee({
          full_name: name, employee_code: form.employee_code.trim() || null,
          department: form.department.trim() || null, job_title: form.job_title.trim() || null,
          images: samples.map((s) => s.image),
        });
        push(`Saved · ${res.enrolled} face(s) enrolled${res.rejected ? `, ${res.rejected} rejected` : ""} · recognition live`, "success");
        onSaved();
        startVerify(res.full_name);
      }
    } catch (e: any) {
      const reasons: string[] = e.data?.error?.details?.reasons || [];
      const hint = reasons.map((r) => REASON_TEXT[r] || r).filter(Boolean)[0];
      push(hint ? `Not saved: ${hint}` : (e.message || "Registration failed"), "error");
    } finally { setSaving(false); }
  };

  const vtone = { ok: "green", warn: "amber", bad: "red", info: "blue" } as const;
  const VIcon = { ok: CheckCircle2, warn: AlertTriangle, bad: XCircle, info: ScanFace };

  return (
    <Dialog open={open} onClose={() => { if (verifyTimer.current) clearInterval(verifyTimer.current); onClose(); }}
      size="lg" title={editing ? `Manage — ${employee?.full_name}` : "Add employee"}
      subtitle="Capture the employee's face directly from the live RTSP camera"
      footer={phase === "verify" ? (
        <button className="btn-primary" onClick={() => { if (verifyTimer.current) clearInterval(verifyTimer.current); onClose(); }}>Done</button>
      ) : (<>
        <button className="btn-ghost" onClick={() => { if (verifyTimer.current) clearInterval(verifyTimer.current); onClose(); }}>Cancel</button>
        <button className="btn-primary" onClick={save} disabled={saving}>
          {saving && <Loader2 className="h-4 w-4 animate-spin" />}{editing ? "Save changes" : "Save employee"}
        </button>
      </>)}>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* left: live camera + capture */}
        <div className="space-y-3">
          {hasCam ? (
            <>
              {cameras.length > 1 && phase === "enroll" && (
                <select className="input" value={camId} onChange={(e) => setCamId(e.target.value)}>
                  {cameras.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              )}
              <CameraStream cameraId={camId} ai={phase === "verify"} showControls={false} badge={phase === "verify" ? "VERIFYING" : undefined} />
              {phase === "enroll" ? (
                <>
                  <button className="btn-primary w-full" onClick={capture} disabled={capturing}>
                    {capturing ? <Loader2 className="h-4 w-4 animate-spin" /> : <CamIcon className="h-4 w-4" />} Capture from camera
                  </button>
                  {verdict && (() => { const I = VIcon[verdict.kind]; return (
                    <div className={cx("flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm font-medium",
                      verdict.kind === "ok" ? "bg-emerald-500/10 text-emerald-500" : verdict.kind === "warn" ? "bg-amber-500/10 text-amber-500" : verdict.kind === "bad" ? "bg-rose-500/10 text-rose-500" : "bg-brand-500/10 text-brand-400")}>
                      <I className="h-4 w-4 shrink-0" /> {verdict.text}
                    </div>); })()}
                </>
              ) : (() => { const I = VIcon[verifyText.kind]; return (
                <div className={cx("flex items-center gap-2 rounded-xl px-3 py-3 text-sm font-semibold",
                  verifyText.kind === "ok" ? "bg-emerald-500/10 text-emerald-500" : verifyText.kind === "warn" ? "bg-amber-500/10 text-amber-500" : "bg-brand-500/10 text-brand-400")}>
                  <I className="h-4 w-4 shrink-0" /> {verifyText.text}
                </div>); })()}
            </>
          ) : (
            <div className="grid aspect-video place-items-center rounded-2xl border border-dashed text-center text-sm text-muted">
              No RTSP camera configured.<br />Add one in Settings — registration captures from the live stream only.
            </div>
          )}

          {/* staged samples (new employee) */}
          {!editing && phase === "enroll" && (
            <div>
              <p className="label">Captured samples ({samples.length})</p>
              {samples.length ? (
                <div className="flex flex-wrap gap-2">
                  {samples.map((s, i) => (
                    <div key={i} className="group relative h-20 w-20 overflow-hidden rounded-xl border">
                      <img src={s.preview} className="h-full w-full object-cover" />
                      {s.multi && <span className="absolute bottom-0.5 left-0.5 rounded bg-amber-500/90 px-1 text-[9px] font-bold text-white">multi</span>}
                      <button onClick={() => removeSample(i)} className="absolute right-0.5 top-0.5 grid h-5 w-5 place-items-center rounded bg-black/60 text-white opacity-0 transition-opacity group-hover:opacity-100"><Trash2 className="h-3 w-3" /></button>
                    </div>
                  ))}
                </div>
              ) : <p className="text-xs text-muted">No samples yet. Click Capture to grab a frame from the live camera.</p>}
            </div>
          )}

          {/* existing images (editing) */}
          {editing && (
            <div>
              <p className="label">Enrolled images ({existingImages.length})</p>
              {existingImages.length ? (
                <div className="flex flex-wrap gap-2">
                  {existingImages.map((im) => (
                    <div key={im.id} className="group relative h-20 w-20 overflow-hidden rounded-xl border">
                      <img src={mediaUrl(im.path)} className="h-full w-full object-cover" />
                      <button onClick={() => deleteExisting(im.id)} className="absolute right-0.5 top-0.5 grid h-5 w-5 place-items-center rounded bg-black/60 text-white opacity-0 transition-opacity group-hover:opacity-100"><Trash2 className="h-3 w-3" /></button>
                    </div>
                  ))}
                </div>
              ) : <p className="text-xs text-muted">No images yet — capture from the camera above.</p>}
            </div>
          )}
        </div>

        {/* right: form */}
        <div className="space-y-3.5">
          <Field label="Full name *"><input className="input" value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} placeholder="e.g. Grace Hopper" /></Field>
          <Field label="Employee ID / code"><input className="input" value={form.employee_code} onChange={(e) => setForm({ ...form, employee_code: e.target.value })} placeholder="EMP-001" /></Field>
          <Field label="Department"><input className="input" value={form.department} onChange={(e) => setForm({ ...form, department: e.target.value })} placeholder="Engineering" /></Field>
          <Field label="Job title"><input className="input" value={form.job_title} onChange={(e) => setForm({ ...form, job_title: e.target.value })} placeholder="Software Engineer" /></Field>
          <div className="rounded-xl bg-brand-500/8 p-3.5 text-xs text-brand-400">
            {editing
              ? "Capturing from the live camera enrols a face immediately; recognition updates without a restart."
              : "Capture at least one valid face from the live camera, then Save. The employee is created and recognition goes live instantly."}
          </div>
          {phase === "verify" && (
            <div className="rounded-xl surface-2 p-3.5 text-xs text-muted">
              <span className="flex items-center gap-1.5 font-semibold text-[rgb(var(--text))]"><Dot tone="green" pulse /> Live recognition check</span>
              <p className="mt-1">The feed now shows recognition overlays (name + confidence) drawn on the camera.</p>
            </div>
          )}
        </div>
      </div>
    </Dialog>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="label">{label}</label>{children}</div>;
}
