import { useEffect, useMemo, useRef, useState } from "react";
import {
  ScanFace, ShieldCheck, ShieldAlert, Users, Gauge, Cpu, AlertTriangle,
  CheckCircle2, XCircle, Camera as CamIcon, Sliders, History, CalendarCheck,
} from "lucide-react";
import { useCameras, useEmployees } from "../hooks/useData";
import { CameraStream } from "../components/CameraStream";
import { Card, Badge, EmptyState, Skeleton, Dot } from "../components/ui/primitives";
import { api, mediaUrl } from "../lib/api";
import type { FaceConfig, FaceConfigResponse, DetectedFace, RecognitionRow, AttendanceRecord } from "../lib/types";
import { cx, dateTime, timeAgo } from "../lib/format";
import { useToast } from "../hooks/useToast";

export default function FaceRecognition() {
  const { data: camData } = useCameras();
  const { data: empData } = useEmployees();
  const { push } = useToast();

  const cameras = camData?.cameras ?? [];
  const [camId, setCamId] = useState<string | null>(null);
  useEffect(() => { if (!camId && cameras.length) setCamId(camData?.active_camera || cameras[0].id); }, [cameras, camData, camId]);

  const [cfg, setCfg] = useState<FaceConfigResponse | null>(null);
  const [faces, setFaces] = useState<DetectedFace[]>([]);
  const [recs, setRecs] = useState<RecognitionRow[]>([]);
  const [attendance, setAttendance] = useState<AttendanceRecord[]>([]);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadConfig = () => api.faceConfig().then(setCfg).catch(() => {});
  useEffect(() => { loadConfig(); }, []);

  // poll live detections + history while a camera is selected
  useEffect(() => {
    if (!camId) return;
    const tick = async () => {
      try {
        const a = await api.analyze(camId);
        setFaces((a.faces || []) as DetectedFace[]);
      } catch { /* transient */ }
    };
    tick();
    timer.current = setInterval(tick, 1000);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [camId]);

  useEffect(() => {
    const load = () => {
      api.faceRecognitions(30).then((r) => setRecs(r.recognitions)).catch(() => {});
      api.attendance("?limit=30").then((r) => setAttendance(r.attendance)).catch(() => {});
    };
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  const employees = empData?.employees ?? [];
  const totalEmbeddings = employees.reduce((s, e) => s + (e.embeddings || 0), 0);
  const config = cfg?.config;

  const backendReady = cfg?.backend_state === "loaded" || cfg?.backend_state === "running";
  const isReal = (cfg?.backend || "").includes("insightface");

  return (
    <div className="space-y-5">
      {/* backend status banner */}
      {cfg && (
        <Card className={cx("flex flex-wrap items-center justify-between gap-3 p-4",
          !backendReady && "border-rose-500/40")}>
          <div className="flex items-center gap-3">
            <span className={cx("grid h-10 w-10 place-items-center rounded-xl",
              backendReady ? "bg-emerald-500/15 text-emerald-500" : "bg-rose-500/15 text-rose-500")}>
              {backendReady ? <ShieldCheck className="h-5 w-5" /> : <ShieldAlert className="h-5 w-5" />}
            </span>
            <div>
              <p className="text-sm font-bold">
                {isReal ? "Real recognition: SCRFD + ArcFace (InsightFace)" : "Recognition backend"}
                {" "}<span className="font-mono text-xs text-muted">{cfg.backend}</span>
              </p>
              <p className="text-xs text-muted">
                {backendReady
                  ? `Active · ${config?.embedding_dim ?? "?"}-d embeddings · index: ${config?.index_backend}${config?.faiss_available ? " (FAISS)" : ""}`
                  : (cfg.backend_detail || "Backend not ready — recognition is unavailable.")}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!isReal && backendReady && (
              <Badge tone="amber"><AlertTriangle className="h-3 w-3" /> Fallback model — not production ArcFace</Badge>
            )}
            <Badge tone={backendReady ? "green" : "red"}><Dot tone={backendReady ? "green" : "red"} pulse={backendReady} />{cfg.backend_state}</Badge>
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        {/* live camera + detections */}
        <div className="space-y-4 xl:col-span-2">
          <Card className="overflow-hidden p-0">
            <div className="flex items-center justify-between gap-3 border-b p-4">
              <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-muted">
                <ScanFace className="h-4 w-4 text-brand-400" /> Live recognition
              </div>
              {cameras.length > 0 && (
                <select className="input max-w-[220px]" value={camId ?? ""} onChange={(e) => setCamId(e.target.value)}>
                  {cameras.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              )}
            </div>
            <div className="p-4">
              {cameras.length ? (
                <CameraStream cameraId={camId} ai badge="RECOGNITION" className="w-full" />
              ) : (
                <EmptyState icon={<CamIcon className="h-10 w-10" />} title="No camera configured"
                  hint="Add an RTSP camera in Settings to see live recognition." />
              )}
            </div>
          </Card>

          {/* detected faces this frame */}
          <Card className="p-5">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-muted">
              <Users className="h-4 w-4" /> Detected faces ({faces.length})
            </h3>
            {faces.length === 0 ? (
              <p className="text-sm text-muted">No faces in the current frame.</p>
            ) : (
              <div className="grid gap-2 sm:grid-cols-2">
                {faces.map((f, i) => <FaceCard key={i} face={f} />)}
              </div>
            )}
          </Card>
        </div>

        {/* right column: threshold + stats */}
        <div className="space-y-4">
          {config && <ThresholdPanel cfg={config} onSaved={(c) => { setCfg((p) => p ? { ...p, config: c } : p); push("Recognition settings updated", "success"); }} />}

          <div className="grid grid-cols-2 gap-3">
            <MiniStat icon={<Users className="h-4 w-4" />} label="Employees" value={employees.length} />
            <MiniStat icon={<ScanFace className="h-4 w-4" />} label="Embeddings" value={totalEmbeddings} />
            <MiniStat icon={<Cpu className="h-4 w-4" />} label="Index" value={(config?.index_backend || "—").toUpperCase()} />
            <MiniStat icon={<Gauge className="h-4 w-4" />} label="Dim" value={config?.embedding_dim ?? "—"} />
          </div>

          {/* employee gallery */}
          <Card className="p-5">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-muted">
              <Users className="h-4 w-4" /> Employee gallery
            </h3>
            {employees.length === 0 ? (
              <p className="text-sm text-muted">No employees enrolled yet.</p>
            ) : (
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                {employees.map((e) => (
                  <div key={e.id} className="flex items-center gap-3 rounded-xl surface-2 p-2">
                    <Avatar emp={e} />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold">{e.full_name}</p>
                      <p className="truncate text-xs text-muted">{e.department || e.job_title || "—"}</p>
                    </div>
                    <Badge tone={e.embeddings >= 10 ? "green" : e.embeddings > 0 ? "amber" : "gray"}>
                      {e.embeddings} sample{e.embeddings === 1 ? "" : "s"}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
            <p className="mt-3 text-[11px] text-muted">Tip: enrol 15–20 samples per employee across angles, lighting &amp; expressions for best accuracy.</p>
          </Card>
        </div>
      </div>

      {/* history + attendance */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card className="p-5">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-muted">
            <History className="h-4 w-4" /> Recognition history
          </h3>
          {recs.length === 0 ? <p className="text-sm text-muted">No recognitions logged yet.</p> : (
            <div className="max-h-80 space-y-1.5 overflow-y-auto pr-1">
              {recs.map((r) => {
                const known = r.type === "face_recognized";
                return (
                  <div key={r.id} className="flex items-center gap-3 rounded-lg surface-2 px-3 py-2 text-sm">
                    {known ? <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" /> : <XCircle className="h-4 w-4 shrink-0 text-rose-500" />}
                    <span className={cx("min-w-0 flex-1 truncate font-medium", !known && "text-muted")}>{r.label || (known ? "Employee" : "Unknown Employee")}</span>
                    {r.confidence != null && <span className="tabular-nums text-xs text-muted">{(r.confidence * 100).toFixed(0)}%</span>}
                    <span className="shrink-0 text-xs text-faint">{timeAgo(r.created_at)}</span>
                  </div>
                );
              })}
            </div>
          )}
        </Card>

        <Card className="p-5">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-muted">
            <CalendarCheck className="h-4 w-4" /> Attendance log
          </h3>
          {attendance.length === 0 ? <p className="text-sm text-muted">No attendance recorded yet.</p> : (
            <div className="max-h-80 space-y-1.5 overflow-y-auto pr-1">
              {attendance.map((a) => (
                <div key={a.id} className="flex items-center gap-3 rounded-lg surface-2 px-3 py-2 text-sm">
                  <CalendarCheck className="h-4 w-4 shrink-0 text-brand-400" />
                  <span className="min-w-0 flex-1 truncate font-medium">{a.employee_name || `#${a.employee_id}`}</span>
                  {a.confidence != null && <span className="tabular-nums text-xs text-muted">{(a.confidence * 100).toFixed(0)}%</span>}
                  <span className="shrink-0 text-xs text-faint">{dateTime(a.created_at)}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function FaceCard({ face }: { face: DetectedFace }) {
  const known = face.employee_id != null;
  const x = face.extra || {};
  const sim = x.similarity_pct ?? (x.similarity != null ? x.similarity * 100 : null);
  return (
    <div className={cx("rounded-xl border-l-4 p-3", known ? "border-emerald-500 bg-emerald-500/5" : "border-rose-500 bg-rose-500/5")}>
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 font-semibold">
          {known ? <CheckCircle2 className="h-4 w-4 text-emerald-500" /> : <ShieldAlert className="h-4 w-4 text-rose-500" />}
          {face.label}
        </span>
        <Badge tone={known ? "green" : "red"}>{known ? "Employee" : "Unknown"}</Badge>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-muted">
        <span>Similarity: <b className="text-[rgb(var(--text))]">{sim != null ? `${sim.toFixed(1)}%` : "—"}</b></span>
        <span>Confidence: <b className="text-[rgb(var(--text))]">{x.confidence != null ? `${(x.confidence * 100).toFixed(0)}%` : "—"}</b></span>
        {x.quality != null && <span>Quality: {(x.quality * 100).toFixed(0)}%</span>}
        {x.det_score != null && <span>Detect: {(x.det_score * 100).toFixed(0)}%</span>}
      </div>
      {!known && x.message && (
        <p className="mt-2 flex items-center gap-1.5 text-xs font-medium text-rose-500">
          <AlertTriangle className="h-3.5 w-3.5" /> {x.message}
        </p>
      )}
    </div>
  );
}

function ThresholdPanel({ cfg, onSaved }: { cfg: FaceConfig; onSaved: (c: FaceConfig) => void }) {
  const [threshold, setThreshold] = useState(cfg.threshold);
  const [margin, setMargin] = useState(cfg.margin);
  const [policy, setPolicy] = useState<FaceConfig["match_policy"]>(cfg.match_policy);
  const [saving, setSaving] = useState(false);
  const { push } = useToast();
  useEffect(() => { setThreshold(cfg.threshold); setMargin(cfg.margin); setPolicy(cfg.match_policy); }, [cfg.threshold, cfg.margin, cfg.match_policy]);

  const dirty = threshold !== cfg.threshold || margin !== cfg.margin || policy !== cfg.match_policy;
  const save = async () => {
    setSaving(true);
    try { const r = await api.setFaceConfig({ threshold, margin, match_policy: policy }); onSaved(r.config); }
    catch (e: any) { push(e.message || "Failed to update", "error"); }
    finally { setSaving(false); }
  };

  return (
    <Card className="p-5">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-muted">
        <Sliders className="h-4 w-4" /> Recognition threshold
      </h3>
      <div className="space-y-4">
        <div>
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="font-medium">Match threshold</span>
            <span className="font-mono font-bold text-brand-400">{threshold.toFixed(2)}</span>
          </div>
          <input type="range" min={0.3} max={0.9} step={0.01} value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))} className="w-full accent-brand-500" />
          <p className="mt-1 text-[11px] text-muted">Higher = stricter (fewer false accepts). Below this cosine similarity a face is “Unknown Employee”.</p>
        </div>
        <div>
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="font-medium">Identity margin</span>
            <span className="font-mono font-bold text-brand-400">{margin.toFixed(2)}</span>
          </div>
          <input type="range" min={0} max={0.3} step={0.01} value={margin}
            onChange={(e) => setMargin(parseFloat(e.target.value))} className="w-full accent-brand-500" />
          <p className="mt-1 text-[11px] text-muted">The best match must beat the runner-up employee by this much — prevents confusing similar people.</p>
        </div>
        <div>
          <span className="mb-1 block text-sm font-medium">Match policy</span>
          <div className="flex gap-2">
            {(["average", "nearest"] as const).map((p) => (
              <button key={p} onClick={() => setPolicy(p)}
                className={cx("flex-1 rounded-lg border px-3 py-1.5 text-sm font-medium capitalize transition-colors",
                  policy === p ? "border-brand-500 bg-brand-500/10 text-brand-400" : "surface-2 text-muted")}>
                {p}
              </button>
            ))}
          </div>
        </div>
        <button className="btn-primary w-full" disabled={!dirty || saving} onClick={save}>
          {saving ? "Saving…" : dirty ? "Apply changes" : "Saved"}
        </button>
      </div>
    </Card>
  );
}

function MiniStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <Card className="p-4">
      <div className="flex items-center gap-2 text-muted"><span className="text-brand-400">{icon}</span><span className="text-xs font-semibold uppercase tracking-wide">{label}</span></div>
      <p className="mt-1.5 truncate text-lg font-bold tabular-nums">{value}</p>
    </Card>
  );
}

function Avatar({ emp }: { emp: { profile_image: string | null; images: { path: string }[]; full_name: string } }) {
  const src = emp.profile_image ? mediaUrl(emp.profile_image) : emp.images[0] ? mediaUrl(emp.images[0].path) : null;
  const initials = emp.full_name.split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase();
  return (
    <div className={cx("grid h-9 w-9 shrink-0 place-items-center overflow-hidden rounded-full text-xs font-bold", src ? "" : "bg-brand-500/15 text-brand-400")}>
      {src ? <img src={src} className="h-full w-full object-cover" /> : initials}
    </div>
  );
}
