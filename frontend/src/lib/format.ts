export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

export function fmt(n: number | null | undefined, digits = 0): string {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function timeAgo(ts: number | null | undefined): string {
  if (!ts) return "—";
  const d = Date.now() / 1000 - ts;
  if (d < 5) return "just now";
  if (d < 60) return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

export function clockTime(ts: number | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function dateTime(ts: number | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}

// Report/panel summaries are stored as JSON and arrive parsed as objects.
// Render them as a short human string instead of crashing on an object child.
export function summaryText(s: unknown): string {
  if (s == null) return "—";
  if (typeof s === "string") return s;
  if (typeof s === "object") {
    const o = s as Record<string, any>;
    const parts: string[] = [];
    if (o.panel_type_name) parts.push(String(o.panel_type_name));
    if (o.component_total != null) parts.push(`${o.component_total} components`);
    if (o.component_types != null) parts.push(`${o.component_types} types`);
    if (o.status != null) parts.push(String(o.status));
    if (o.n_mismatches != null) parts.push(`${o.n_mismatches} mismatch${o.n_mismatches === 1 ? "" : "es"}`);
    if (parts.length) return parts.join(" · ");
    const keys = Object.keys(o).filter((k) => o[k] != null && typeof o[k] !== "object");
    return keys.length ? keys.slice(0, 3).map((k) => `${k}: ${o[k]}`).join(" · ") : "—";
  }
  return String(s);
}

export const STATE_TONE: Record<string, "green" | "red" | "amber" | "blue" | "gray"> = {
  connected: "green", running: "green", loaded: "blue", ready: "green",
  connecting: "amber", reconnecting: "amber", loading: "amber", disabled: "gray",
  not_loaded: "gray", stopped: "gray", error: "red", initializing: "amber",
};

export const EVENT_LABEL: Record<string, string> = {
  face_recognized: "Face recognized",
  unknown_person: "Unknown person",
  wiring_error: "Wiring error",
  component_detected: "Component detected",
  system_alert: "System alert",
};

export const REASON_TEXT: Record<string, string> = {
  no_face_detected: "No face detected — position your face in view and try again.",
  blurry: "Face too blurry — hold still or improve lighting.",
  motion_blur: "Motion blur — hold still and capture again.",
  face_crop_empty: "No face detected — try again.",
  face_too_small: "Face too small — move closer to the camera.",
  multiple_faces: "Multiple faces — only one person may be captured at a time.",
  multiple_faces_warning: "Multiple faces detected — the largest will be used.",
  overexposed: "Bad lighting — too bright. Reduce glare / backlight.",
  underexposed: "Bad lighting — too dark. Add more light.",
  bad_lighting: "Bad lighting — adjust the lighting and try again.",
  low_detection_confidence: "Face not clear enough — face the camera directly.",
  embedding_failed: "Could not read face features — try another capture.",
  enrollment_error: "That frame could not be processed — try another capture.",
  no_valid_face: "No usable face in the captured frames — capture again.",
  face_service_unavailable: "Face module is unavailable. Enable it on the AI Models page.",
};

export const TASK_META: Record<string, { label: string; desc: string }> = {
  face: { label: "Face Recognition", desc: "Employee identification & attendance" },
  detection: { label: "Object Detection", desc: "General object detection (YOLO)" },
  components: { label: "Component Detection", desc: "Electrical component recognition" },
  wires: { label: "Wire Topology", desc: "Wiring analysis & fault detection" },
};
