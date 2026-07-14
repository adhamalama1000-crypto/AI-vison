// Types mirror the FastAPI backend responses exactly.

export interface CameraLatency { last_ms: number | null; avg_ms: number | null; max_ms: number | null; }
export interface CameraStatistics {
  frames_captured: number; frames_dropped: number; read_failures_total: number;
  reconnect_count: number; connected_since: number | null; last_frame_at: number | null;
  last_error: string | null; last_error_at: number | null; uptime_seconds: number;
}
export interface Camera {
  id: string; name: string; url: string; transport: string; transport_in_use: string | null;
  state: string; healthy: boolean; has_frame: boolean; frame_seq: number; frame_age_ms: number | null;
  connected_for_seconds: number | null; fps: number; latency: CameraLatency; statistics: CameraStatistics;
}
export interface CamerasResponse { active_camera: string | null; cameras: Camera[]; }

export interface Health {
  status: string; service: string; rtsp_only: boolean; cameras_total: number;
  cameras_connected: number; active_camera: string | null; event_subscribers: number; timestamp: number;
}

export interface EmployeeImage { id: number; path: string; created_at: number; }
export interface Employee {
  id: number; employee_code: string | null; full_name: string; department: string | null;
  job_title: string | null; profile_image: string | null; created_at: number; updated_at: number;
  images: EmployeeImage[]; embeddings: number;
}
export interface EmployeesResponse { employees: Employee[]; total: number; }

export interface FaceVerdict {
  faces: number; ok: boolean; reason: string | null; blur_score: number | null; min_blur: number;
  bbox: number[] | null; multiple_faces: boolean; camera_id?: string; image?: string; face_preview?: string | null;
}

export interface RegisterResult {
  id: number; full_name: string; enrolled: number; rejected: number;
  recognition_enabled: boolean; results: any[];
}

export interface AIEvent {
  id: number; type: string; camera_id: string | null; camera_name: string | null;
  label: string | null; confidence: number | null; employee_id: number | null;
  snapshot: string | null; payload: any; created_at: number;
}
export interface EventsResponse { events: AIEvent[]; total: number; }
export interface EventType { type: string; c: number; }

export interface BackendInfo {
  backend_id: string; task: string; display_name: string; ready: boolean; status: string;
  error: string | null; reason: string | null; loading: boolean; params: Record<string, any>;
  requires_weights: boolean;
}
export interface CatalogItem { backend_id: string; display_name: string; requires_weights: boolean; }
export interface TaskStatus {
  task: string; enabled: boolean; selected_backend: string; state: string; reason: string | null;
  detail: string | null; backend: BackendInfo; available_backends: CatalogItem[];
  metrics: { fps: number; avg_inference_ms: number | null }; last_error: string | null;
}
export interface Resources {
  cpu_percent: number | null; ram_percent: number | null; ram_used_mb: number | null;
  gpu_percent: number | null; gpu_mem_mb: number | null; gpu_available: boolean;
}
export interface AIStatus {
  tasks: Record<string, TaskStatus>;
  resources: Resources;
  catalog: Record<string, CatalogItem[]>;
}

export interface DashboardStats {
  employees: { total: number; enrolled_faces: number };
  recognition: { recognized_events: number; unknown_events: number };
  electrical: { components_detected: number; wiring_errors: number };
  cameras: { total: number; connected: number; active: string | null; total_fps: number; avg_latency_ms: number | null };
  resources: Resources;
  ai_tasks: Record<string, { enabled: boolean; backend: string; ready: boolean }>;
  events_total: number;
}

export type WSMessage =
  | { type: "hello"; timestamp: number; active_camera: string | null; cameras: Camera[] }
  | { type: "stats"; camera_id: string; timestamp: number; state: string; healthy: boolean; fps: number; latency: CameraLatency; frame_age_ms: number | null; statistics: CameraStatistics }
  | { type: "ai_event"; event_id: number; event_type: string; camera_id: string; camera_name: string; label: string; confidence: number | null; employee_id: number | null; snapshot: string | null; timestamp: number }
  | { type: string; [k: string]: any };
