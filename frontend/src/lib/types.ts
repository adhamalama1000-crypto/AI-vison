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

// ---- Attendance (Part 1) ----
export interface AttendanceRecord {
  id: number; employee_id: number | null; employee_name: string | null; camera_name: string | null;
  confidence: number | null; snapshot: string | null; day: string; created_at: number;
}
export interface AttendanceResponse { attendance: AttendanceRecord[]; total: number; }
export interface AttendanceToday { day: string; records: AttendanceRecord[]; present: number; employees_total: number; }
export interface AttendanceSummaryRow { day: string; present: number; records: number; }
export interface AttendanceSummary { summary: AttendanceSummaryRow[]; }
export interface AttendanceConfig { timeout_seconds: number; }

// ---- Datasets (Part 2) ----
export interface Dataset {
  id: number; name: string; kind: string; status: string;
  n_images: number; n_labels: number; n_classes: number; created_at: number;
}
export interface DatasetsResponse { datasets: Dataset[]; total: number; }
export interface DatasetReport {
  kind: string; n_images: number; n_labels: number; classes: string[];
  class_counts: Record<string, number>; missing_labels: string[]; corrupt_images: string[];
  warnings: string[]; errors: string[]; imbalance_ratio: number | null; n_classes: number; ok: boolean;
}
export interface DatasetDetail extends Dataset { classes: string[]; report: DatasetReport; }
export interface DatasetUploadResult { id: number; name: string; path: string; report: DatasetReport; }

// ---- Training (Parts 3,4,5) ----
export interface TrainingCatalog {
  classification_models: string[]; detection_models: string[]; tunable: string[];
}
export type TrainingTask = "classification" | "detection";
export interface TrainingJob {
  id: number; name: string; task: string; status: string; progress: number;
  best_model: string | null; created_at: number;
}
export interface TrainingResponse { jobs: TrainingJob[]; total: number; }
export interface TrainingConfig {
  epochs: number; augment: boolean; hpo: boolean; hpo_trials: number;
  learning_rate: number; weight_decay: number; early_stopping_patience: number;
  image_size: number; batch_size: number;
}
export interface TrainingResources {
  cpu_percent: number | null; ram_percent: number | null; ram_used_mb: number | null; gpu_available: boolean;
}
export interface TrainingMetrics {
  model: string | null; epoch: number; epochs: number;
  train_loss: number | null; val_loss: number | null; accuracy: number | null;
  precision: number | null; recall: number | null; f1: number | null;
  learning_rate: number | null; elapsed_s: number | null; eta_s: number | null;
  resources: TrainingResources;
}
export interface TrainingDetail extends TrainingJob {
  metrics: TrainingMetrics | null; history: TrainingMetrics[];
  comparison: ComparisonEntry[]; artifacts: Record<string, string> | null;
}
export interface ComparisonTestMetrics {
  accuracy: number | null; loss: number | null; precision: number | null; recall: number | null; f1: number | null;
}
export interface ComparisonEntry {
  model: string; status: string;
  metrics: { train?: Record<string, number>; val?: Record<string, number>; test: ComparisonTestMetrics };
  selected: boolean; onnx: { verification: { ok: boolean } } | null; reason: string | null;
}
export interface ComparisonResponse { comparison: ComparisonEntry[]; best_model: string | null; }
export interface StartTrainingBody {
  name: string; dataset_id?: number | null; task: TrainingTask; models: string[]; config: TrainingConfig;
}
export interface StartTrainingResult { job_id: number; status: string; }

// ---- Reference designs (Part 9) ----
export interface ReferenceDesign {
  id: number; name: string; kind: string; path: string; description: string | null; created_at: number;
}
export interface ReferenceResponse { references: ReferenceDesign[]; total: number; }
export interface ReferenceSpec {
  component_counts: Record<string, number>; wire_color_counts: Record<string, number>;
}
export interface ReferenceDetail extends ReferenceDesign { spec?: ReferenceSpec | null; }
export interface ReferenceUploadResult { id: number; name: string; kind: string; path: string; }

// ---- Panel analysis (Part 8) ----
export interface PanelComponent { label: string; confidence: number | null; bbox: number[]; position: string | null; }
export interface PanelWire { wire_uid: string; start: string | null; end: string | null; color: string | null; status: string | null; }
export interface PanelResult {
  components: PanelComponent[]; component_counts: Record<string, number>; component_total: number;
  wires: PanelWire[]; wire_color_counts: Record<string, number>; wire_total: number;
  topology: { nodes: number; edges: number }; notes: string[];
}
export interface PanelReport { id: number; title: string; path: string; summary: string | Record<string, any> | null; created_at: number; }
export interface PanelsResponse { panels: PanelReport[]; total: number; }
export interface PanelAnalyzeResult {
  id: number; result: PanelResult; annotated: string | null; pdf: string | null; json: string | null;
}

// ---- Inspection (Part 10) ----
export interface Mismatch { type: string; detail: string | null; expected: unknown; found: unknown; }
export interface Inspection {
  id: number; reference_id: number | null; camera_id: string | null; source: string | null;
  status: string; n_mismatches: number; report_path: string | null; created_at: number;
}
export interface InspectionResponse { inspections: Inspection[]; total: number; }
export interface InspectionRunResult {
  id: number; status: string; n_mismatches: number; mismatches: Mismatch[];
  annotated: string | null; pdf: string | null; json: string | null; result: unknown;
}

// ---- Reports (Parts 8,10) ----
export interface Report { id: number; kind: string; title: string; path: string; summary: string | Record<string, any> | null; created_at: number; }
export interface ReportsResponse { reports: Report[]; total: number; }
export interface ReportsSummary { by_kind: Record<string, number>; }

export type WSMessage =
  | { type: "hello"; timestamp: number; active_camera: string | null; cameras: Camera[] }
  | { type: "stats"; camera_id: string; timestamp: number; state: string; healthy: boolean; fps: number; latency: CameraLatency; frame_age_ms: number | null; statistics: CameraStatistics }
  | { type: "ai_event"; event_id: number; event_type: string; camera_id: string; camera_name: string; label: string; confidence: number | null; employee_id: number | null; snapshot: string | null; timestamp: number }
  | { type: string; [k: string]: any };

// ---- Reference Panels (Industrial Inspection) ----
export interface RefError {
  error_type: string; severity: "error" | "warning" | "info";
  target: string | null; detail: string; confidence: number;
  x?: number; y?: number;
}
export interface RefPanel {
  id: number; name: string; version: string; description: string | null;
  status: string; n_images: number; n_components: number; n_terminals: number;
  n_wires: number; thumbnail: string | null; note: string | null;
  created_at: number; updated_at: number;
}
export interface RefPanelsResponse { panels: RefPanel[]; total: number; }
export interface RefGraph { nodes: any[]; edges: any[]; }
export interface RefPanelDetail extends RefPanel {
  template?: any; features?: any;
  images: any[]; components: any[]; terminals: any[]; wires: any[];
  graph?: RefGraph;
}
export interface RefImageAdd {
  id: number; panel_id: number; path: string; source: string;
  width: number; height: number; is_primary: boolean;
}
export interface CompareResult {
  id: number; panel_id: number; status: string; score: number;
  n_errors: number; n_warnings: number; errors: RefError[];
  alignment: any; snapshot: string | null; observed: any; result: any;
}
export interface InspectionResultRow {
  id: number; panel_id: number; camera_id: string | null; source: string | null;
  status: string; score: number; n_errors: number; n_warnings: number;
  snapshot: string | null; created_at: number;
}
export interface InspectionResultsResponse {
  panel_id: number; results: InspectionResultRow[]; total: number;
}

// ---- Datasheets (OCR / schematic understanding) ----
export interface Datasheet {
  id: number; name: string; kind: string; path: string; panel_id: number | null;
  description: string | null; ocr_engine: string | null; status: string;
  created_at: number; updated_at: number;
}
export interface DatasheetsResponse { datasheets: Datasheet[]; total: number; }
export interface DatasheetExtract {
  id: number; ocr_engine: string; text_chars: number;
  parsed: {
    component_ids: string[]; terminal_ids: string[]; wire_ids: string[];
    connections: { from: string; to: string }[];
    n_components: number; n_terminals: number; n_connections: number;
    component_types?: Record<string, string>;
  };
  expected_graph: RefGraph & { node_count: number; edge_count: number };
  note: string | null;
}
