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
  message?: string | null; quality?: number | null; brightness?: number | null; det_score?: number | null;
}

// ---- Face recognition (real SCRFD + ArcFace pipeline) ----
export interface FaceConfig {
  threshold: number; margin: number; match_policy: "average" | "nearest";
  min_det_score: number; min_blur: number; min_recog_blur: number;
  min_face_size: number; enroll_min_face_size: number;
  index_backend: string; faiss_available: boolean; embedder: string;
  embedding_dim: number | null; enrolled_vectors: number; enrolled_employees: number;
}
export interface FaceConfigResponse {
  config: FaceConfig; backend: string; backend_state: string;
  backend_detail: string | null; backend_info: BackendInfo; params: Record<string, any>;
}
export interface RecognitionRow {
  id: number; type: string; camera_id: string | null; camera_name: string | null;
  label: string | null; confidence: number | null; employee_id: number | null;
  snapshot: string | null; created_at: number;
}
export interface RecognitionsResponse { recognitions: RecognitionRow[]; total: number; }
export interface EmbeddingRow {
  id: number; image_id: number | null; embedder: string; dim: number;
  quality: number | null; meta: Record<string, any> | null; created_at: number;
}
export interface EmbeddingsResponse { employee_id: number; embeddings: EmbeddingRow[]; total: number; }
export interface RetrainResult {
  employee_id: number; images: number; enrolled: number; embedder: string; results: any[];
}
export interface DetectedFace {
  label: string; confidence: number; bbox: number[]; kind: string;
  identity: string | null; employee_id: number | null;
  extra: {
    similarity?: number; similarity_pct?: number; confidence?: number; margin?: number;
    status?: string; quality?: number; blur?: number; min_side?: number;
    brightness?: number; det_score?: number; reason?: string | null; message?: string | null;
    closest_name?: string | null;
  };
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

// ---- Panel inspection (Madkour AI Panel Inspector) ----

/** One fully-annotated component, as produced by electrical/expert.py. */
export interface PanelComponent {
  index: number;
  class_id: string;
  label: string;
  title: string;
  confidence: number;
  confidence_pct: number;
  bbox: number[];
  center: number[];
  position: string;
  row: number | null;
  row_position: number | null;
  category: string;
  domain: string;
  function: string;
  purpose: string;
  mounting: string[];
  manufacturer: string | null;
  product_family: string | null;
  part_number: string | null;
  nameplate_text: string | null;
  identification_basis: string;
  notes: string[];
  extra: Record<string, any>;
}

export interface BomEntry {
  class_id: string; name: string; category: string; quantity: number;
  mean_confidence: number; min_confidence: number; max_confidence: number;
  manufacturers: string[]; function: string; indices: number[];
}

export interface PanelTypeCandidate {
  id: string; name: string; confidence: number; score: number;
  evidence: string[]; function: string; description: string;
}

export interface PanelClassification {
  panel_type: string; panel_type_name: string; confidence: number;
  function: string; evidence: string[];
  candidates: PanelTypeCandidate[]; reason: string | null;
}

export interface ApplicationGuess {
  application: string | null; confidence: number; evidence: string[];
}

export interface MissingComponent {
  class_id: string; name: string; severity: string; rationale: string;
}

export interface MaintenanceNote { code: string; severity: string; message: string; }

export interface ConfidenceStats {
  count: number; mean: number | null; median: number | null;
  min: number | null; max: number | null; below_0_5: number; unknown: number;
}

export interface GateDiagnostics {
  input_count: number; output_count: number; dropped_total: number;
  dropped_by_reason: Record<string, number>;
  relabelled_unknown: number; suppression_rate: number;
}

export interface PanelResult {
  engine: string;
  engine_version: string;
  image_size: number[];
  components: PanelComponent[];
  component_total: number;
  component_counts: Record<string, number>;
  component_counts_by_id?: Record<string, number>;
  bill_of_materials: BomEntry[];
  panel: PanelClassification;
  application: ApplicationGuess;
  missing_components: MissingComponent[];
  maintenance_notes: MaintenanceNote[];
  confidence: ConfidenceStats;
  layout: { rows: number; description: string[] };
  diagnostics: GateDiagnostics;
  notes: string[];
  ocr: { engine: string | null; item_count: number; note?: string | null };
  /** Wiring detection is disabled by design; this states that explicitly. */
  wire_analysis: { enabled: boolean; reason: string };
  component_model_loaded?: boolean;
  duration_ms?: number;
  inspected_at?: number;
  report?: Record<string, any>;
  // legacy keys retained for older clients
  wires: unknown[];
  wire_total: number;
  wire_color_counts: Record<string, number>;
  topology?: { nodes: unknown; edges: unknown };
}

export interface PanelReport { id: number; title: string; path: string; summary: string | Record<string, any> | null; created_at: number; }
export interface PanelsResponse { panels: PanelReport[]; total: number; }
export interface PanelAnalyzeResult {
  id: number; result: PanelResult;
  annotated: string | null; pdf: string | null; json: string | null;
  panel_type?: string | null; panel_type_name?: string | null;
  panel_type_confidence?: number | null; panel_function?: string | null;
  application?: string | null; component_types?: number;
  unknown_components?: number; mean_confidence?: number | null;
  duration_ms?: number | null;
  component_total: number; wire_total: number;
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

// ---- AI Image Analysis & Comparison ----
export interface ImgObject { label: string; confidence: number; x1: number; y1: number; x2: number; y2: number; source: string | null; }
export interface OcrItem { text: string; confidence: number | null; x1: number | null; y1: number | null; x2: number | null; y2: number | null; engine: string | null; }
export interface DominantColor { hex: string; rgb: number[]; name: string; ratio: number; }
export interface ImageRow {
  id: number; name: string | null; path: string; format: string | null;
  width: number | null; height: number | null; n_objects: number;
  summary: string | null; status: string; created_at: number;
}
export interface ImagesResponse { images: ImageRow[]; total: number; }
export interface ImageDetail extends ImageRow {
  bytes: number | null; sha256: string | null; phash: string | null;
  dominant_colors: DominantColor[]; tags: string[]; metadata: any;
  analysis: any; ocr_text: string | null;
  objects: ImgObject[]; ocr_items: OcrItem[];
}
export interface ImageUploadResult { id: number; name: string; path: string; format: string; width: number; height: number; bytes: number; status: string; }

export interface ImgDiff { diff_type: string; severity: string; detail: string | null; confidence: number | null; x1: number | null; y1: number | null; x2: number | null; y2: number | null; }
export interface ComparisonResult {
  id: number; ref_image_id: number | null; cur_image_id: number | null;
  similarity: number; n_diffs: number; status: string;
  overlay_path: string | null; heatmap_path: string | null; aligned_path: string | null;
  report_pdf: string | null; report: any; diffs: ImgDiff[]; created_at: number;
}
export interface ComparisonsResponse { comparisons: ComparisonResult[]; total: number; }
export interface ImageHistory { images: ImageRow[]; comparisons: ComparisonResult[]; }
