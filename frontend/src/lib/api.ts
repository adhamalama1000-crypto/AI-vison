import type {
  AIStatus, Camera, CamerasResponse, DashboardStats, Employee, EmployeesResponse,
  EventsResponse, EventType, FaceVerdict, Health, RegisterResult, TaskStatus,
  AttendanceResponse, AttendanceToday, AttendanceSummary, AttendanceConfig,
  DatasetsResponse, DatasetDetail, DatasetUploadResult,
  TrainingCatalog, TrainingResponse, TrainingDetail, ComparisonResponse,
  StartTrainingBody, StartTrainingResult,
  ReferenceResponse, ReferenceDetail, ReferenceUploadResult, ReferenceSpec,
  PanelsResponse, PanelAnalyzeResult,
  InspectionResponse, InspectionRunResult, Inspection,
  ReportsResponse, ReportsSummary,
  FaceConfig, FaceConfigResponse, RecognitionsResponse, EmbeddingsResponse, RetrainResult,
} from "./types";

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let data: any = null;
  if (text) { try { data = JSON.parse(text); } catch { data = text; } }
  if (!res.ok) {
    const msg =
      data?.error?.message || data?.detail?.message || data?.detail ||
      data?.message || res.statusText || "Request failed";
    const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg)) as Error & { status?: number; data?: any };
    err.status = res.status; err.data = data;
    throw err;
  }
  return data as T;
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const opts: RequestInit = { method, headers: {} };
  if (body !== undefined) {
    (opts.headers as Record<string, string>)["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  return parse<T>(await fetch(path, opts));
}

// multipart/form-data upload — do NOT set Content-Type (browser sets the boundary).
async function upload<T>(path: string, form: FormData): Promise<T> {
  return parse<T>(await fetch(path, { method: "POST", body: form }));
}

export const api = {
  health: () => req<Health>("GET", "/health"),
  dashboard: () => req<DashboardStats>("GET", "/api/stats/dashboard"),

  cameras: () => req<CamerasResponse>("GET", "/cameras"),
  camera: (id: string) => req<Camera>("GET", `/cameras/${id}`),
  createCamera: (b: any) => req<Camera>("POST", "/cameras", b),
  updateCamera: (id: string, b: any) => req<Camera>("PUT", `/cameras/${id}`, b),
  deleteCamera: (id: string) => req<{ deleted: string }>("DELETE", `/cameras/${id}`),
  setActiveCamera: (id: string) => req<{ active_camera: string }>("POST", "/active-camera", { id }),
  diagnose: (id: string) => req<any>("GET", `/cameras/${id}/diagnose`),

  employees: () => req<EmployeesResponse>("GET", "/api/employees"),
  employee: (id: number) => req<Employee>("GET", `/api/employees/${id}`),
  createEmployee: (b: any) => req<Employee>("POST", "/api/employees", b),
  updateEmployee: (id: number, b: any) => req<Employee>("PUT", `/api/employees/${id}`, b),
  deleteEmployee: (id: number) => req<{ deleted: number }>("DELETE", `/api/employees/${id}`),
  registerEmployee: (b: any) => req<RegisterResult>("POST", "/api/employees/register", b),
  validateCapture: (b: { camera_id?: string }) => req<FaceVerdict>("POST", "/api/employees/validate", b),
  addImage: (id: number, b: any) => req<any>("POST", `/api/employees/${id}/images`, b),
  captureImage: (id: number, b: { camera_id?: string; make_profile?: boolean }) =>
    req<any>("POST", `/api/employees/${id}/capture`, b),
  deleteImage: (id: number, imgId: number) => req<any>("DELETE", `/api/employees/${id}/images/${imgId}`),
  employeeEmbeddings: (id: number) => req<EmbeddingsResponse>("GET", `/api/employees/${id}/embeddings`),
  deleteEmbedding: (id: number, embId: number) => req<any>("DELETE", `/api/employees/${id}/embeddings/${embId}`),
  retrainEmployee: (id: number) => req<RetrainResult>("POST", `/api/employees/${id}/retrain`),

  events: (q = "") => req<EventsResponse>("GET", `/api/events${q}`),
  eventTypes: () => req<{ types: EventType[] }>("GET", "/api/events/types"),
  clearEvents: () => req<{ cleared: boolean }>("DELETE", "/api/events"),

  aiStatus: () => req<AIStatus>("GET", "/api/ai/status"),
  aiMetrics: () => req<any>("GET", "/api/ai/metrics"),
  selectModel: (task: string, b: any) => req<TaskStatus>("POST", `/api/ai/models/${task}/select`, b),
  enableModel: (task: string, enabled: boolean) => req<TaskStatus>("POST", `/api/ai/models/${task}/enable`, { enabled }),
  setModelParams: (task: string, params: Record<string, any>) => req<TaskStatus>("POST", `/api/ai/models/${task}/params`, { params }),
  getSettings: () => req<Record<string, any>>("GET", "/api/ai/settings"),
  setSetting: (key: string, value: any) => req<any>("PUT", `/api/ai/settings/${encodeURIComponent(key)}`, { value }),

  // ---- Face recognition config + insight ----
  faceConfig: () => req<FaceConfigResponse>("GET", "/api/ai/face/config"),
  setFaceConfig: (b: Partial<FaceConfig>) => req<{ ok: boolean; config: FaceConfig }>("PUT", "/api/ai/face/config", b),
  faceMessages: () => req<{ messages: Record<string, string> }>("GET", "/api/ai/face/messages"),
  faceRecognitions: (limit = 50) => req<RecognitionsResponse>("GET", `/api/ai/face/recognitions?limit=${limit}`),

  analyze: (camId: string) => req<any>("GET", `/api/cameras/${camId}/analyze`),

  metrics: () => req<any>("GET", "/api/metrics"),

  // ---- Attendance ----
  attendance: (q = "") => req<AttendanceResponse>("GET", `/api/attendance${q}`),
  attendanceToday: () => req<AttendanceToday>("GET", "/api/attendance/today"),
  attendanceSummary: (days = 7) => req<AttendanceSummary>("GET", `/api/attendance/summary?days=${days}`),
  attendanceConfig: () => req<AttendanceConfig>("GET", "/api/attendance/config"),
  setAttendanceConfig: (seconds: number) => req<AttendanceConfig>("PUT", "/api/attendance/config", { seconds }),
  clearAttendance: () => req<{ deleted: number }>("DELETE", "/api/attendance"),

  // ---- Datasets ----
  datasets: () => req<DatasetsResponse>("GET", "/api/datasets"),
  dataset: (id: number) => req<DatasetDetail>("GET", `/api/datasets/${id}`),
  uploadDataset: (form: FormData) => upload<DatasetUploadResult>("/api/datasets/upload", form),
  revalidateDataset: (id: number) => req<{ id: number; report: DatasetDetail["report"] }>("POST", `/api/datasets/${id}/revalidate`),
  deleteDataset: (id: number) => req<{ deleted: number }>("DELETE", `/api/datasets/${id}`),

  // ---- Training ----
  trainingCatalog: () => req<TrainingCatalog>("GET", "/api/training/catalog"),
  training: () => req<TrainingResponse>("GET", "/api/training"),
  trainingJob: (id: number | string) => req<TrainingDetail>("GET", `/api/training/${id}`),
  trainingComparison: (id: number | string) => req<ComparisonResponse>("GET", `/api/training/${id}/comparison`),
  startTraining: (b: StartTrainingBody) => req<StartTrainingResult>("POST", "/api/training", b),
  pauseTraining: (id: number | string) => req<{ ok: boolean }>("POST", `/api/training/${id}/pause`),
  resumeTraining: (id: number | string) => req<{ ok: boolean }>("POST", `/api/training/${id}/resume`),
  stopTraining: (id: number | string) => req<{ ok: boolean }>("POST", `/api/training/${id}/stop`),

  // ---- Reference designs ----
  references: () => req<ReferenceResponse>("GET", "/api/reference"),
  reference: (id: number) => req<ReferenceDetail>("GET", `/api/reference/${id}`),
  uploadReference: (form: FormData) => upload<ReferenceUploadResult>("/api/reference/upload", form),
  setReferenceSpec: (id: number, spec: ReferenceSpec) => req<ReferenceDetail>("PUT", `/api/reference/${id}/spec`, { spec }),
  deleteReference: (id: number) => req<{ deleted: number }>("DELETE", `/api/reference/${id}`),

  // ---- Panel analysis ----
  panels: () => req<PanelsResponse>("GET", "/api/panels"),
  analyzePanel: (form: FormData) => upload<PanelAnalyzeResult>("/api/panels/analyze", form),

  // ---- Inspection ----
  inspections: () => req<InspectionResponse>("GET", "/api/inspection"),
  inspection: (id: number) => req<Inspection>("GET", `/api/inspection/${id}`),
  runInspection: (form: FormData) => upload<InspectionRunResult>("/api/inspection/run", form),

  // ---- Reports ----
  reports: (kind = "") => req<ReportsResponse>("GET", `/api/reports${kind ? `?kind=${encodeURIComponent(kind)}` : ""}`),
  reportsSummary: () => req<ReportsSummary>("GET", "/api/reports/summary"),
  deleteReport: (id: number) => req<{ deleted: number }>("DELETE", `/api/reports/${id}`),
};

// media / stream URL helpers (cache-busted where needed)
export const streamUrl = (camId: string, ai = false, q = 80, fps = 20) =>
  ai
    ? `/api/cameras/${camId}/ai-stream?quality=${q}&fps=${fps}&t=${Date.now()}`
    : `/cameras/${camId}/stream?quality=${q}&fps=${fps}&t=${Date.now()}`;
export const snapshotUrl = (camId: string) => `/cameras/${camId}/snapshot?t=${Date.now()}`;
export const mediaUrl = (path: string) => `/api/media/${path}`;
