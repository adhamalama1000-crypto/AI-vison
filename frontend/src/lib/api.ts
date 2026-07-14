import type {
  AIStatus, Camera, CamerasResponse, DashboardStats, Employee, EmployeesResponse,
  EventsResponse, EventType, FaceVerdict, Health, RegisterResult, TaskStatus,
} from "./types";

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const opts: RequestInit = { method, headers: {} };
  if (body !== undefined) {
    (opts.headers as Record<string, string>)["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
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

  analyze: (camId: string) => req<any>("GET", `/api/cameras/${camId}/analyze`),
};

// media / stream URL helpers (cache-busted where needed)
export const streamUrl = (camId: string, ai = false, q = 80, fps = 20) =>
  ai
    ? `/api/cameras/${camId}/ai-stream?quality=${q}&fps=${fps}&t=${Date.now()}`
    : `/cameras/${camId}/stream?quality=${q}&fps=${fps}&t=${Date.now()}`;
export const snapshotUrl = (camId: string) => `/cameras/${camId}/snapshot?t=${Date.now()}`;
export const mediaUrl = (path: string) => `/api/media/${path}`;
