import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 5000 });
export const useDashboard = () => useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard, refetchInterval: 3000 });
export const useCameras = () => useQuery({ queryKey: ["cameras"], queryFn: api.cameras, refetchInterval: 4000 });
export const useEmployees = () => useQuery({ queryKey: ["employees"], queryFn: api.employees });
export const useAIStatus = () => useQuery({ queryKey: ["ai-status"], queryFn: api.aiStatus, refetchInterval: 3000 });
export const useAIMetrics = () => useQuery({ queryKey: ["ai-metrics"], queryFn: api.aiMetrics, refetchInterval: 2000 });
export const useEvents = (q: string) => useQuery({ queryKey: ["events", q], queryFn: () => api.events(q), refetchInterval: 5000 });
export const useEventTypes = () => useQuery({ queryKey: ["event-types"], queryFn: api.eventTypes, refetchInterval: 8000 });
export const useSettings = () => useQuery({ queryKey: ["settings"], queryFn: api.getSettings });

export function useInvalidate() {
  const qc = useQueryClient();
  return (...keys: string[]) => keys.forEach((k) => qc.invalidateQueries({ queryKey: [k] }));
}

export function useEnableModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ task, enabled }: { task: string; enabled: boolean }) => api.enableModel(task, enabled),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["ai-status"] }); qc.invalidateQueries({ queryKey: ["ai-metrics"] }); },
  });
}
export function useSelectModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ task, backend_id, params }: { task: string; backend_id: string; params?: any }) =>
      api.selectModel(task, { backend_id, params }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ai-status"] }),
  });
}
export function useSetParams() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ task, params }: { task: string; params: Record<string, any> }) => api.setModelParams(task, params),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ai-status"] }),
  });
}
