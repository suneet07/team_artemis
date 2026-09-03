import { useMutation, useQuery } from "@tanstack/react-query";
import type { JobStatus } from "@contracts/types";
import { apiGet, apiSend } from "./client";

/**
 * The polling twin of every SSE stream. Enabled only once `useEventStream`
 * has reported it fell back, so the happy path opens exactly one connection.
 */
export function useJobPolling(jobId: string | null, enabled: boolean) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => apiGet<JobStatus>(`/jobs/${jobId}`),
    enabled: Boolean(jobId) && enabled,
    refetchInterval: 2000,
  });
}

export function useCancelJob() {
  return useMutation({
    mutationFn: (jobId: string) =>
      apiSend<JobStatus>("POST", `/jobs/${jobId}/cancel`, {}),
  });
}
