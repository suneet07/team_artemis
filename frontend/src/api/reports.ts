import { useMutation, useQuery } from "@tanstack/react-query";
import type { CreateReportRequest, Report } from "@contracts/types";
import { apiGet, apiSend, resolveUrl } from "./client";

export function useCreateReport() {
  return useMutation({
    mutationFn: ({
      queryId,
      body,
    }: {
      queryId: string;
      body: CreateReportRequest;
    }) => apiSend<Report>("POST", `/queries/${queryId}/report`, body),
  });
}

export function useReport(reportId: string | null, poll: boolean) {
  return useQuery({
    queryKey: ["report", reportId],
    queryFn: () => apiGet<Report>(`/reports/${reportId}`),
    enabled: Boolean(reportId),
    refetchInterval: poll ? 1200 : false,
  });
}

export function reportFileUrl(reportId: string): string {
  return resolveUrl(`/reports/${reportId}/file`);
}
