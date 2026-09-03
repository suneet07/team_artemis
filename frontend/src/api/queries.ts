import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  AssetRef,
  CreateQueryRequest,
  QueryResult,
  Trace,
} from "@contracts/types";
import { apiGet, apiSend, resolveUrl } from "./client";

export interface CreateQueryResponse {
  query_id: string;
  stream_url: string;
}

export function useCreateQuery() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateQueryRequest) =>
      apiSend<CreateQueryResponse>("POST", "/queries", body),
    onSuccess: (_data, variables) => {
      void client.invalidateQueries({
        queryKey: ["queries", variables.bundle_id],
      });
    },
  });
}

export function useQueryResult(queryId: string | null) {
  return useQuery({
    queryKey: ["query", queryId],
    queryFn: () => apiGet<QueryResult>(`/queries/${queryId}`),
    enabled: Boolean(queryId),
  });
}

export function useQueryHistory(bundleId: string | null) {
  return useQuery({
    queryKey: ["queries", bundleId],
    queryFn: () =>
      apiGet<{ queries: QueryResult[] }>(`/queries?bundle_id=${bundleId}`),
    enabled: Boolean(bundleId),
    select: (data) => data.queries,
  });
}

export function useQueryEvidence(queryId: string | null) {
  return useQuery({
    queryKey: ["query-evidence", queryId],
    queryFn: () =>
      apiGet<{ assets: AssetRef[] }>(`/queries/${queryId}/evidence`),
    enabled: Boolean(queryId),
    select: (data) => data.assets,
  });
}

export function useTrace(queryId: string | null) {
  return useQuery({
    queryKey: ["trace", queryId],
    queryFn: () => apiGet<Trace>(`/queries/${queryId}/trace`),
    enabled: Boolean(queryId),
  });
}

export function useCancelQuery() {
  return useMutation({
    mutationFn: (queryId: string) =>
      apiSend<{ query_id: string; state: string }>(
        "POST",
        `/queries/${queryId}/cancel`,
        {},
      ),
  });
}

/** The graded artifact. Its download button is not optional (§4.4). */
export function traceDownloadUrl(queryId: string): string {
  return resolveUrl(`/queries/${queryId}/trace?download=1`);
}
