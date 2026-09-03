import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { CreateQueryRequest } from "@contracts/types";
import { useCreateQuery } from "@/api/queries";
import { QUERY_EVENTS, useEventStream } from "@/api/stream";
import { useRuns } from "@/store/runs";
import { useWorkspace } from "@/store/workspace";
import { ApiFailure } from "@/api/client";

/**
 * Drives one query from submission to `done`.
 *
 * The SSE sequence is the trace being assembled, so every event is folded
 * into the run store as it lands and the panels re-render from it. Evidence
 * assets are pushed onto the map the moment the `evidence` event arrives,
 * rather than after the answer — which is what makes the execution legible
 * instead of a spinner followed by a result.
 */
export function useQueryStream(bundleId: string) {
  const [activeQueryId, setActiveQueryId] = useState<string | null>(null);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<ApiFailure | null>(null);

  const createQuery = useCreateQuery();
  const start = useRuns((s) => s.start);
  const apply = useRuns((s) => s.apply);
  const addLayers = useWorkspace((s) => s.addLayers);
  const selectQuery = useWorkspace((s) => s.selectQuery);
  const queryClient = useQueryClient();

  const activeRef = useRef<string | null>(null);
  activeRef.current = activeQueryId;

  const ask = useCallback(
    async (question: string, options?: CreateQueryRequest["options"]) => {
      const trimmed = question.trim();
      if (!trimmed) return;
      setSubmitError(null);
      try {
        const response = await createQuery.mutateAsync({
          bundle_id: bundleId,
          question: trimmed,
          options: { allow_llm_router: true, stream: true, ...options },
        });
        start(response.query_id, bundleId, trimmed);
        selectQuery(response.query_id);
        setActiveQueryId(response.query_id);
        setStreamUrl(response.stream_url);
      } catch (error) {
        if (error instanceof ApiFailure) setSubmitError(error);
        else
          setSubmitError(
            new ApiFailure(0, {
              code: "INTERNAL",
              message: "The query could not be submitted.",
            }),
          );
      }
    },
    [bundleId, createQuery, start, selectQuery],
  );

  const transport = useEventStream(streamUrl, {
    events: QUERY_EVENTS,
    enabled: Boolean(streamUrl),
    onEvent: (name, data) => {
      const queryId = activeRef.current;
      if (!queryId) return;
      apply(queryId, name, data);

      if (name === "evidence") {
        const assets = (data as { assets?: unknown[] })?.assets ?? [];
        addLayers(assets as never[], queryId);
      }
      if (name === "done" || name === "error") {
        setStreamUrl(null);
        void queryClient.invalidateQueries({ queryKey: ["queries", bundleId] });
      }
    },
  });

  // A query that ends without a `done` still has to settle the UI.
  useEffect(() => {
    if (transport === "polling" && activeQueryId) {
      setStreamUrl(null);
    }
  }, [transport, activeQueryId]);

  return {
    ask,
    activeQueryId,
    streaming: Boolean(streamUrl),
    transport,
    submitError,
    clearSubmitError: () => setSubmitError(null),
  };
}
