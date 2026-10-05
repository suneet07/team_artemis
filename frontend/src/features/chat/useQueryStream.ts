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

  const [submitting, setSubmitting] = useState(false);

  const createQuery = useCreateQuery();
  const begin = useRuns((s) => s.begin);
  const confirm = useRuns((s) => s.confirm);
  const discard = useRuns((s) => s.discard);
  const apply = useRuns((s) => s.apply);
  const addLayers = useWorkspace((s) => s.addLayers);
  const selectQuery = useWorkspace((s) => s.selectQuery);
  const queryClient = useQueryClient();

  const activeRef = useRef<string | null>(null);
  activeRef.current = activeQueryId;

  // One question at a time. The stream is addressed by a single active id, so
  // a second submission while one is in flight would orphan the first turn —
  // and the suggestion chips call `ask` directly, past the composer's own
  // disabled state.
  const busyRef = useRef(false);
  busyRef.current = submitting || Boolean(streamUrl);

  const ask = useCallback(
    async (question: string, options?: CreateQueryRequest["options"]) => {
      const trimmed = question.trim();
      if (!trimmed || busyRef.current) return;
      busyRef.current = true;
      setSubmitError(null);

      /* The turn opens now, not when the POST returns. The backend computes
         the whole answer inside that request — seconds when it is warm, a
         minute or more when the GPU container and the model have to start —
         and until then the chat used to show nothing at all, which reads as a
         click that did not register. */
      const provisionalId = `pending_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
      begin(provisionalId, bundleId, trimmed);
      setSubmitting(true);
      try {
        const response = await createQuery.mutateAsync({
          bundle_id: bundleId,
          question: trimmed,
          options: { allow_llm_router: true, stream: true, ...options },
        });
        confirm(provisionalId, response.query_id);
        selectQuery(response.query_id);
        setActiveQueryId(response.query_id);
        setStreamUrl(response.stream_url);
      } catch (error) {
        discard(provisionalId);
        if (error instanceof ApiFailure) setSubmitError(error);
        else
          setSubmitError(
            new ApiFailure(0, {
              code: "INTERNAL",
              message: "The query could not be submitted.",
            }),
          );
      } finally {
        setSubmitting(false);
      }
    },
    [bundleId, createQuery, begin, confirm, discard, selectQuery],
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
    /** Submitted, not yet acknowledged by the API. */
    submitting,
    streaming: Boolean(streamUrl),
    transport,
    submitError,
    clearSubmitError: () => setSubmitError(null),
  };
}
