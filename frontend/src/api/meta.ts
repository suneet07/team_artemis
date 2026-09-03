import { useQuery } from "@tanstack/react-query";
import type {
  DisagreementCause,
  Health,
  TaskMeta,
  ToolManifest,
} from "@contracts/types";
import { apiGet } from "./client";

/**
 * Metadata drives every label in the app. Nothing here is hardcoded in a
 * component: task names, tool descriptions, permitted parameter ranges and
 * disagreement explanations all come from the backend, so the trace viewer
 * stays correct when the manifest changes.
 *
 * Fetched once and cached for the session (§4.7).
 */

const FOREVER = { staleTime: Infinity, gcTime: Infinity } as const;

export function useTools() {
  return useQuery({
    queryKey: ["meta", "tools"],
    queryFn: () => apiGet<{ tools: ToolManifest[] }>("/meta/tools"),
    select: (data) => {
      const byName = new Map<string, ToolManifest>();
      for (const tool of data.tools) byName.set(tool.name, tool);
      return { tools: data.tools, byName };
    },
    ...FOREVER,
  });
}

export function useTasks() {
  return useQuery({
    queryKey: ["meta", "tasks"],
    queryFn: () => apiGet<{ tasks: TaskMeta[] }>("/meta/tasks"),
    select: (data) => {
      const byTask = new Map<string, TaskMeta>();
      for (const task of data.tasks) byTask.set(task.task, task);
      return { tasks: data.tasks, byTask };
    },
    ...FOREVER,
  });
}

export function useDisagreementCauses() {
  return useQuery({
    queryKey: ["meta", "disagreement-causes"],
    queryFn: () =>
      apiGet<{ causes: DisagreementCause[] }>("/meta/disagreement-causes"),
    select: (data) => {
      const byCode = new Map<string, DisagreementCause>();
      for (const cause of data.causes) byCode.set(cause.code, cause);
      return { causes: data.causes, byCode };
    },
    ...FOREVER,
  });
}

/**
 * Health is polled, not cached forever — at the venue the serving mode is
 * expected to degrade, and that has to show in the chrome within seconds.
 */
export function useHealth() {
  return useQuery({
    queryKey: ["meta", "health"],
    queryFn: () => apiGet<Health>("/meta/health"),
    refetchInterval: 15_000,
    retry: 1,
  });
}

/** A manifest lookup that survives a versioned tool name (`name@adapter`). */
export function lookupTool(
  byName: Map<string, ToolManifest> | undefined,
  invoked: string,
): ToolManifest | undefined {
  if (!byName) return undefined;
  return byName.get(invoked) ?? byName.get(invoked.split("@")[0]);
}
