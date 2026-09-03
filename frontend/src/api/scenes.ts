import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Scene, SceneRole } from "@contracts/types";
import { apiGet, apiSend, uploadScene, type UploadHandle } from "./client";

export function useScenes(params?: { is_demo?: boolean }) {
  const search =
    params?.is_demo === undefined ? "" : `?is_demo=${params.is_demo}`;
  return useQuery({
    queryKey: ["scenes", params ?? {}],
    queryFn: () => apiGet<{ scenes: Scene[] }>(`/scenes${search}`),
    select: (data) => data.scenes,
  });
}

export function useScene(sceneId: string | null, options?: { poll?: boolean }) {
  return useQuery({
    queryKey: ["scene", sceneId],
    queryFn: () => apiGet<Scene>(`/scenes/${sceneId}`),
    enabled: Boolean(sceneId),
    refetchInterval: options?.poll ? 1500 : false,
  });
}

export interface ScenePatch {
  role?: SceneRole | null;
  acquired_at?: string | null;
  declared_modality?: "optical" | "sar";
}

export function usePatchScene() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ sceneId, patch }: { sceneId: string; patch: ScenePatch }) =>
      apiSend<Scene>("PATCH", `/scenes/${sceneId}`, patch),
    onSuccess: (scene) => {
      client.setQueryData(["scene", scene.scene_id], scene);
      void client.invalidateQueries({ queryKey: ["scenes"] });
    },
  });
}

export function useDeleteScene() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (sceneId: string) =>
      apiSend<void>("DELETE", `/scenes/${sceneId}`),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["scenes"] }),
  });
}

export interface UploadFields {
  declared_modality?: "optical" | "sar";
  acquired_at?: string;
  role?: SceneRole;
}

/** Thin pass-through so features never import the transport directly. */
export function startSceneUpload(
  file: File,
  fields: UploadFields,
  onProgress: (loaded: number, total: number) => void,
): UploadHandle {
  const stringFields: Record<string, string> = {};
  for (const [key, value] of Object.entries(fields)) {
    if (value !== undefined && value !== null) stringFields[key] = String(value);
  }
  return uploadScene(file, stringFields, onProgress);
}
