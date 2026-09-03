import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  Bundle,
  CreateBundleRequest,
  PairType,
  SceneRole,
} from "@contracts/types";
import { apiGet, apiSend } from "./client";

export interface CreateBundleResponse {
  bundle_id: string;
  job_id: string;
  stream_url: string;
  status: "preparing";
}

export function useBundles() {
  return useQuery({
    queryKey: ["bundles"],
    queryFn: () => apiGet<{ bundles: Bundle[] }>("/bundles"),
    select: (data) => data.bundles,
  });
}

export function useDemoBundles() {
  return useQuery({
    queryKey: ["demo-bundles"],
    queryFn: () => apiGet<{ bundles: Bundle[] }>("/demo/bundles"),
    select: (data) => data.bundles,
    staleTime: Infinity,
  });
}

export function useBundle(bundleId: string | null, poll = false) {
  return useQuery({
    queryKey: ["bundle", bundleId],
    queryFn: () => apiGet<Bundle>(`/bundles/${bundleId}`),
    enabled: Boolean(bundleId),
    refetchInterval: poll ? 2000 : false,
  });
}

export function useCreateBundle() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateBundleRequest) =>
      apiSend<CreateBundleResponse>("POST", "/bundles", body),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["bundles"] }),
  });
}

export function useRepreparebundle() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (bundleId: string) =>
      apiSend<CreateBundleResponse>(
        "POST",
        `/bundles/${bundleId}/reprepare`,
        {},
      ),
    onSuccess: (_data, bundleId) => {
      void client.invalidateQueries({ queryKey: ["bundle", bundleId] });
    },
  });
}

/**
 * Client-side pair validation.
 *
 * §4.2 asks us to check this before posting so a mismatched pair never costs
 * a round trip and a 409 — the Prepare button is blocked instead, with the
 * reason stated where the user is looking.
 */
export interface PairValidation {
  valid: boolean;
  reason: string | null;
  requiredRoles: SceneRole[];
}

export function validatePair(
  pairType: PairType,
  assignments: { role: SceneRole | null; acquired_at: string | null }[],
): PairValidation {
  const required: Record<PairType, SceneRole[]> = {
    single: ["optical"],
    crossmodal: ["optical", "sar"],
    bitemporal: ["t1", "t2"],
  };
  const requiredRoles = required[pairType];
  const roles = assignments.map((a) => a.role);

  if (pairType === "single") {
    if (assignments.length !== 1) {
      return {
        valid: false,
        reason: "A single-scene bundle takes exactly one scene.",
        requiredRoles,
      };
    }
    return { valid: true, reason: null, requiredRoles };
  }

  if (assignments.length !== 2) {
    return {
      valid: false,
      reason: `A ${pairType} bundle needs exactly two scenes.`,
      requiredRoles,
    };
  }

  for (const role of requiredRoles) {
    if (!roles.includes(role)) {
      return {
        valid: false,
        reason: `No scene is assigned the ${role.toUpperCase()} role.`,
        requiredRoles,
      };
    }
  }

  if (pairType === "bitemporal") {
    const missingDate = assignments.find((a) => !a.acquired_at);
    if (missingDate) {
      return {
        valid: false,
        reason:
          "Both scenes in a bi-temporal pair need an acquisition date, so the pipeline knows which is earlier.",
        requiredRoles,
      };
    }
  }

  return { valid: true, reason: null, requiredRoles };
}
