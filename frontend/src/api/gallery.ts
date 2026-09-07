import { useMutation, useQuery } from "@tanstack/react-query";
import { apiGet, apiSend } from "./client";

/**
 * The testing-corpus gallery.
 *
 * Every item is a row from the *test or held-out* split of a benchmark one of
 * the adapters was scored on, carried with its published gold answer. The point
 * is that a visitor can pick one and watch the system answer a question whose
 * correct answer was fixed by someone else, months before this page existed.
 */

export type GalleryStatus = "verified" | "scored" | "candidate";

export interface GalleryImage {
  role: string;
  preview_url: string;
}

export interface GalleryItem {
  item_id: string;
  segment: string;
  status: GalleryStatus;
  question: string;
  gold: string;
  benchmark_answer?: string | number[] | null;
  question_type?: string | null;
  benchmark?: string | null;
  contract?: string | null;
  corpus: string;
  split: string;
  account: string;
  trained_on: boolean;
  headline?: string | null;
  evidence?: string | null;
  iou?: number;
  rouge_l?: number;
  blind_floor_rouge_l?: number;
  images: GalleryImage[];
  /** Present once `verify_gallery.py` has run against a build. */
  served_answer?: string | null;
  passed?: boolean | null;
  reason?: string | null;
}

export interface GalleryIndex {
  available: boolean;
  counts: Record<string, number>;
  statuses: Record<string, number>;
  accounts: Record<string, string>;
  verification?: {
    build?: string;
    checked?: number;
    passed?: number;
    per_segment?: Record<string, { total: number; passed: number; pass_rate: number }>;
  } | null;
  items: GalleryItem[];
}

export function useGallery() {
  return useQuery({
    queryKey: ["gallery"],
    queryFn: () => apiGet<GalleryIndex>("/gallery"),
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Load one item into a bundle, server-side.
 *
 * The imagery already sits on the volume beside the API, so the browser never
 * downloads and re-uploads it. What comes back is an ordinary bundle id: from
 * here the item goes through the same workspace, router and tools as anything
 * a user drags in.
 */
export function useOpenGalleryItem() {
  return useMutation({
    mutationFn: (itemId: string) =>
      apiSend<{ bundle_id: string; question: string; item: GalleryItem }>(
        "POST",
        `/gallery/${itemId}/bundle`,
      ),
  });
}
