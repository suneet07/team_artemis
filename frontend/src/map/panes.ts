import type { Bundle, Scene } from "@contracts/types";

/**
 * The two scenes a bundle can show: t1 vs t2 for a bi-temporal pair, optical
 * vs SAR otherwise.
 *
 * Shared because three places have to agree on it — the map, the image viewer
 * and the base-imagery switch. The store's `baseLayer` names a pane, not a
 * modality: "optical" is pane A and "sar" is pane B. When the switch decided
 * by modality on its own, both scenes of a bi-temporal pair were "optical", so
 * both buttons selected pane A and the second date could never be shown.
 */
/** Short names for the two panes, as the comparison controls print them. */
export function paneLabels(bundle: Bundle): readonly [string, string] {
  return bundle.pair_type === "bitemporal" ? ["T1", "T2"] : ["OPT", "SAR"];
}

export function bundlePanes(bundle: Bundle): {
  paneA: Scene | null;
  paneB: Scene | null;
} {
  const scenes = bundle.scenes;
  if (bundle.pair_type === "bitemporal") {
    return {
      paneA: scenes.find((s) => s.role === "t1") ?? scenes[0] ?? null,
      paneB: scenes.find((s) => s.role === "t2") ?? scenes[1] ?? null,
    };
  }
  // The role assigned at receiving decides first, the probed modality second.
  // Going by modality alone left pane B empty whenever the SAR scene carried
  // no sensor tag — an ordinary PNG or JPEG probes as optical — and with no
  // pane B the swipe and side-by-side modes silently fell back to single.
  const sar =
    scenes.find((s) => s.role === "sar") ??
    scenes.find(
      (s) => s.role !== "optical" && s.compatibility?.modality === "sar",
    ) ??
    null;
  const optical =
    scenes.find((s) => s !== sar && s.role === "optical") ??
    scenes.find((s) => s !== sar) ??
    null;
  // Two scenes are always two panes, even when neither can be told apart.
  if (!optical) return { paneA: sar, paneB: null };
  return {
    paneA: optical,
    paneB: sar ?? scenes.find((s) => s !== optical) ?? null,
  };
}
