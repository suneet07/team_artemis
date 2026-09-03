import { create } from "zustand";
import type { AssetRef } from "@contracts/types";

export interface LayerState {
  asset: AssetRef;
  visible: boolean;
  opacity: number;
  /** The query that produced it, so the layer row can link to its trace step. */
  queryId: string;
}

export type WorkspaceTab = "chat" | "trace";
export type BaseLayer = "optical" | "sar" | "none";

interface WorkspaceStore {
  /* evidence layers, in production order */
  layers: LayerState[];
  addLayers: (assets: AssetRef[], queryId: string) => void;
  setLayerVisible: (assetId: string, visible: boolean) => void;
  setLayerOpacity: (assetId: string, opacity: number) => void;
  clearLayers: () => void;

  /* base imagery */
  baseLayer: BaseLayer;
  setBaseLayer: (base: BaseLayer) => void;
  baseOpacity: number;
  setBaseOpacity: (opacity: number) => void;

  /* bi-temporal A/B swipe */
  swipeEnabled: boolean;
  swipePosition: number;
  setSwipeEnabled: (enabled: boolean) => void;
  setSwipePosition: (position: number) => void;

  /* right pane */
  tab: WorkspaceTab;
  setTab: (tab: WorkspaceTab) => void;
  selectedQueryId: string | null;
  selectQuery: (queryId: string | null) => void;
  selectedStepIndex: number | null;
  selectStep: (index: number | null) => void;

  /* chrome */
  armature: boolean;
  toggleArmature: () => void;
  showTileGrid: boolean;
  setShowTileGrid: (show: boolean) => void;
}

export const useWorkspace = create<WorkspaceStore>((set) => ({
  layers: [],
  addLayers: (incoming, queryId) =>
    set((state) => {
      const known = new Set(state.layers.map((l) => l.asset.asset_id));
      const fresh = incoming
        .filter((asset) => asset && !known.has(asset.asset_id))
        .map((asset) => ({
          asset,
          visible: true,
          // A mask covering a large extent at high opacity hides the imagery
          // it is meant to be read against.
          opacity: 0.62,
          queryId,
        }));
      return { layers: [...state.layers, ...fresh] };
    }),
  setLayerVisible: (assetId, visible) =>
    set((state) => ({
      layers: state.layers.map((layer) =>
        layer.asset.asset_id === assetId ? { ...layer, visible } : layer,
      ),
    })),
  setLayerOpacity: (assetId, opacity) =>
    set((state) => ({
      layers: state.layers.map((layer) =>
        layer.asset.asset_id === assetId ? { ...layer, opacity } : layer,
      ),
    })),
  clearLayers: () => set({ layers: [] }),

  baseLayer: "optical",
  setBaseLayer: (baseLayer) => set({ baseLayer }),
  baseOpacity: 1,
  setBaseOpacity: (baseOpacity) => set({ baseOpacity }),

  swipeEnabled: false,
  swipePosition: 0.5,
  setSwipeEnabled: (swipeEnabled) => set({ swipeEnabled }),
  setSwipePosition: (swipePosition) => set({ swipePosition }),

  tab: "chat",
  setTab: (tab) => set({ tab }),
  selectedQueryId: null,
  selectQuery: (selectedQueryId) => set({ selectedQueryId }),
  selectedStepIndex: null,
  selectStep: (selectedStepIndex) => set({ selectedStepIndex }),

  armature: false,
  toggleArmature: () => set((state) => ({ armature: !state.armature })),
  showTileGrid: false,
  setShowTileGrid: (showTileGrid) => set({ showTileGrid }),
}));
