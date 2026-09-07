import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import Map, {
  Layer,
  ScaleControl,
  Source,
  type MapRef,
  type ViewState,
} from "react-map-gl/maplibre";
import type { StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Bundle, Scene } from "@contracts/types";
import { ENABLE_BASEMAP, resolveUrl } from "@/api/client";
import { useWorkspace, type LayerState } from "@/store/workspace";
import { formatLatLon } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * The imagery window.
 *
 * No external tile provider on the demo path — the venue is offline. The
 * ground is a plain dark canvas and every pixel over it comes from the API:
 * the scene's own raster tiles, then one raster layer per evidence asset in
 * production order.
 *
 * Nothing is reprojected here. Masks are GeoTIFF in the source CRS — that is
 * the graded output — and the API serves web-friendly tiles for display,
 * already carrying the colour the backend assigned in `AssetRef.colour`.
 */

/** A style with no sources of its own: the dark canvas the plan asks for. */
const EMPTY_STYLE: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [
    {
      id: "ground",
      type: "background",
      paint: { "background-color": "#0a0f11" },
    },
  ],
};

/** Development only, behind VITE_ENABLE_BASEMAP. Never on the demo path. */
const BASEMAP_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [
    {
      id: "ground",
      type: "background",
      paint: { "background-color": "#0a0f11" },
    },
    { id: "osm", type: "raster", source: "osm" },
  ],
};

type Vs = Pick<ViewState, "longitude" | "latitude" | "zoom" | "bearing" | "pitch">;

interface PaneProps {
  paneId: string;
  scene: Scene | null;
  layers: LayerState[];
  baseOpacity: number;
  viewState: Vs;
  onMove: (next: Vs) => void;
  onCursor?: (position: { lat: number; lon: number } | null) => void;
  tileGrid: GeoJSON.FeatureCollection | null;
  footprint: GeoJSON.FeatureCollection | null;
  interactive: boolean;
  mapRef?: (ref: MapRef | null) => void;
  onLoad?: () => void;
  children?: React.ReactNode;
}

function MapPane({
  paneId,
  scene,
  layers,
  baseOpacity,
  viewState,
  onMove,
  onCursor,
  tileGrid,
  footprint,
  interactive,
  mapRef,
  onLoad,
  children,
}: PaneProps) {
  return (
    <Map
      ref={mapRef}
      id={paneId}
      mapStyle={ENABLE_BASEMAP ? BASEMAP_STYLE : EMPTY_STYLE}
      {...viewState}
      onMove={(event) => onMove(event.viewState)}
      onMouseMove={
        onCursor
          ? (event) => onCursor({ lat: event.lngLat.lat, lon: event.lngLat.lng })
          : undefined
      }
      onMouseOut={onCursor ? () => onCursor(null) : undefined}
      onLoad={onLoad}
      attributionControl={false}
      interactive={interactive}
      style={{ width: "100%", height: "100%" }}
    >
      {scene?.tile_url_template ? (
        <Source
          id={`${paneId}-scene`}
          type="raster"
          tiles={[resolveUrl(scene.tile_url_template)]}
          tileSize={256}
          bounds={scene.bounds_wgs84 ?? undefined}
        >
          <Layer
            id={`${paneId}-scene-layer`}
            type="raster"
            paint={{
              "raster-opacity": baseOpacity,
              "raster-fade-duration": 100,
            }}
          />
        </Source>
      ) : scene?.preview_url && scene.bounds_wgs84 ? (
        /* No tile pyramid is built for an uploaded scene, so the single
           percentile-stretched preview is placed on its own footprint instead.
           Without this the imagery window renders an empty black canvas with a
           thin outline -- the scene is loaded, prepared and answerable, and the
           map simply had nothing to paint, which reads as a broken upload.

           Quality caveat worth keeping in mind: this is one 512 px image
           stretched across the whole extent, so it will not sharpen as you zoom
           the way real tiles would. It is orientation, not analysis -- every
           measured claim still comes from a tool, with its own overlay. */
        <Source
          id={`${paneId}-scene-preview`}
          type="image"
          url={resolveUrl(scene.preview_url)}
          coordinates={[
            [scene.bounds_wgs84[0], scene.bounds_wgs84[3]],
            [scene.bounds_wgs84[2], scene.bounds_wgs84[3]],
            [scene.bounds_wgs84[2], scene.bounds_wgs84[1]],
            [scene.bounds_wgs84[0], scene.bounds_wgs84[1]],
          ]}
        >
          <Layer
            id={`${paneId}-scene-preview-layer`}
            type="raster"
            paint={{
              "raster-opacity": baseOpacity,
              "raster-fade-duration": 100,
            }}
          />
        </Source>
      ) : null}

      {/* Largest extent underneath: a broad mask drawn last would bury the
          narrow ones the answer actually turns on. */}
      {layers
        .filter(
          (layer) =>
            layer.visible &&
            (layer.asset.tile_url_template ||
              /* No tile pyramid is built for evidence, so an overlay PNG is
                 placed on its own georeferenced footprint instead. Requiring
                 tiles meant every mask the tools produced -- the NDWI water
                 mask, the texture segmentation, the change map -- was written,
                 registered, listed in the evidence panel, and never drawn. The
                 answer cited a measurement the user could not see. */
              (layer.asset.overlay_url && layer.asset.bounds_wgs84)),
        )
        .slice()
        .sort(
          (a, b) =>
            (b.asset.stats?.area_km2 ?? 0) - (a.asset.stats?.area_km2 ?? 0),
        )
        .map((layer) =>
          layer.asset.tile_url_template ? (
            <Source
              key={layer.asset.asset_id}
              id={`${paneId}-asset-${layer.asset.asset_id}`}
              type="raster"
              tiles={[resolveUrl(layer.asset.tile_url_template)]}
              tileSize={256}
              bounds={layer.asset.bounds_wgs84 ?? undefined}
            >
              <Layer
                id={`${paneId}-asset-layer-${layer.asset.asset_id}`}
                type="raster"
                paint={{
                  "raster-opacity": layer.opacity,
                  "raster-fade-duration": 100,
                  "raster-resampling": "nearest",
                }}
              />
            </Source>
          ) : (
            <Source
              key={layer.asset.asset_id}
              id={`${paneId}-asset-img-${layer.asset.asset_id}`}
              type="image"
              url={resolveUrl(layer.asset.overlay_url!)}
              coordinates={[
                [layer.asset.bounds_wgs84![0], layer.asset.bounds_wgs84![3]],
                [layer.asset.bounds_wgs84![2], layer.asset.bounds_wgs84![3]],
                [layer.asset.bounds_wgs84![2], layer.asset.bounds_wgs84![1]],
                [layer.asset.bounds_wgs84![0], layer.asset.bounds_wgs84![1]],
              ]}
            >
              <Layer
                id={`${paneId}-asset-img-layer-${layer.asset.asset_id}`}
                type="raster"
                paint={{
                  "raster-opacity": layer.opacity,
                  "raster-fade-duration": 100,
                  "raster-resampling": "nearest",
                }}
              />
            </Source>
          ),
        )}

      {/* The scene footprint, so the ground outside it reads as "no data"
          rather than as a failed render. */}
      {footprint ? (
        <Source id={`${paneId}-footprint`} type="geojson" data={footprint}>
          <Layer
            id={`${paneId}-footprint-line`}
            type="line"
            paint={{
              "line-color": "#8f9b98",
              "line-width": 1,
              "line-opacity": 0.65,
            }}
          />
        </Source>
      ) : null}

      {tileGrid ? (
        <Source id={`${paneId}-tile-grid`} type="geojson" data={tileGrid}>
          <Layer
            id={`${paneId}-tile-grid-line`}
            type="line"
            paint={{
              "line-color": "#dc4a1e",
              "line-width": 1,
              "line-opacity": 0.8,
              "line-dasharray": [3, 2],
            }}
          />
        </Source>
      ) : null}

      {children}
    </Map>
  );
}

export function SceneMap({ bundle }: { bundle: Bundle }) {
  const primaryRef = useRef<MapRef | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [cursor, setCursor] = useState<{ lat: number; lon: number } | null>(
    null,
  );

  const layers = useWorkspace((s) => s.layers);
  const baseLayer = useWorkspace((s) => s.baseLayer);
  const baseOpacity = useWorkspace((s) => s.baseOpacity);
  const showTileGrid = useWorkspace((s) => s.showTileGrid);
  const swipeEnabled = useWorkspace((s) => s.swipeEnabled);
  const swipePosition = useWorkspace((s) => s.swipePosition);
  const setSwipePosition = useWorkspace((s) => s.setSwipePosition);

  const bounds = bundle.bounds_wgs84;
  const centre = bounds
    ? {
        longitude: (bounds[0] + bounds[2]) / 2,
        latitude: (bounds[1] + bounds[3]) / 2,
      }
    : { longitude: 77.18, latitude: 28.56 };

  const [viewState, setViewState] = useState<Vs>({
    ...centre,
    zoom: 11,
    bearing: 0,
    pitch: 0,
  });

  /** A/B panes: t1 vs t2 for a bi-temporal pair, optical vs SAR otherwise. */
  const { paneA, paneB } = useMemo(() => {
    const scenes = bundle.scenes;
    if (bundle.pair_type === "bitemporal") {
      return {
        paneA: scenes.find((s) => s.role === "t1") ?? scenes[0] ?? null,
        paneB: scenes.find((s) => s.role === "t2") ?? scenes[1] ?? null,
      };
    }
    const optical =
      scenes.find((s) => s.compatibility?.modality !== "sar") ?? null;
    const sar = scenes.find((s) => s.compatibility?.modality === "sar") ?? null;
    return { paneA: optical ?? scenes[0] ?? null, paneB: sar };
  }, [bundle]);

  const singleScene = useMemo(() => {
    if (baseLayer === "none") return null;
    if (baseLayer === "sar") return paneB ?? paneA;
    return paneA ?? paneB;
  }, [baseLayer, paneA, paneB]);

  /** The tiling partition, drawn from the bundle's own tile index. */
  const tileGrid = useMemo(() => {
    if (!showTileGrid || !bundle.tiles || !bounds) return null;
    const [west, south, east, north] = bounds;
    const columns = Math.ceil(Math.sqrt(bundle.tiles.tile_count));
    const rows = Math.ceil(bundle.tiles.tile_count / columns);
    const features: GeoJSON.Feature[] = [];
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const x0 = west + ((east - west) * column) / columns;
        const x1 = west + ((east - west) * (column + 1)) / columns;
        const y0 = north - ((north - south) * (row + 1)) / rows;
        const y1 = north - ((north - south) * row) / rows;
        features.push({
          type: "Feature",
          properties: { tile: `${row}·${column}` },
          geometry: {
            type: "Polygon",
            coordinates: [
              [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0],
              ],
            ],
          },
        });
      }
    }
    return { type: "FeatureCollection", features } as GeoJSON.FeatureCollection;
  }, [showTileGrid, bundle.tiles, bounds]);

  const footprint = useMemo(() => {
    if (!bounds) return null;
    const [west, south, east, north] = bounds;
    return {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: {},
          geometry: {
            type: "Polygon",
            coordinates: [
              [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
              ],
            ],
          },
        },
      ],
    } as GeoJSON.FeatureCollection;
  }, [bounds]);

  const fit = useCallback(() => {
    const map = primaryRef.current;
    if (!map || !bounds) return;
    map.fitBounds(
      [
        [bounds[0], bounds[1]],
        [bounds[2], bounds[3]],
      ],
      { padding: 36, duration: 0 },
    );
    // fitBounds with duration 0 does not emit a move event, so the readout
    // and the comparison pane would keep the stale zoom.
    const centreAfter = map.getCenter();
    setViewState({
      longitude: centreAfter.lng,
      latitude: centreAfter.lat,
      zoom: map.getZoom(),
      bearing: map.getBearing(),
      pitch: map.getPitch(),
    });
  }, [bounds]);

  /**
   * Fit once the map has actually loaded and the container has its final
   * size. Fitting on mount races the layout and lands the scene at the wrong
   * zoom — the footprint here is smaller than a single z10 tile, so being
   * two zoom levels out means looking at an empty canvas.
   */
  const fittedRef = useRef(false);
  const handleLoad = useCallback(() => {
    fittedRef.current = true;
    fit();
  }, [fit]);

  useEffect(() => {
    const container = containerRef.current;
    const map = primaryRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      map?.resize();
      if (!fittedRef.current) fit();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [fit]);

  /* the swipe divider */
  const dragging = useRef(false);
  const moveDivider = useCallback(
    (clientX: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      setSwipePosition(
        Math.min(0.98, Math.max(0.02, (clientX - rect.left) / rect.width)),
      );
    },
    [setSwipePosition],
  );

  useEffect(() => {
    if (!swipeEnabled) return;
    const onMove = (event: PointerEvent) => {
      if (dragging.current) moveDivider(event.clientX);
    };
    const onUp = () => {
      dragging.current = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [swipeEnabled, moveDivider]);

  const canSwipe = Boolean(paneA && paneB);

  return (
    <div ref={containerRef} className="on-window relative h-full w-full">
      <MapPane
        paneId="pane-a"
        mapRef={(ref) => {
          primaryRef.current = ref;
        }}
        scene={swipeEnabled && canSwipe ? paneA : singleScene}
        layers={layers}
        baseOpacity={baseOpacity}
        viewState={viewState}
        onMove={setViewState}
        onCursor={setCursor}
        tileGrid={tileGrid}
        footprint={footprint}
        onLoad={handleLoad}
        interactive
      >
        <ScaleControl position="bottom-left" maxWidth={110} unit="metric" />
      </MapPane>

      {/* The comparison pane. Two synchronised maps rather than one, because
          a swipe has to clip the imagery itself, and a WebGL layer cannot be
          clipped from CSS while it shares a canvas. */}
      {swipeEnabled && canSwipe ? (
        <div
          className="pointer-events-none absolute inset-0 z-[5]"
          style={{ clipPath: `inset(0 0 0 ${swipePosition * 100}%)` }}
          aria-hidden="true"
        >
          <MapPane
            paneId="pane-b"
            scene={paneB}
            layers={layers}
            baseOpacity={baseOpacity}
            viewState={viewState}
            onMove={setViewState}
            tileGrid={tileGrid}
            footprint={footprint}
            interactive={false}
          />
        </div>
      ) : null}

      {swipeEnabled && canSwipe ? (
        <div
          role="separator"
          aria-label="Comparison divider"
          aria-orientation="vertical"
          aria-valuenow={Math.round(swipePosition * 100)}
          aria-valuemin={2}
          aria-valuemax={98}
          tabIndex={0}
          onPointerDown={(event: ReactPointerEvent) => {
            dragging.current = true;
            moveDivider(event.clientX);
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft") {
              setSwipePosition(Math.max(0.02, swipePosition - 0.02));
            }
            if (event.key === "ArrowRight") {
              setSwipePosition(Math.min(0.98, swipePosition + 0.02));
            }
          }}
          className="absolute inset-y-0 z-10 -ml-3 w-6 cursor-ew-resize touch-none"
          style={{ left: `${swipePosition * 100}%` }}
        >
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-signal"
          />
          <span className="pointer-events-none absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-1 border border-signal bg-window-0 px-1.5 py-1">
            <span className="t-code-sm text-signal">
              {bundle.pair_type === "bitemporal" ? "T1" : "OPT"}
            </span>
            <span aria-hidden="true" className="text-signal">
              ◂▸
            </span>
            <span className="t-code-sm text-signal">
              {bundle.pair_type === "bitemporal" ? "T2" : "SAR"}
            </span>
          </span>
        </div>
      ) : null}

      {/* readouts, in the window's own chrome */}
      <div className="pointer-events-none absolute bottom-2 right-2 z-10 flex flex-col items-end gap-1">
        <span className="t-data border border-window-2 bg-window-0/85 px-1.5 py-[3px] text-[10.5px] text-window-ink">
          {cursor
            ? formatLatLon(cursor.lat, cursor.lon)
            : "hover the scene for coordinates"}
        </span>
        <span className="t-code-sm border border-window-2 bg-window-0/85 px-1.5 py-[3px] text-window-ink-2">
          WGS84 · Z {viewState.zoom.toFixed(1)}
        </span>
      </div>

      <button
        type="button"
        onClick={fit}
        className={cn(
          "t-code-sm absolute right-2 top-2 z-10 border border-window-2 bg-window-0/85 px-2 py-[5px] text-window-ink",
          "transition-colors hover:border-window-ink-2 hover:text-white",
        )}
      >
        Fit to bounds
      </button>
    </div>
  );
}
