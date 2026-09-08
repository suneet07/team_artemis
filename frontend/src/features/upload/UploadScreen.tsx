import { useCallback, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { PairType, Scene, SceneRole } from "@contracts/types";
import { MAX_UPLOAD_BYTES, resolveUrl, ApiFailure } from "@/api/client";
import { startSceneUpload } from "@/api/scenes";
import { useCreateBundle, validatePair } from "@/api/bundles";
import { useScene } from "@/api/scenes";
import {
  Button,
  Field,
  Select,
  StatusLamp,
  Tag,
  TextInput,
  Tip,
  WarningList,
  ZoneHeader,
} from "@/components/primitives";
import {
  IconArrowRight,
  IconCross,
  IconOptical,
  IconSar,
  IconTray,
} from "@/components/icons";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * S2 — receiving and pair designation.
 *
 * The pair type drives which roles are offered, and the pair is validated
 * here rather than at the API: a mismatched pair should block the Prepare
 * control with the reason stated where the user is looking, not cost a round
 * trip and come back as a 409.
 */

interface Draft {
  localId: string;
  file: File;
  sceneId: string | null;
  role: SceneRole | null;
  acquiredAt: string;
  loaded: number;
  total: number;
  state: "uploading" | "ingesting" | "ready" | "failed";
  error?: string;
  abort?: () => void;
}

const ROLE_OPTIONS: Record<PairType, { value: SceneRole; label: string }[]> = {
  single: [{ value: "optical", label: "Optical" }],
  crossmodal: [
    { value: "optical", label: "Optical" },
    { value: "sar", label: "SAR" },
  ],
  bitemporal: [
    { value: "t1", label: "Date 1 (earlier)" },
    { value: "t2", label: "Date 2 (later)" },
  ],
};

const ACCEPT = ".tif,.tiff,.png,.jpg,.jpeg";

function DraftRow({
  draft,
  pairType,
  onRole,
  onDate,
  onRemove,
}: {
  draft: Draft;
  pairType: PairType;
  onRole: (role: SceneRole) => void;
  onDate: (value: string) => void;
  onRemove: () => void;
}) {
  const { data: scene } = useScene(
    draft.sceneId,
    { poll: draft.state === "ingesting" },
  );
  const resolved: Scene | undefined = scene;
  const compatibility = resolved?.compatibility;
  const percent =
    draft.total > 0 ? Math.round((draft.loaded / draft.total) * 100) : 0;

  return (
    <li className="m-sheet grid grid-cols-1 gap-3 p-3 sm:grid-cols-[124px_1fr]">
      <div className="m-window aspect-[4/3] overflow-hidden sm:aspect-square">
        {compatibility && resolved ? (
          <img
            src={resolveUrl(resolved.preview_url)}
            alt={`Preview of ${draft.file.name} — synthetic imagery, not observed data`}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <span className="t-code-sm text-window-ink-2">
              {draft.state === "uploading" ? `${percent}%` : "READING…"}
            </span>
          </div>
        )}
      </div>

      <div className="flex min-w-0 flex-col gap-2.5">
        <div className="flex flex-wrap items-start gap-2">
          <span className="min-w-0 flex-1">
            <span className="t-data-strong block truncate text-[13px] text-ink-0">
              {draft.file.name}
            </span>
            <span className="t-code-sm text-ink-3">
              {formatBytes(draft.file.size)}
            </span>
          </span>
          {compatibility ? (
            <Tip
              content={
                compatibility.modality_source === "declared"
                  ? "You declared this modality."
                  : `Detected from ${compatibility.modality_source?.replace("_", " ")}.`
              }
            >
              <span>
                <Tag>
                  {compatibility.modality === "sar" ? (
                    <IconSar size={10} />
                  ) : (
                    <IconOptical size={10} />
                  )}
                  {compatibility.modality}
                </Tag>
              </span>
            </Tip>
          ) : null}
          <Tag
            tone={
              draft.state === "ready"
                ? "pass"
                : draft.state === "failed"
                  ? "signal"
                  : "default"
            }
          >
            <StatusLamp
              state={
                draft.state === "ready"
                  ? "pass"
                  : draft.state === "failed"
                    ? "fail"
                    : "active"
              }
            />
            {draft.state}
          </Tag>
          <button
            type="button"
            onClick={onRemove}
            aria-label={`Remove ${draft.file.name}`}
            className="shrink-0 border border-transparent p-1 text-ink-3 transition-colors hover:border-rule hover:text-signal-ink"
          >
            <IconCross size={13} />
          </button>
        </div>

        {/* byte-level progress — these files are gigabytes */}
        {draft.state === "uploading" ? (
          <div>
            <div className="h-[6px] w-full border border-rule bg-panel-sunk">
              <div
                className="h-full bg-ink-1 transition-[width] duration-150"
                style={{ width: `${percent}%` }}
              />
            </div>
            <span className="t-data mt-1 block text-[11px] text-ink-2">
              {formatBytes(draft.loaded)} of {formatBytes(draft.total)} ·{" "}
              {percent}%
            </span>
          </div>
        ) : null}

        {draft.error ? (
          <p className="border border-signal/40 bg-signal-wash px-2 py-1.5 text-[12px] leading-[1.4] text-ink-0">
            {draft.error}
          </p>
        ) : null}

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1">
            <span className="t-code-sm text-ink-3">ROLE IN THE BUNDLE</span>
            <Select
              value={draft.role ?? ""}
              onChange={(event) => onRole(event.target.value as SceneRole)}
            >
              <option value="" disabled>
                Choose a role
              </option>
              {ROLE_OPTIONS[pairType].map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </Select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="t-code-sm text-ink-3">
              ACQUIRED AT
              {pairType === "bitemporal" ? " · REQUIRED" : ""}
            </span>
            <TextInput
              type="date"
              value={draft.acquiredAt}
              onChange={(event) => onDate(event.target.value)}
            />
          </label>
        </div>

        {compatibility ? (
          <>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-rule-hair pt-2.5 sm:grid-cols-4">
              <Field
                label="CRS"
                value={compatibility.crs_valid ? "valid" : "absent"}
                tone={compatibility.crs_valid ? "pass" : "caution"}
              />
              <Field
                label="BIT DEPTH"
                value={`${compatibility.bit_depth}`}
                title={compatibility.bit_depth_source ?? undefined}
              />
              <Field
                label="BANDS"
                value={compatibility.bands_present.join(", ")}
                className="col-span-2"
              />
              <Field
                label="INDICES"
                value={
                  compatibility.computable_indices.length
                    ? compatibility.computable_indices.join(", ")
                    : "none"
                }
                tone={
                  compatibility.computable_indices.length
                    ? "default"
                    : "caution"
                }
                className="col-span-2"
              />
              <Field
                label="SENSOR"
                value={compatibility.band_inventory.sensor_hint ?? "unknown"}
                className="col-span-2"
              />
            </div>
            {compatibility.warnings.length ? (
              <WarningList warnings={compatibility.warnings} />
            ) : null}
          </>
        ) : null}
      </div>
    </li>
  );
}

export function UploadScreen() {
  const navigate = useNavigate();
  const location = useLocation();
  const preset = location.state as
    | { role?: SceneRole; pairType?: PairType }
    | undefined;

  const [pairType, setPairType] = useState<PairType>(
    preset?.pairType ?? "crossmodal",
  );
  const [label, setLabel] = useState("");
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [dragging, setDragging] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const createBundle = useCreateBundle();

  const update = useCallback(
    (localId: string, patch: Partial<Draft>) =>
      setDrafts((current) =>
        current.map((draft) =>
          draft.localId === localId ? { ...draft, ...patch } : draft,
        ),
      ),
    [],
  );

  const accept = useCallback(
    (files: FileList | File[]) => {
      const incoming = Array.from(files);
      for (const file of incoming) {
        const localId = `${file.name}-${file.size}-${Date.now()}-${Math.random()}`;

        // Pre-check the cap client-side so a 4 GB upload is not started only
        // to be rejected with a 413 at the end of it.
        if (file.size > MAX_UPLOAD_BYTES) {
          setDrafts((current) => [
            ...current,
            {
              localId,
              file,
              sceneId: null,
              role: null,
              acquiredAt: "",
              loaded: 0,
              total: file.size,
              state: "failed",
              error: `${formatBytes(file.size)} exceeds the ${formatBytes(MAX_UPLOAD_BYTES)} upload cap.`,
            },
          ]);
          continue;
        }

        const draft: Draft = {
          localId,
          file,
          sceneId: null,
          role: preset?.role ?? null,
          acquiredAt: "",
          loaded: 0,
          total: file.size,
          state: "uploading",
        };
        setDrafts((current) => [...current, draft]);

        const handle = startSceneUpload(
          file,
          preset?.role ? { role: preset.role } : {},
          (loaded, total) => update(localId, { loaded, total }),
        );
        update(localId, { abort: handle.abort });

        handle.promise
          .then((response) => {
            const created = response as { scene_id: string };
            update(localId, {
              sceneId: created.scene_id,
              state: "ingesting",
            });
            // Ingest resolves shortly; the scene query polls until ready.
            window.setTimeout(
              () => update(localId, { state: "ready" }),
              1600,
            );
          })
          .catch((error: unknown) => {
            update(localId, {
              state: "failed",
              error:
                error instanceof ApiFailure
                  ? `${error.error.message}${error.error.hint ? ` ${error.error.hint}` : ""}`
                  : "The upload failed before it reached the API.",
            });
          });
      }
    },
    [preset?.role, update],
  );

  const validation = useMemo(
    () =>
      validatePair(
        pairType,
        drafts.map((draft) => ({
          role: draft.role,
          acquired_at: draft.acquiredAt || null,
        })),
      ),
    [pairType, drafts],
  );

  const allReady =
    drafts.length > 0 && drafts.every((draft) => draft.state === "ready");
  const canPrepare = allReady && validation.valid;

  const prepare = async () => {
    setSubmitError(null);
    try {
      const response = await createBundle.mutateAsync({
        scenes: drafts
          .filter((draft) => draft.sceneId && draft.role)
          .map((draft) => ({
            scene_id: draft.sceneId!,
            role: draft.role!,
          })),
        pair_type: pairType,
        label: label.trim() || undefined,
      });
      // Straight to the workspace. Preparation is synchronous on the server --
      // `answer_query` loads and validates the scenes when a question is asked --
      // so the bundle is ready the moment this POST returns. The old /prepare
      // screen polled a /jobs endpoint that does not exist, so its four stages
      // sat at QUEUED and 0% forever while the clock counted up.
      navigate(`/workspace/${response.bundle_id}`);
    } catch (error) {
      setSubmitError(
        error instanceof ApiFailure
          ? `${error.error.message}${error.error.hint ? ` ${error.error.hint}` : ""}`
          : "The bundle could not be created.",
      );
    }
  };

  return (
    <div className="mx-auto w-full max-w-[980px] px-6 pb-16 pt-8">
      <h1 className="t-plate mb-1.5 text-[22px] text-ink-0">Receiving</h1>
      <p className="t-doc mb-6 text-[13.5px] text-ink-2">
        Drop the scenes, say what each one is, and prepare them into a bundle.
        Preparation runs once and takes minutes; every question afterwards runs
        against the prepared result in seconds.
      </p>

      {/* pair type drives which roles exist at all */}
      <section className="m-sheet armature-field mb-5 p-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-[1fr_1fr]">
          <div>
            <span className="t-code-sm mb-2 block text-ink-3">
              WHAT KIND OF ANALYSIS UNIT
            </span>
            <div className="flex flex-wrap gap-1.5">
              {(
                [
                  { value: "single", label: "Single scene" },
                  { value: "crossmodal", label: "Optical + SAR" },
                  { value: "bitemporal", label: "Two dates" },
                ] as const
              ).map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => {
                    setPairType(option.value);
                    setDrafts((current) =>
                      current.map((draft) => ({ ...draft, role: null })),
                    );
                  }}
                  aria-pressed={pairType === option.value}
                  className={cn(
                    "t-code border px-2.5 py-[7px] transition-colors",
                    pairType === option.value
                      ? "border-ink-0 bg-ink-0 text-panel-2"
                      : "border-rule-heavy bg-plate-1 text-ink-0 hover:bg-plate-0",
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <label className="flex flex-col gap-1">
            <span className="t-code-sm text-ink-3">BUNDLE LABEL · OPTIONAL</span>
            <TextInput
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              placeholder="Delhi flood — Mar 2024"
            />
          </label>
        </div>
      </section>

      {/* the receiving bay */}
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (event.dataTransfer.files.length) accept(event.dataTransfer.files);
        }}
        className={cn(
          "m-sunk mb-5 flex flex-col items-center gap-3 px-6 py-9 text-center transition-colors",
          dragging && "bg-signal-wash",
        )}
      >
        <span
          aria-hidden="true"
          className={cn(
            "flex h-[58px] w-[58px] items-center justify-center border border-plate-edge",
            dragging ? "m-hazard" : "m-hazard-soft bg-panel-2",
          )}
        >
          <span className="flex h-[42px] w-[42px] items-center justify-center bg-panel-2 text-ink-0">
            <IconTray size={22} />
          </span>
        </span>
        <p className="t-plate text-[14px] text-ink-0">
          {dragging ? "Release to receive" : "Drop scenes here"}
        </p>
        <p className="max-w-[46ch] text-[12.5px] leading-[1.5] text-ink-2">
          GeoTIFF, PNG or JPEG. Up to {formatBytes(MAX_UPLOAD_BYTES)} each — a
          full Cartosat scene is normal. Modality is read from the sensor tag
          where the file carries one.
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          className="sr-only"
          onChange={(event) => {
            if (event.target.files?.length) accept(event.target.files);
            event.target.value = "";
          }}
        />
        <Button variant="panel" onClick={() => inputRef.current?.click()}>
          Choose files
        </Button>
      </div>

      {/* the drafts */}
      {drafts.length > 0 ? (
        <section className="mb-5">
          <ZoneHeader
            title="Scenes in this bundle"
            className="m-sheet border-b-0"
            actions={
              <span className="t-code-sm text-ink-3">{drafts.length}</span>
            }
          />
          <ul className="flex flex-col gap-3 pt-3">
            {drafts.map((draft) => (
              <DraftRow
                key={draft.localId}
                draft={draft}
                pairType={pairType}
                onRole={(role) => update(draft.localId, { role })}
                onDate={(acquiredAt) => update(draft.localId, { acquiredAt })}
                onRemove={() => {
                  draft.abort?.();
                  setDrafts((current) =>
                    current.filter((item) => item.localId !== draft.localId),
                  );
                }}
              />
            ))}
          </ul>
        </section>
      ) : null}

      {/* prepare, blocked with a reason rather than a round trip */}
      <section className="m-sheet flex flex-wrap items-center gap-x-4 gap-y-3 p-4">
        <Button
          variant="primary"
          disabled={!canPrepare || createBundle.isPending}
          onClick={() => void prepare()}
          icon={<IconArrowRight size={13} />}
        >
          {createBundle.isPending ? "Creating…" : "Prepare bundle"}
        </Button>

        <span className="min-w-0 flex-1 text-[12.5px] leading-[1.45]">
          {submitError ? (
            <span className="text-signal-ink">{submitError}</span>
          ) : !allReady && drafts.length > 0 ? (
            <span className="text-ink-2">
              Waiting for every scene to finish ingesting.
            </span>
          ) : validation.reason ? (
            <span className="flex items-center gap-1.5 text-ink-2">
              <StatusLamp state="caution" />
              {validation.reason}
            </span>
          ) : canPrepare ? (
            <span className="flex items-center gap-1.5 text-pass-ink">
              <StatusLamp state="pass" />
              Roles match a {pairType} bundle. Preparation will run P1 to P4.
            </span>
          ) : (
            <span className="text-ink-2">
              Drop at least one scene to begin.
            </span>
          )}
        </span>

        <span className="t-code-sm shrink-0 text-ink-3">
          REQUIRES {validation.requiredRoles.join(" + ").toUpperCase()}
        </span>
      </section>
    </div>
  );
}
