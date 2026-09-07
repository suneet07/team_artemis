import type { AssetRef } from "@contracts/types";
import { useDisagreementCauses } from "@/api/meta";
import { resolveUrl } from "@/api/client";
import { IconOptical, IconSar } from "@/components/icons";
import { Field, StatusLamp, Tag, Tip } from "@/components/primitives";
import type { LiveAgreement } from "@/store/runs";
import { formatArea } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * The disagreement panel (§7.3).
 *
 * When optical and SAR reach different conclusions, most systems average
 * them and report a number. This one shows both readings side by side, names
 * the physical cause, says which sensor is trusted and why, and lowers the
 * confidence — which is the whole argument for building a cross-modal
 * pipeline in the first place. It gets real estate accordingly.
 */

function ModalityColumn({
  modality,
  asset,
  trusted,
}: {
  modality: "optical" | "sar";
  asset: AssetRef | undefined;
  trusted: boolean;
}) {
  const Mark = modality === "sar" ? IconSar : IconOptical;
  return (
    <div
      className={cn(
        "flex min-w-0 flex-col border",
        trusted ? "border-pass bg-pass-wash" : "border-rule bg-panel-1",
      )}
    >
      <div className="flex items-center gap-2 border-b border-inherit px-2.5 py-2">
        <Mark size={14} className="shrink-0 text-ink-1" />
        <span className="t-code text-ink-0">
          {modality === "sar" ? "SAR says" : "Optical says"}
        </span>
        {trusted ? (
          <Tag tone="pass" className="ml-auto">
            <StatusLamp state="pass" />
            TRUSTED
          </Tag>
        ) : (
          <Tag className="ml-auto">SUPERSEDED</Tag>
        )}
      </div>

      <figure className="m-window mx-2.5 mt-2.5 aspect-[4/3] overflow-hidden">
        {asset?.overlay_url ? (
          <img
            src={resolveUrl(asset.overlay_url)}
            alt={`${asset.label} — synthetic imagery, not observed data`}
            loading="lazy"
            className="h-full w-full object-cover"
            style={{ imageRendering: "pixelated" }}
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <span className="t-code-sm text-window-ink-2">NO MASK</span>
          </div>
        )}
      </figure>

      <div className="flex flex-col gap-2 px-2.5 py-2.5">
        <Field
          label="EXTENT"
          value={formatArea(asset?.stats?.area_km2)}
          title={
            asset?.stats?.area_km2 !== undefined
              ? `${asset.stats.area_km2} km²`
              : undefined
          }
        />
        {/* The tool name is one of the graded fields — it wraps rather than
            truncating, whatever the column width. */}
        <div className="flex min-w-0 flex-col gap-[3px]">
          <span className="t-code-sm text-ink-3">FROM TOOL</span>
          <span className="t-data break-words text-[12px] leading-[1.35] text-ink-0">
            {asset?.produced_by ?? "—"}
          </span>
        </div>
      </div>
    </div>
  );
}

export function DisagreementPanel({
  agreement,
  evidence,
}: {
  agreement: LiveAgreement;
  evidence: AssetRef[];
}) {
  const { data: causes } = useDisagreementCauses();
  const cause = agreement.disagreement_cause
    ? causes?.byCode.get(agreement.disagreement_cause)
    : undefined;

  const consistent = agreement.verdict === "consistent";
  const trusted =
    agreement.winning_modality ?? cause?.trusted_modality ?? null;

  // The two masks that disagreed, matched by which modality produced them.
  //
  // Both halves of this used to be too narrow, and the panel showed "NO MASK"
  // beside an answer that was reporting an agreed extent -- so the one screen
  // whose job is to show the disagreement showed nothing on a real conflict:
  //
  //   optical: matched only `spectral_index`. The optical opinion comes from
  //            `texture_seg` whenever the bands cannot compute an index, which
  //            is every scene that arrives without band descriptions.
  //   sar:     also required /water|flood/ in the label. Fusion runs on
  //            whatever target was asked for, and a built-up query never
  //            matched.
  //
  // Matched on the producing tool alone, then narrowed to the fused target when
  // the label names it -- rather than the other way round, which is what made
  // the common case fall through.
  const OPTICAL_TOOLS = ["spectral_index", "texture_seg"];
  const target = (agreement as { target?: string | null }).target;

  const pick = (tools: string[]) => {
    const candidates = evidence.filter(
      (asset) => tools.includes(asset.produced_by) && asset.overlay_url,
    );
    if (target) {
      const onTarget = candidates.find((asset) =>
        new RegExp(target, "i").test(asset.label),
      );
      if (onTarget) return onTarget;
    }
    return candidates[0];
  };

  const opticalAsset = pick(OPTICAL_TOOLS);
  const sarAsset = pick(["sar_backscatter"]);

  return (
    <section
      className={cn(
        "print-break border",
        consistent ? "border-pass/40" : "border-signal/45",
      )}
    >
      <header
        className={cn(
          "flex flex-wrap items-center gap-x-3 gap-y-2 border-b px-3 py-2.5",
          consistent
            ? "border-pass/30 bg-pass-wash"
            : "border-signal/30 bg-signal-wash",
        )}
      >
        <StatusLamp state={consistent ? "pass" : "fail"} size={11} />
        <h3 className="t-plate text-[13px] text-ink-0">
          {consistent
            ? "Optical and SAR agree"
            : "Optical and SAR disagree"}
        </h3>
        {agreement.iou !== null && agreement.iou !== undefined ? (
          <Tip content={`Intersection over union of the two masks: ${agreement.iou}. Computed by the backend; the frontend never derives it.`}>
            <span>
              <Tag tone={consistent ? "pass" : "signal"}>
                IoU {agreement.iou}
              </Tag>
            </span>
          </Tip>
        ) : null}
        {trusted ? (
          <Tag tone="solid" className="ml-auto">
            {trusted.toUpperCase()} TRUSTED
          </Tag>
        ) : null}
      </header>

      <div className="grid grid-cols-1 gap-3 p-3 sm:grid-cols-2">
        <ModalityColumn
          modality="optical"
          asset={opticalAsset}
          trusted={trusted === "optical"}
        />
        <ModalityColumn
          modality="sar"
          asset={sarAsset}
          trusted={trusted === "sar"}
        />
      </div>

      {!consistent ? (
        <div className="border-t border-signal/25 bg-panel-1 px-3 py-3">
          <div className="mb-1.5 flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="t-code-sm text-ink-3">CAUSE</span>
            <span className="t-code text-signal-ink">
              {cause?.label ?? agreement.disagreement_cause ?? "unclassified"}
            </span>
            {agreement.disagreement_cause ? (
              <span className="t-data text-[11px] text-ink-3">
                {agreement.disagreement_cause}
              </span>
            ) : null}
          </div>
          <p className="t-doc text-[12.5px] text-ink-1">
            {cause?.explanation ??
              agreement.explanation ??
              "The two modalities produced different extents and the backend did not classify the cause."}
          </p>
          {agreement.explanation && cause?.explanation ? (
            <p className="mt-2 border-t border-rule-hair pt-2 text-[12px] leading-[1.5] text-ink-2">
              {agreement.explanation}
            </p>
          ) : null}
          <p className="mt-2.5 flex items-start gap-1.5 text-[11.5px] leading-[1.5] text-ink-2">
            <span aria-hidden="true" className="text-signal">
              △
            </span>
            The reported confidence is lowered because the two sensors did not
            corroborate each other. The trusted extent is the one carried into
            the answer.
          </p>
        </div>
      ) : null}
    </section>
  );
}
