import { useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useBundles, useDemoBundles } from "@/api/bundles";
import { SlaGauge, type GaugeTrack } from "@/components/SlaGauge";
import {
  Button,
  EmptyState,
  ZoneHeader,
} from "@/components/primitives";
import { IconTray } from "@/components/icons";
import { useRuns } from "@/store/runs";
import { BundleRecord } from "./BundleRecord";

/**
 * S1 — Sessions.
 *
 * The first viewport has one job: make the two-SLA model legible before
 * anything is clicked, because that is the claim the whole system rests on.
 * The gauge sits above the fold with the last measured values on it.
 */
export function LandingScreen() {
  // Still fetched: session bundles are de-duplicated against these, and the
  // gauge falls back to a demo bundle's recorded prep time. They are no
  // longer listed as a section of their own.
  const { data: demoBundles } = useDemoBundles();
  const { data: sessionBundles } = useBundles();
  const runs = useRuns((s) => s.runs);
  const navigate = useNavigate();

  const recent = useMemo(
    () =>
      (sessionBundles ?? []).filter(
        (bundle) => !(demoBundles ?? []).some((d) => d.bundle_id === bundle.bundle_id),
      ),
    [sessionBundles, demoBundles],
  );

  // Both SLAs are reported from what this session actually measured; the
  // pre-warmed bundle's recorded prep time stands in until a bundle is
  // prepared here, and it is labelled as such.
  const lastQuery = useMemo(() => {
    const finished = Object.values(runs)
      .filter((run) => run.totalLatencyMs !== null)
      .sort((a, b) => (b.finishedAt ?? 0) - (a.finishedAt ?? 0));
    return finished[0] ?? null;
  }, [runs]);

  const preparedHere = useMemo(
    () => recent.find((bundle) => bundle.prep_ms !== null) ?? null,
    [recent],
  );

  const tracks: GaugeTrack[] = [
    {
      code: "SLA-1",
      label: "Scene preparation",
      budgetMs: 300_000,
      budgetLabel: "≤ 5 MIN PER SCENE",
      measuredMs:
        preparedHere?.prep_ms ?? demoBundles?.[0]?.prep_ms ?? null,
      measuredNote: preparedHere
        ? "MEASURED THIS SESSION · P1–P4"
        : "RECORDED ON THE PRE-WARMED BUNDLE · P1–P4",
    },
    {
      code: "SLA-2",
      label: "Query latency",
      budgetMs: 20_000,
      budgetLabel: "< 20 S PER QUESTION",
      measuredMs: lastQuery?.totalLatencyMs ?? null,
      measuredNote: "MEASURED THIS SESSION · LAST QUERY",
    },
  ];

  return (
    <div className="mx-auto w-full max-w-[1180px] px-6 pb-16 pt-9">
      {/* ── thesis ──────────────────────────────────────────────────── */}
      <section className="mb-10 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(360px,46%)] lg:items-start">
        <div>
          <h1 className="t-plate max-w-[15ch] text-[clamp(30px,4.6vw,52px)] text-ink-0">
            Every answer arrives with the record that produced it
          </h1>
          <p className="t-doc mt-5 text-[14.5px] text-ink-1">
            Ask a question of optical, SAR, bi-temporal or cross-modal imagery.
            A deterministic pipeline validates the inputs, selects the task,
            enforces every tool parameter against its manifest, and fuses
            optical with SAR — then hands back the mask, the statistic, and the
            trace showing exactly how each was obtained.
          </p>
          <p className="t-doc mt-3 text-[14.5px] text-ink-2">
            When the imagery cannot honestly support a question, the system
            says so and offers the fix. That refusal is a graded deliverable,
            not an error.
          </p>
        </div>

        <div className="m-sheet armature-field p-5">
          <div className="mb-4 flex items-baseline justify-between gap-3 border-b border-rule pb-2.5">
            <h2 className="t-code text-ink-1">Service budgets, measured</h2>
            <span className="t-code-sm text-ink-3">TWO CLOCKS</span>
          </div>
          <SlaGauge tracks={tracks} />
          <p className="mt-5 border-t border-rule-hair pt-3 text-[12px] leading-[1.5] text-ink-2">
            Preparation runs once per scene and is the slow clock. Questions
            run against the prepared bundle and are the fast one. They are
            reported separately because averaging them would hide both.
          </p>
        </div>
      </section>

      {/* ── this session ────────────────────────────────────────────── */}
      <section className="mb-10">
        <ZoneHeader
          code="S1·B"
          title="Prepared in this session"
          className="m-sheet border-b-0"
        />
        <div className="pt-4">
          {recent.length === 0 ? (
            <EmptyState
              code="NO RECORDS"
              title="Nothing prepared yet"
              action={
                <Button
                  variant="panel"
                  onClick={() => navigate("/upload")}
                  icon={<IconTray size={13} />}
                >
                  Open receiving
                </Button>
              }
            >
              Scenes are prepared once — ingest, SAR normalisation,
              coregistration and tiling, a few minutes per scene. Questions
              against a prepared bundle then answer in seconds.
            </EmptyState>
          ) : (
            <div className="flex flex-col gap-4">
              {recent.map((bundle, index) => (
                <BundleRecord
                  key={bundle.bundle_id}
                  bundle={bundle}
                  index={index}
                />
              ))}
            </div>
          )}
        </div>
      </section>

      {/* ── receiving bay ───────────────────────────────────────────── */}
      <section>
        <Link
          to="/upload"
          className="m-sunk armature-field group flex flex-col items-start gap-3 px-6 py-7 transition-colors hover:bg-panel-1 sm:flex-row sm:items-center sm:gap-6"
        >
          <span
            aria-hidden="true"
            className="m-hazard flex h-[52px] w-[52px] shrink-0 items-center justify-center border border-plate-edge"
          >
            <span className="flex h-[38px] w-[38px] items-center justify-center bg-panel-2 text-ink-0">
              <IconTray size={20} />
            </span>
          </span>
          <span className="min-w-0 flex-1">
            <span className="t-plate block text-[15px] text-ink-0">
              Receiving bay
            </span>
            <span className="mt-1.5 block text-[12.5px] leading-[1.5] text-ink-2">
              Drop GeoTIFF, PNG or JPEG scenes, designate their roles, and
              prepare a new bundle. Files up to 4 GB each — a full Cartosat
              scene is normal.
            </span>
          </span>
          <span className="t-code shrink-0 border border-rule-heavy bg-plate-1 px-3 py-2 text-ink-0 transition-colors group-hover:bg-plate-0">
            S2 · Open
          </span>
        </Link>
      </section>
    </div>
  );
}
