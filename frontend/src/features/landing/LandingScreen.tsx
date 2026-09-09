import { useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useBundles, useDemoBundles } from "@/api/bundles";
import {
  Button,
  EmptyState,
  Skeleton,
  Tip,
  ZoneHeader,
} from "@/components/primitives";
import { IconArrowRight, IconTray } from "@/components/icons";
import { BundleRecord } from "./BundleRecord";
import { BenchmarkRecord } from "./BenchmarkRecord";
import { CapabilityRecord } from "./CapabilityRecord";
import { ConsoleMock } from "./ConsoleMock";
import { RECORD_SUMMARY } from "./benchmarks.data";
import { PROOF_RAIL, SPEC_STRIP } from "./hero.data";

/**
 * S1 — Sessions, and the front page of the document.
 *
 * The screen reads top to bottom as one argument, and the order is the
 * argument:
 *
 * 1. **Masthead** — what the system is, in one sentence, beside a drawing of
 *    it answering. A judge who reads nothing else has seen a question, an
 *    answer, the evidence and the trace.
 * 2. **The capability record** — the four things it answers and the one
 *    layer that decides which of them should.
 * 3. **The evaluation record** — every score, every comparison system, every
 *    blind floor, untruncated.
 * 4. **The bundles** — pre-warmed and prepared-here, the three-second path to
 *    asking it something yourself.
 *
 * The masthead and the two records are editorial and do not move; the bundle
 * sections are live state from this session. That boundary is deliberate —
 * the numbers measured before the venue and the numbers being measured in the
 * room are never mixed into one figure.
 */
export function LandingScreen() {
  const { data: demoBundles, isLoading: loadingDemo } = useDemoBundles();
  const { data: sessionBundles } = useBundles();
  const navigate = useNavigate();

  const recent = useMemo(
    () =>
      (sessionBundles ?? []).filter(
        (bundle) =>
          !(demoBundles ?? []).some((d) => d.bundle_id === bundle.bundle_id),
      ),
    [sessionBundles, demoBundles],
  );

  /** The three-second path: straight into a bundle that needs no preparation. */
  const firstDemo = demoBundles?.[0] ?? null;

  return (
    <div className="mx-auto w-full max-w-[1180px] px-6 pb-16 pt-8">
      {/* ── masthead ────────────────────────────────────────────────── */}
      <section className="mb-10 grid gap-x-10 gap-y-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,470px)] lg:items-start">
        <div className="min-w-0">
          {/* the docket line, the way a controlled document is stamped */}
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-b border-rule pb-2.5">
            <span className="t-code-sm border border-rule-heavy bg-panel-sunk px-1.5 py-[3px] text-ink-1">
              PS SIH26167
            </span>
            <span className="t-code-sm text-ink-2">Space Technology</span>
            <span aria-hidden="true" className="text-rule">
              ·
            </span>
            <span className="t-code-sm text-ink-2">Software</span>
            <span aria-hidden="true" className="text-rule">
              ·
            </span>
            <span className="t-code-sm text-ink-2">
              ISRO / Space Applications Centre
            </span>
          </div>

          <h1 className="t-plate mt-5 max-w-[19ch] text-[clamp(27px,3.4vw,40px)] text-ink-0">
            Ask satellite imagery a question.
            {/* The second line is the promise, and it is set a size down: it
                qualifies the first line rather than competing with it. */}
            <span className="mt-2 block text-[0.72em] leading-[1.15] text-ink-2">
              Get the answer, the evidence, and the reasoning.
            </span>
          </h1>

          <p className="t-doc mt-5 text-[14.5px] text-ink-1">
            SatQuery AI is a vision-language system for remote-sensing
            interpretation: one 4-billion-parameter Apache-2.0 backbone, two
            40.3M LoRA adapters, an MIT-licensed radar classifier and a
            deterministic geospatial toolchain, bound together by a rules-first
            orchestrator. It takes optical scenes, SAR scenes, before-and-after
            pairs, or both sensors over the same ground, and answers questions
            about them in plain language.
          </p>
          <p className="t-doc mt-3 text-[14.5px] text-ink-2">
            Where a measurement can be computed, the orchestrator bypasses the
            network and computes it — then hands back the mask as a GeoTIFF,
            the threshold it chose and why, the alignment error in pixels, and
            the trace of every step. Where two sensors disagree and no physical
            rule explains it, the system reports only the agreed extent, at
            reduced confidence, and says that it did. It is built to be
            checked, and it says when it does not know.
          </p>

          {/* ── the way in ──────────────────────────────────────────── */}
          {/* The primary control is the one thing on this screen a judge is
              meant to press, so it is sized as a keyed switch rather than as
              a link: taller, wider, and carrying what happens next under it.
              The other two are deliberately left at panel scale. */}
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Button
              variant="primary"
              disabled={!firstDemo}
              onClick={() =>
                firstDemo && navigate(`/workspace/${firstDemo.bundle_id}`)
              }
              icon={<IconArrowRight size={16} />}
              className="min-h-[46px] gap-2.5 px-5 text-[12.5px] tracking-[0.13em]"
            >
              {firstDemo ? "Ask a prepared scene" : "Loading prepared scenes"}
            </Button>
            <Button
              variant="panel"
              onClick={() => navigate("/upload")}
              icon={<IconTray size={14} />}
              className="min-h-[46px] px-4 text-[11.5px]"
            >
              Receiving bay
            </Button>
            <a
              href="#evaluation"
              className="t-code border border-transparent px-1 py-[7px] text-ink-2 underline decoration-rule underline-offset-4 transition-colors hover:text-ink-0"
            >
              The evaluation record ↓
            </a>
          </div>
          <p className="t-code-sm mt-2.5 text-ink-3">
            No upload needed · answers in seconds
          </p>

          {/* ── the four numerals ───────────────────────────────────── */}
          <div className="m-sheet armature-field mt-6">
            <div className="grid grid-cols-2 sm:grid-cols-4">
              {PROOF_RAIL.map((proof) => (
                <div
                  key={proof.benchmark}
                  className="min-w-0 border-b border-r border-rule-hair px-3 py-2.5 last:border-r-0 sm:border-b-0 [&:nth-child(2)]:border-r-0 sm:[&:nth-child(2)]:border-r"
                >
                  <span className="t-data-strong block text-[21px] leading-none text-ink-0">
                    {proof.value}
                  </span>
                  <span className="t-code-sm mt-1.5 block text-ink-1">
                    {proof.benchmark}
                  </span>
                  <span className="mt-1 block text-[11px] leading-[1.35] text-ink-2">
                    {proof.against}
                  </span>
                </div>
              ))}
            </div>
            <p className="border-t border-rule-hair bg-panel-1 px-3 py-2 text-[11.5px] leading-[1.4] text-ink-2">
              Measured on held-out public splits with the in-repo comparator,
              and reported beside the score a system reaches{" "}
              <em className="not-italic text-ink-0">without opening the image</em>.
              The full record, including the one benchmark this system does not
              win, is below.
            </p>
          </div>

          {/* ── the specification plate ─────────────────────────────── */}
          <dl className="mt-4 flex flex-wrap gap-x-5 gap-y-2.5 border-t border-rule pt-3">
            {SPEC_STRIP.map((spec) => (
              <div key={spec.label} className="min-w-0">
                <dt className="t-code-sm text-ink-3">{spec.label}</dt>
                <dd className="t-data mt-[3px] text-[11.5px] text-ink-1">
                  {spec.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        {/* the instrument, doing the thing the sentence describes */}
        <ConsoleMock />
      </section>

      {/* ── what it answers ─────────────────────────────────────────── */}
      <section id="capabilities" className="mb-10 scroll-mt-4">
        <ZoneHeader
          title="What it answers — four capabilities, one orchestrator"
          className="m-sheet border-b-0"
          actions={
            <span className="t-code-sm hidden text-ink-3 sm:inline">
              ONE 4B BACKBONE · ADAPTERS HOT-SWAPPED
            </span>
          }
        />
        <div className="pt-4">
          <CapabilityRecord />
        </div>
      </section>

      {/* ── the evaluation record ───────────────────────────────────── */}
      <section id="evaluation" className="mb-10 scroll-mt-4">
        <ZoneHeader
          title="Evaluation record"
          className="m-sheet border-b-0"
          actions={
            <span className="t-code-sm hidden text-ink-3 sm:inline">
              {RECORD_SUMMARY.benchmarks} BENCHMARKS · {RECORD_SUMMARY.systems}{" "}
              SYSTEMS · {RECORD_SUMMARY.wins} WIN / {RECORD_SUMMARY.level} LEVEL
            </span>
          }
        />
        <div className="pt-4">
          <BenchmarkRecord />
        </div>
      </section>

      {/* ── pre-warmed records ──────────────────────────────────────── */}
      <section id="bundles" className="mb-10 scroll-mt-4">
        <ZoneHeader
          title="Pre-warmed bundles — ready without preparation"
          className="m-sheet border-b-0"
          actions={
            <span className="t-code-sm text-ink-3">
              {demoBundles?.length ?? 0} AVAILABLE
            </span>
          }
        />
        <div className="flex flex-col gap-4 pt-4">
          {loadingDemo ? (
            <>
              <Skeleton className="h-[200px] w-full" />
              <Skeleton className="h-[200px] w-full" />
            </>
          ) : (
            (demoBundles ?? []).map((bundle, index) => (
              <BundleRecord
                key={bundle.bundle_id}
                bundle={bundle}
                index={index}
              />
            ))
          )}
        </div>
      </section>

      {/* ── this session ────────────────────────────────────────────── */}
      <section className="mb-10">
        <ZoneHeader
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
          <Tip content="Preparation is the slow clock: ingest, SAR normalisation, coregistration and tiling. It runs once per scene.">
            <span className="t-code shrink-0 cursor-help border border-rule-heavy bg-plate-1 px-3 py-2 text-ink-0 transition-colors group-hover:bg-plate-0">
              Open
            </span>
          </Tip>
        </Link>
      </section>
    </div>
  );
}
