import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { resolveUrl } from "@/api/client";
import {
  type GalleryItem,
  type GalleryStatus,
  useGallery,
  useOpenGalleryItem,
} from "@/api/gallery";
import {
  Button,
  EmptyState,
  Sheet,
  Skeleton,
  Tag,
  Tip,
  ZoneHeader,
} from "@/components/primitives";

/**
 * S8 — the testing corpus.
 *
 * Every item is a row from the **test or held-out** split of a benchmark one of
 * the adapters was scored on, shown with the gold answer that benchmark
 * published. None of it was trained on. That is the whole point: a visitor can
 * pick an image whose correct answer was fixed by someone else, ask the system
 * the same question, and compare.
 *
 * The honesty rules this page follows, because a gallery is the easiest place
 * in a product to quietly overclaim:
 *
 *  - **The status is shown, never flattened.** `verified` means a per-item
 *    record from the benchmark run exists and the model got that row right.
 *    `scored` means a number was measured but there is no right answer to have
 *    -- captioning. `candidate` means nothing recorded it. Painting all three
 *    as "correct" is the one claim this page cannot support.
 *  - **The sample is stated.** These are rows the model got right, drawn from
 *    sets where it does not get everything right, so the real split accuracies
 *    sit at the top. A reader who notices the curation should find it already
 *    admitted rather than discover it.
 *  - **The gold answer is visible before you ask.** Hiding it would make the
 *    reveal feel better and mean less.
 */

const SEGMENT_LABELS: Record<string, string> = {
  rs_vqa: "Single-image VQA",
  change_vqa: "Bi-temporal change",
  grounding: "Referring grounding",
  captioning: "Captioning",
  sar: "SAR / optical–SAR",
};

const STATUS_COPY: Record<
  GalleryStatus,
  { label: string; tone: "pass" | "caution" | "default"; tip: string }
> = {
  verified: {
    label: "VERIFIED",
    tone: "pass",
    tip: "A per-item record from the benchmark run exists, and the model answered this row correctly in it.",
  },
  scored: {
    label: "SCORED",
    tone: "caution",
    tip: "Measured, but a caption is not right or wrong. The ROUGE-L is shown against the run's own blind floor.",
  },
  candidate: {
    label: "UNCHECKED",
    tone: "default",
    tip: "No per-item prediction was ever dumped for this segment, so whether the model gets this row right is genuinely unknown until it is asked.",
  },
};

function ItemCard({ item }: { item: GalleryItem }) {
  const navigate = useNavigate();
  const open = useOpenGalleryItem();

  const ask = () => {
    open.mutate(item.item_id, {
      onSuccess: (result) =>
        navigate(
          `/workspace/${result.bundle_id}?q=${encodeURIComponent(result.question)}`,
        ),
    });
  };

  const status = STATUS_COPY[item.status] ?? STATUS_COPY.candidate;
  const gold = String(item.gold);

  return (
    <Sheet className="flex flex-col gap-2.5 p-3">
      <div className="flex gap-2">
        {item.images.map((image) => (
          <figure key={image.role} className="min-w-0 flex-1">
            <img
              src={resolveUrl(image.preview_url)}
              alt={`${item.segment} ${image.role}`}
              loading="lazy"
              className="aspect-square w-full border border-rule object-cover"
            />
            {item.images.length > 1 ? (
              <figcaption className="t-code-sm mt-1 text-ink-3">
                {image.role}
              </figcaption>
            ) : null}
          </figure>
        ))}
      </div>

      <div className="flex items-start justify-between gap-2">
        <p className="t-doc min-w-0 text-[12.5px] leading-snug text-ink-1">
          {item.question}
        </p>
        <Tip content={status.tip}>
          <span>
            <Tag tone={status.tone}>{status.label}</Tag>
          </span>
        </Tip>
      </div>

      <dl className="t-doc flex flex-col gap-1 text-[11.5px]">
        <div className="flex gap-2">
          <dt className="shrink-0 text-ink-3">Gold</dt>
          <dd className="min-w-0 break-words text-ink-0">
            {gold.length > 150 ? `${gold.slice(0, 150)}…` : gold}
          </dd>
        </div>
        {item.iou !== undefined ? (
          <div className="flex gap-2">
            <dt className="shrink-0 text-ink-3">Benchmark IoU</dt>
            <dd className="t-code-sm text-ink-1">{item.iou.toFixed(3)}</dd>
          </div>
        ) : null}
        {item.rouge_l !== undefined ? (
          <div className="flex gap-2">
            <dt className="shrink-0 text-ink-3">ROUGE-L</dt>
            <dd className="t-code-sm text-ink-1">
              {item.rouge_l.toFixed(3)}
              {item.blind_floor_rouge_l !== undefined ? (
                <span className="text-ink-3">
                  {" "}
                  · floor {item.blind_floor_rouge_l.toFixed(3)}
                </span>
              ) : null}
            </dd>
          </div>
        ) : null}
        {item.served_answer ? (
          <div className="flex gap-2">
            <dt className="shrink-0 text-ink-3">Served</dt>
            <dd className={item.passed ? "text-pass-ink" : "text-signal-ink"}>
              {String(item.served_answer).slice(0, 110)}
            </dd>
          </div>
        ) : null}
      </dl>

      <div className="mt-auto flex items-center justify-between gap-2 border-t border-rule pt-2">
        <Tip
          content={`${item.corpus} · ${item.split} split · Modal workspace ${item.account}`}
        >
          <span className="t-code-sm min-w-0 truncate text-ink-3">
            {item.corpus}
          </span>
        </Tip>
        <Button variant="primary" onClick={ask} disabled={open.isPending}>
          {open.isPending ? "Loading…" : "Ask this"}
        </Button>
      </div>

      {open.isError ? (
        <p className="t-doc text-[11.5px] text-signal-ink">
          Could not load this item: {(open.error as Error)?.message}
        </p>
      ) : null}
    </Sheet>
  );
}

export function GalleryScreen() {
  const { data, isLoading, isError, error } = useGallery();
  const [segment, setSegment] = useState<string>("all");

  const segments = useMemo(() => {
    const names = new Set<string>();
    for (const item of data?.items ?? []) names.add(item.segment);
    return ["all", ...Array.from(names).sort()];
  }, [data]);

  const shown = useMemo(() => {
    const items = data?.items ?? [];
    return segment === "all" ? items : items.filter((i) => i.segment === segment);
  }, [data, segment]);

  return (
    <div className="mx-auto flex w-full max-w-[1180px] flex-col gap-4 px-6 py-6">
      <ZoneHeader code="S8" title="TESTING CORPUS" />

      <Sheet className="flex flex-col gap-2 p-4">
        <p className="t-doc text-[13px] leading-relaxed text-ink-1">
          Every image below comes from the <strong>test or held-out split</strong>{" "}
          of a public benchmark, with the question and the gold answer that
          benchmark published.{" "}
          <strong>The models were not trained on any of it.</strong> Pick one and
          ask it — the answer comes back through the same router, tools and
          adapters as an image you upload yourself.
        </p>
        <p className="t-doc text-[12px] leading-relaxed text-ink-2">
          These are rows the system was measured on, not a random sample of the
          world. On their full splits these segments score{" "}
          <strong>85.06</strong> RSVQA-HR, <strong>83.08</strong> RSVQA-LR,{" "}
          <strong>68.0% AA</strong> CDVQA, <strong>62.7% acc@0.5</strong>{" "}
          VRSBench referring and <strong>74.95%</strong> reBEN radar — so the
          system is wrong on a real fraction of rows like these. Items marked{" "}
          <em>verified</em> are ones it answered correctly in the recorded run;
          items marked <em>unchecked</em> never had a prediction recorded at all.
          The distinction is kept rather than smoothed over.
        </p>
      </Sheet>

      {data?.verification?.per_segment ? (
        <Sheet className="flex flex-col gap-1.5 p-4">
          <p className="t-code-sm text-ink-2">
            LAST VERIFIED AGAINST BUILD{" "}
            <span className="text-ink-0">{data.verification.build ?? "unknown"}</span>
          </p>
          <p className="t-doc text-[12px] leading-relaxed text-ink-2">
            Every item below was replayed through the live API and scored with
            its own benchmark's answer contract — the same formatter that
            produced the published number, not a second implementation. This is
            what fraction of this sample held up:
          </p>
          <ul className="t-code-sm flex flex-wrap gap-x-4 gap-y-1 text-ink-1">
            {Object.entries(data.verification.per_segment).map(([name, row]) => (
              <li key={name}>
                {SEGMENT_LABELS[name] ?? name}{" "}
                <span
                  className={row.pass_rate >= 0.8 ? "text-pass-ink" : "text-caution"}
                >
                  {row.passed}/{row.total}
                </span>
              </li>
            ))}
          </ul>
        </Sheet>
      ) : null}

      {data?.available === false ? (
        <EmptyState code="S8·0" title="No testing corpus on this deployment">
          The gallery manifest has not been uploaded to the volume yet.
        </EmptyState>
      ) : null}

      <div className="flex flex-wrap items-center gap-1.5">
        {segments.map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => setSegment(name)}
            className={`t-code-sm border px-2 py-[5px] transition-colors ${
              segment === name
                ? "border-ink-0 bg-ink-0 text-panel-2"
                : "border-rule bg-panel-sunk text-ink-2 hover:text-ink-0"
            }`}
          >
            {name === "all" ? "ALL" : (SEGMENT_LABELS[name] ?? name)}
            {name !== "all" && data?.counts?.[name] ? (
              <span className="ml-1 opacity-60">{data.counts[name]}</span>
            ) : null}
          </button>
        ))}
        {data?.statuses ? (
          <span className="t-code-sm ml-auto text-ink-3">
            {Object.entries(data.statuses)
              .map(([name, count]) => `${count} ${name}`)
              .join(" · ")}
          </span>
        ) : null}
      </div>

      {isError ? (
        <EmptyState code="S8·E" title="Could not load the testing corpus">
          {(error as Error)?.message ?? "The gallery endpoint did not respond."}
        </EmptyState>
      ) : null}

      {isLoading ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="h-72 w-full" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map((item) => (
            <ItemCard key={item.item_id} item={item} />
          ))}
        </div>
      )}

      {!isLoading && !isError && shown.length === 0 && data?.available !== false ? (
        <EmptyState code="S8·1" title="Nothing in this segment">
          Try another filter.
        </EmptyState>
      ) : null}

      <footer className="t-doc pb-6 text-[11.5px] leading-relaxed text-ink-3">
        Corpora: RSVQA-HR and RSVQA-LR (test), CDVQA Val over SECOND, VRSBench
        referring and captioning (EVAL), reBEN held-out. Licence and provenance
        for each are recorded in CREDITS.md. Scores were produced with the
        in-repo comparator rather than the official scorers, and are quoted for
        run-to-run comparison.
      </footer>
    </div>
  );
}
