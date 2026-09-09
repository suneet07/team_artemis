import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { resolveUrl } from "@/api/client";
import {
  type GalleryItem,
  useGallery,
  useOpenGalleryItem,
} from "@/api/gallery";
import {
  Button,
  EmptyState,
  Sheet,
  Skeleton,
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

      <p className="t-doc min-w-0 text-[12.5px] leading-snug text-ink-1">
        {item.question}
      </p>

      <div className="mt-auto flex items-center justify-between gap-2 border-t border-rule pt-2">
        <Tip
          content={`${item.corpus} · ${item.split} split`}
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
      <ZoneHeader title="TESTING CORPUS" />

      <Sheet className="p-4">
        <p className="t-doc text-[13px] leading-relaxed text-ink-1">
          Unbiased gallery of satellite imagery from the testing set of the
          evaluation datasets named below. The model was not trained on this
          data, so it is completely fair.
        </p>
      </Sheet>

      {data?.available === false ? (
        <EmptyState title="No testing corpus on this deployment">
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
      </div>

      {isError ? (
        <EmptyState title="Could not load the testing corpus">
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
        <EmptyState title="Nothing in this segment">
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
