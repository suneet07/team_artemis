import { useNavigate } from "react-router-dom";
import type { Bundle, RefusalCategory, RefusalWithRemedy } from "@contracts/types";
import { useTasks } from "@/api/meta";
import { Button, Tag } from "@/components/primitives";
import { IconArrowRight, IconInfo, IconTray } from "@/components/icons";

/**
 * A refusal is an answer (§7.1).
 *
 * Compatibility checking is a named deliverable, so when the system declines
 * a question it has to look like a considered result: an advisory panel, the
 * category named, the backend's reason verbatim, and a control that performs
 * the remedy rather than re-running the same query. Never a red toast, never
 * a bare retry.
 */

const CATEGORY_LABEL: Record<RefusalCategory, string> = {
  missing_input: "Missing input",
  validator: "Validator",
  modality_limitation: "Modality limitation",
  parameter_gate: "Parameter rejected",
  unsupported_class: "Unsupported class",
};

const CATEGORY_NOTE: Record<RefusalCategory, string> = {
  missing_input:
    "The task the router selected needs an input this bundle does not contain.",
  validator:
    "The request did not pass validation against what the prepared inputs can support.",
  modality_limitation:
    "The sensor that produced this scene cannot measure what the question asks for.",
  parameter_gate:
    "One or more parameters fell outside the tool manifest, so the tool was never invoked.",
  unsupported_class:
    "The target class is not one this system is trained or calibrated to identify.",
};

export function RefusalCard({
  refusal,
  bundle,
  onAskSuggested,
}: {
  refusal: RefusalWithRemedy;
  bundle: Bundle | undefined;
  onAskSuggested: (question: string) => void;
}) {
  const navigate = useNavigate();
  const { data: tasks } = useTasks();

  // Suggestions come from the backend's remedy when it offered one, and
  // otherwise from the bundle's own supported task list — never from a
  // routing rule reimplemented in the client.
  const suggestions =
    refusal.remedy?.suggested_questions ??
    (bundle?.supported_tasks ?? [])
      .map((task) => tasks?.byTask.get(task)?.label)
      .filter((label): label is string => Boolean(label))
      .slice(0, 3);

  const remedyAction = refusal.remedy?.action;
  const opensUpload =
    remedyAction === "add_second_image" ||
    remedyAction === "add_optical" ||
    remedyAction === "add_sar" ||
    remedyAction === "reupload";

  return (
    <div className="border border-advisory/35 bg-advisory-wash">
      <div className="flex items-center gap-2 border-b border-advisory/25 px-3 py-2">
        <IconInfo size={14} className="shrink-0 text-advisory" />
        <span className="t-code text-advisory">
          {CATEGORY_LABEL[refusal.category]}
        </span>
        <Tag tone="advisory" className="ml-auto">
          NOT AN ERROR
        </Tag>
      </div>

      <div className="flex flex-col gap-3 px-3 py-3">
        <p className="t-doc text-[13px] text-ink-0">{refusal.reason}</p>

        <p className="text-[11.5px] leading-[1.5] text-ink-2">
          {CATEGORY_NOTE[refusal.category]} The full reasoning is in the trace,
          including the routing notes that led here.
        </p>

        {refusal.remedy && remedyAction !== "none" ? (
          <div className="flex flex-wrap items-center gap-2 border-t border-advisory/25 pt-3">
            {opensUpload ? (
              <Button
                variant="primary"
                icon={<IconTray size={13} />}
                onClick={() =>
                  navigate("/upload", {
                    state: {
                      // Pre-select the role the remedy asks for, so the fix
                      // is one action rather than a re-derivation.
                      role:
                        remedyAction === "add_sar"
                          ? "sar"
                          : remedyAction === "add_optical"
                            ? "optical"
                            : "t2",
                      pairType:
                        remedyAction === "add_second_image"
                          ? "bitemporal"
                          : "crossmodal",
                      keepBundle: bundle?.bundle_id,
                    },
                  })
                }
              >
                {refusal.remedy.label}
              </Button>
            ) : null}
          </div>
        ) : null}

        {suggestions && suggestions.length > 0 ? (
          <div className="border-t border-advisory/25 pt-3">
            <p className="t-code-sm mb-2 text-advisory">
              QUESTIONS THIS BUNDLE CAN ANSWER
            </p>
            <ul className="flex flex-col gap-1.5">
              {suggestions.map((suggestion) => (
                <li key={suggestion}>
                  <button
                    type="button"
                    onClick={() => onAskSuggested(suggestion)}
                    className="group flex w-full items-center gap-2 border border-advisory/30 bg-panel-2 px-2 py-1.5 text-left text-[12.5px] leading-[1.4] text-ink-0 transition-colors hover:border-advisory hover:bg-panel-1"
                  >
                    <IconArrowRight
                      size={12}
                      className="shrink-0 text-advisory"
                    />
                    {suggestion}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  );
}
