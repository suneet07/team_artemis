import { IconSwipe } from "@/components/icons";
import { useWorkspace } from "@/store/workspace";
import { cn } from "@/lib/cn";

const OPTIONS = [
  { value: "single", label: "SINGLE", icon: false },
  { value: "swipe", label: "SWIPE", icon: true },
  { value: "side", label: "SIDE BY SIDE", icon: false },
] as const;

/**
 * How a two-scene bundle is laid out in the imagery window.
 *
 * It sits in that window's header rather than among the layer controls: it
 * changes the window itself, not what is drawn in it. Both comparisons hold
 * the two scenes at the same position — one pan and zoom drives both.
 */
export function CompareSwitch() {
  const compareMode = useWorkspace((s) => s.compareMode);
  const setCompareMode = useWorkspace((s) => s.setCompareMode);

  return (
    <div
      role="group"
      aria-label="How the two scenes are shown"
      className="flex items-center"
    >
      <span className="t-code-sm mr-2 hidden text-ink-3 xl:inline">COMPARE</span>
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => setCompareMode(option.value)}
          aria-pressed={compareMode === option.value}
          className={cn(
            "t-code-sm -ml-px flex items-center gap-1.5 whitespace-nowrap border px-2 py-[4px] transition-colors first-of-type:ml-0",
            compareMode === option.value
              ? "z-[1] border-signal bg-signal text-white"
              : "border-rule bg-panel-2 text-ink-1 hover:border-rule-heavy",
          )}
        >
          {option.icon ? <IconSwipe size={12} /> : null}
          {option.label}
        </button>
      ))}
    </div>
  );
}
