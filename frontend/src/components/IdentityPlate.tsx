import { Link, useLocation } from "react-router-dom";
import { useHealth } from "@/api/meta";
import { USE_MOCKS } from "@/api/client";
import { useWorkspace } from "@/store/workspace";
import { cn } from "@/lib/cn";
import { IconChip, IconGrid } from "./icons";
import { StatusLamp, Tip } from "./primitives";

/**
 * The equipment plate.
 *
 * Brushed anodised aluminium with engraved lettering, carrying the same
 * things a real hardware plate carries: what this unit is, who it belongs
 * to, and its live operating state. Health lives in the chrome rather than
 * buried in a settings screen — at the venue the system will genuinely be
 * degraded, and that has to look deliberate.
 */

const NAV = [
  { to: "/", label: "Sessions", code: "S1" },
  { to: "/upload", label: "Receiving", code: "S2" },
  { to: "/system", label: "System", code: "S7" },
];

function PlateRivet({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "absolute h-[7px] w-[7px] rounded-full",
        "bg-[radial-gradient(circle_at_35%_30%,#e7eae1_0%,#9aa094_55%,#6f766b_100%)]",
        "shadow-[inset_0_0_0_0.5px_rgb(0_0_0/0.25),0_1px_0_rgb(255_255_255/0.45)]",
        className,
      )}
    />
  );
}

export function IdentityPlate() {
  const { data: health, isError } = useHealth();
  const armature = useWorkspace((s) => s.armature);
  const toggleArmature = useWorkspace((s) => s.toggleArmature);
  const location = useLocation();

  const degraded =
    isError ||
    !health ||
    health.status !== "ok" ||
    !health.gpu ||
    health.serving !== "vllm";

  return (
    <header className="m-plate relative z-30 border-x-0 border-t-0 print:hidden">
      <PlateRivet className="left-2 top-2" />
      <PlateRivet className="right-2 top-2" />
      <PlateRivet className="bottom-2 left-2" />
      <PlateRivet className="bottom-2 right-2" />

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 px-7 py-2.5">
        {/* identity */}
        <Link
          to="/"
          className="group flex items-baseline gap-2.5 focus-visible:outline-offset-4"
        >
          <span className="m-engraved t-plate text-[17px] leading-none">
            SatQuery
            <span className="ml-[3px] font-normal opacity-70">AI</span>
          </span>
          <span className="m-engraved t-code-sm hidden opacity-80 sm:inline">
            SIH26167 · ISRO / SAC
          </span>
        </Link>

        <span aria-hidden="true" className="hidden h-6 w-px bg-plate-edge/60 md:block" />

        {/* zones */}
        <nav aria-label="Screens" className="flex items-center gap-1">
          {NAV.map((item) => {
            const active =
              item.to === "/"
                ? location.pathname === "/"
                : location.pathname.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "t-code-sm flex items-center gap-1.5 border px-2 py-[5px] transition-colors",
                  active
                    ? "border-ink-0 bg-ink-0 text-panel-2"
                    : "m-engraved border-plate-edge/50 hover:border-plate-edge hover:bg-plate-1",
                )}
              >
                <span className={active ? "text-panel-2/60" : "opacity-60"}>
                  {item.code}
                </span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex-1" />

        {/* live state, as equipment codes */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {USE_MOCKS ? (
            <Tip content="No backend is attached. Every response comes from the MSW mock layer against the frozen contract, and the imagery is generated, not observed.">
              <span className="t-code-sm m-hazard-soft m-engraved border border-signal/50 px-1.5 py-[3px]">
                MOCK DATA
              </span>
            </Tip>
          ) : null}

          <PlateReadout
            label="SERVING"
            value={health?.serving ?? "—"}
            lamp={
              !health ? "idle" : health.serving === "vllm" ? "pass" : "caution"
            }
            tip={
              health?.serving === "vllm"
                ? "vLLM is serving the adapters on GPU."
                : "Running a local quantised model. Deterministic tools are unaffected; generated answers are slower and weaker."
            }
          />

          <PlateReadout
            label="GPU"
            value={health ? (health.gpu ? "present" : "absent") : "—"}
            lamp={!health ? "idle" : health.gpu ? "pass" : "caution"}
            tip={
              health?.gpu
                ? "A GPU is available to the serving layer."
                : "No GPU. Query latency will exceed the 20 s budget; the deterministic pipeline still runs."
            }
          />

          <PlateReadout
            label="LINK"
            value={health?.offline_mode ? "offline" : "online"}
            lamp={health?.offline_mode ? "caution" : "pass"}
            tip={
              health?.offline_mode
                ? "Offline mode: nothing leaves this machine. This is the intended venue state."
                : "The API reports network access. The demo path still makes no external requests."
            }
          />

          <Tip
            content={
              health
                ? `${health.adapters_loaded.length} adapter(s): ${health.adapters_loaded.join(", ") || "none"}`
                : "Adapter inventory unavailable."
            }
          >
            <span className="m-engraved t-code-sm flex items-center gap-1.5">
              <IconChip size={12} className="opacity-70" />
              {health?.adapters_loaded.length ?? "—"} ADAPTERS
            </span>
          </Tip>

          <Tip content={`Trace schema v${health?.trace_schema_version ?? "?"} · build ${health?.version ?? "?"}`}>
            <span className="m-engraved t-code-sm">
              TRACE v{health?.trace_schema_version ?? "?"}
            </span>
          </Tip>

          <button
            type="button"
            onClick={toggleArmature}
            aria-pressed={armature}
            title="Show the layout grid and tiling partition used to build this screen"
            className={cn(
              "t-code-sm flex items-center gap-1.5 border px-1.5 py-[4px] transition-colors",
              armature
                ? "border-signal bg-signal text-white"
                : "m-engraved border-plate-edge/50 hover:border-plate-edge hover:bg-plate-1",
            )}
          >
            <IconGrid size={11} />
            REGISTRATION
          </button>
        </div>
      </div>

      {degraded ? (
        <div className="flex items-center gap-2 border-t border-plate-edge/60 bg-caution-wash px-7 py-1.5">
          <StatusLamp state="caution" />
          <span className="t-code-sm text-caution">
            DEGRADED MODE
          </span>
          <span className="text-[12px] leading-tight text-ink-1">
            {isError || !health
              ? "The API health endpoint is unreachable. Deterministic tools and cached bundles still work; generated answers do not."
              : `Serving ${health.serving}${health.gpu ? "" : " without a GPU"}. Query latency will exceed the 20 s budget — the measured value on screen is the real one.`}
          </span>
        </div>
      ) : null}
    </header>
  );
}

function PlateReadout({
  label,
  value,
  lamp,
  tip,
}: {
  label: string;
  value: string;
  lamp: "pass" | "caution" | "idle";
  tip: string;
}) {
  return (
    <Tip content={tip}>
      <span className="flex items-center gap-1.5">
        <StatusLamp state={lamp} />
        <span className="m-engraved t-code-sm">
          {label}
          <span className="ml-1 opacity-65">{value}</span>
        </span>
      </span>
    </Tip>
  );
}
