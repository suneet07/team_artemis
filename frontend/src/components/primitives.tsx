import {
  forwardRef,
  type ButtonHTMLAttributes,
  type HTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cn } from "@/lib/cn";

/* ── zone chrome ─────────────────────────────────────────────────────── */

/**
 * Every pane on this instrument carries a stencilled zone code, the way a
 * real panel labels its sections. It is not decoration: the code is what the
 * user says out loud when pointing at a region during a demo.
 */
export function ZoneHeader({
  code,
  title,
  actions,
  className,
}: {
  code: string;
  title: string;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn(
        "flex items-center gap-3 border-b border-rule bg-panel-1 px-3 py-2",
        className,
      )}
    >
      <span className="t-code-sm shrink-0 border border-rule bg-panel-sunk px-1.5 py-[3px] text-ink-2">
        {code}
      </span>
      <h2 className="t-code min-w-0 flex-1 truncate text-ink-1">{title}</h2>
      {actions ? (
        <div className="flex shrink-0 items-center gap-1">{actions}</div>
      ) : null}
    </header>
  );
}

/** A sheet of the operations document, resting on the panel. */
export const Sheet = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  function Sheet({ className, ...rest }, ref) {
    return <div ref={ref} className={cn("m-sheet", className)} {...rest} />;
  },
);

/* ── labelled fields ─────────────────────────────────────────────────── */

/**
 * The marginalia row: a stencilled field name and its measured value. This
 * is the atom the whole interface is built from — nothing is asserted here
 * without the label that says what it is.
 */
export function Field({
  label,
  value,
  title,
  mono = true,
  tone = "default",
  className,
}: {
  label: string;
  value: ReactNode;
  title?: string;
  mono?: boolean;
  tone?: "default" | "signal" | "pass" | "caution" | "muted";
  className?: string;
}) {
  const toneClass = {
    default: "text-ink-0",
    signal: "text-signal-ink",
    pass: "text-pass-ink",
    caution: "text-caution",
    muted: "text-ink-2",
  }[tone];

  return (
    <div className={cn("flex min-w-0 flex-col gap-[3px]", className)}>
      <span className="t-code-sm text-ink-3">{label}</span>
      <span
        title={title}
        className={cn(
          // Values wrap rather than truncate: a CRS or a tool name clipped to
          // "EPSG:326…" is the one thing a reader came to check.
          "min-w-0 break-words text-[12.5px] leading-[1.35]",
          mono ? "t-data" : "font-sans",
          toneClass,
        )}
      >
        {value}
      </span>
    </div>
  );
}

/** A horizontal key/value line for dense record blocks. */
export function FieldRow({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-baseline gap-3 border-b border-rule-hair py-[5px] last:border-b-0",
        className,
      )}
    >
      <span className="t-code-sm w-[128px] shrink-0 pt-[2px] text-ink-3">
        {label}
      </span>
      <div className="min-w-0 flex-1 text-[12.5px] leading-[1.45]">
        {children}
      </div>
    </div>
  );
}

/* ── status ──────────────────────────────────────────────────────────── */

export type Lamp = "pass" | "fail" | "caution" | "idle" | "active" | "skipped";

/**
 * A lamp with a drawn shape, not just a colour. Circle = nominal, square =
 * fault, triangle = caution, dash = skipped. Accessible without hue, and it
 * matches how a real panel legend distinguishes its indicators.
 */
export function StatusLamp({ state, size = 9 }: { state: Lamp; size?: number }) {
  const colour = {
    pass: "var(--color-pass)",
    fail: "var(--color-signal)",
    caution: "var(--color-caution)",
    idle: "var(--color-rule-heavy)",
    active: "var(--color-signal)",
    skipped: "var(--color-ink-3)",
  }[state];

  const shape =
    state === "fail" ? (
      <rect x="1.5" y="1.5" width="9" height="9" fill={colour} />
    ) : state === "caution" ? (
      <path d="M6 1 11 10.5H1L6 1Z" fill={colour} />
    ) : state === "skipped" ? (
      <rect x="1" y="5" width="10" height="2" fill={colour} />
    ) : state === "idle" ? (
      <circle cx="6" cy="6" r="3.6" fill="none" stroke={colour} strokeWidth="1.6" />
    ) : (
      <circle cx="6" cy="6" r="4.2" fill={colour} />
    );

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 12 12"
      aria-hidden="true"
      className={cn("shrink-0", state === "active" && "pulse-mark")}
    >
      {shape}
    </svg>
  );
}

export function Tag({
  children,
  tone = "default",
  title,
  className,
}: {
  children: ReactNode;
  tone?: "default" | "signal" | "pass" | "caution" | "advisory" | "solid";
  title?: string;
  className?: string;
}) {
  const tones = {
    default: "border-rule bg-panel-sunk text-ink-1",
    signal: "border-signal/45 bg-signal-wash text-signal-ink",
    pass: "border-pass/40 bg-pass-wash text-pass-ink",
    caution: "border-caution/40 bg-caution-wash text-caution",
    advisory: "border-advisory/35 bg-advisory-wash text-advisory",
    solid: "border-ink-0 bg-ink-0 text-panel-2",
  }[tone];

  return (
    <span
      title={title}
      className={cn(
        "t-code-sm inline-flex shrink-0 items-center gap-1 border px-1.5 py-[3px]",
        tones,
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ── controls ────────────────────────────────────────────────────────── */

type ButtonVariant = "primary" | "panel" | "ghost" | "danger";

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: ButtonVariant;
    icon?: ReactNode;
  }
>(function Button(
  { className, variant = "panel", icon, children, ...rest },
  ref,
) {
  const variants: Record<ButtonVariant, string> = {
    // The primary control is a keyed switch: dark, engraved, unmistakable.
    primary:
      "border-ink-0 bg-ink-0 text-panel-2 shadow-[inset_0_1px_0_rgb(255_255_255/0.16)] hover:bg-[#20282a] active:translate-y-px disabled:border-rule disabled:bg-panel-sunk disabled:text-ink-3 disabled:shadow-none",
    // The workhorse: a milled key on the panel.
    panel:
      "border-rule-heavy bg-plate-1 text-ink-0 shadow-[inset_0_1px_0_rgb(255_255_255/0.65),inset_0_-1px_0_rgb(0_0_0/0.08)] hover:bg-plate-0 active:translate-y-px active:shadow-[inset_0_1px_2px_rgb(0_0_0/0.16)] disabled:border-rule disabled:bg-panel-sunk disabled:text-ink-3 disabled:shadow-none",
    ghost:
      "border-transparent bg-transparent text-ink-1 hover:border-rule hover:bg-panel-1 active:translate-y-px disabled:text-ink-3",
    danger:
      "border-signal bg-signal text-white shadow-[inset_0_1px_0_rgb(255_255_255/0.2)] hover:bg-[#c8400f] active:translate-y-px disabled:border-rule disabled:bg-panel-sunk disabled:text-ink-3 disabled:shadow-none",
  };

  return (
    <button
      ref={ref}
      className={cn(
        "t-code inline-flex items-center justify-center gap-1.5 border px-2.5 py-[7px] transition-[background-color,box-shadow,transform] duration-100 disabled:cursor-not-allowed",
        variants[variant],
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
});

export const IconButton = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { label: string }
>(function IconButton({ className, label, children, ...rest }, ref) {
  return (
    <button
      ref={ref}
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "inline-flex h-[26px] w-[26px] items-center justify-center border border-transparent text-ink-2 transition-colors hover:border-rule hover:bg-panel-1 hover:text-ink-0 disabled:cursor-not-allowed disabled:text-ink-3 disabled:hover:border-transparent disabled:hover:bg-transparent",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
});

export const TextInput = forwardRef<
  HTMLInputElement,
  InputHTMLAttributes<HTMLInputElement>
>(function TextInput({ className, ...rest }, ref) {
  return (
    <input
      ref={ref}
      className={cn(
        "m-sunk t-data w-full px-2 py-[6px] text-[12.5px] text-ink-0 placeholder:text-ink-3 disabled:cursor-not-allowed disabled:text-ink-3",
        className,
      )}
      {...rest}
    />
  );
});

export const Select = forwardRef<
  HTMLSelectElement,
  SelectHTMLAttributes<HTMLSelectElement>
>(function Select({ className, children, ...rest }, ref) {
  return (
    <div className="relative">
      <select
        ref={ref}
        className={cn(
          "m-sunk t-code w-full appearance-none py-[6px] pl-2 pr-7 text-ink-0 disabled:cursor-not-allowed disabled:text-ink-3",
          className,
        )}
        {...rest}
      >
        {children}
      </select>
      <svg
        width="9"
        height="6"
        viewBox="0 0 9 6"
        aria-hidden="true"
        className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-ink-2"
      >
        <path d="M0.5 0.5 4.5 4.5 8.5 0.5" stroke="currentColor" fill="none" strokeWidth="1.5" />
      </svg>
    </div>
  );
});

/* ── explanation ─────────────────────────────────────────────────────── */

export function Tip({
  content,
  children,
}: {
  content: ReactNode;
  children: ReactNode;
}) {
  return (
    <TooltipPrimitive.Root delayDuration={140}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          side="top"
          sideOffset={5}
          collisionPadding={10}
          className="z-50 max-w-[280px] border border-ink-0 bg-ink-0 px-2 py-1.5 text-[11.5px] leading-[1.45] text-panel-2 shadow-[0_2px_10px_-4px_rgb(19_24_25/0.5)]"
        >
          {content}
          <TooltipPrimitive.Arrow className="fill-ink-0" width={9} height={4} />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}

export const TipProvider = TooltipPrimitive.Provider;

/* ── notices ─────────────────────────────────────────────────────────── */

/**
 * Warnings render as an always-visible list, never collapsed by default
 * (§4.1). They are a product of the compatibility check, so they read as
 * findings rather than as errors.
 */
export function WarningList({
  warnings,
  className,
  title = "Compatibility warnings",
}: {
  warnings: string[];
  className?: string;
  title?: string;
}) {
  if (warnings.length === 0) return null;
  return (
    <div
      className={cn(
        "border border-caution/35 bg-caution-wash px-2.5 py-2",
        className,
      )}
    >
      <div className="t-code-sm mb-1.5 flex items-center gap-1.5 text-caution">
        <StatusLamp state="caution" />
        {title} · {warnings.length}
      </div>
      <ul className="flex flex-col gap-1">
        {warnings.map((warning) => (
          <li
            key={warning}
            className="flex gap-1.5 text-[12px] leading-[1.45] text-ink-1"
          >
            <span aria-hidden="true" className="text-caution">
              —
            </span>
            <span>{warning}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Loading placeholder that keeps the record's shape — never a spinner. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "relative overflow-hidden bg-panel-sunk carriage",
        className,
      )}
      aria-hidden="true"
    />
  );
}

export function EmptyState({
  code,
  title,
  children,
  action,
}: {
  code: string;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-start gap-2.5 border border-dashed border-rule bg-panel-1 px-4 py-5">
      <span className="t-code-sm border border-rule bg-panel-sunk px-1.5 py-[3px] text-ink-3">
        {code}
      </span>
      <p className="t-plate text-[13px] text-ink-1">{title}</p>
      {children ? (
        <div className="t-doc text-[12.5px] text-ink-2">{children}</div>
      ) : null}
      {action}
    </div>
  );
}
