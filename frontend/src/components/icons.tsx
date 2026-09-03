import type { SVGProps } from "react";

/**
 * Authored icon set.
 *
 * One 16-unit grid, 1.5 stroke, butt caps, no fills except where a mark is
 * genuinely solid. Drawn in the same technical-drawing hand as the rest of
 * the panel — an equipment legend, not a general-purpose UI library.
 */

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 16, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="butt"
      strokeLinejoin="miter"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconVisible = (p: IconProps) => (
  <Icon {...p}>
    <path d="M1 8s2.6-4 7-4 7 4 7 4-2.6 4-7 4-7-4-7-4Z" />
    <circle cx="8" cy="8" r="1.9" />
  </Icon>
);

export const IconHidden = (p: IconProps) => (
  <Icon {...p}>
    <path d="M2.2 5.4C1.4 6.4 1 8 1 8s2.6 4 7 4c1.2 0 2.2-.3 3.1-.7" />
    <path d="M6.2 4.2A7.9 7.9 0 0 1 8 4c4.4 0 7 4 7 4s-.7 1.1-2 2.2" />
    <path d="M2 2l12 12" />
  </Icon>
);

export const IconDownload = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 1.5v8.5" />
    <path d="M4.5 7 8 10.5 11.5 7" />
    <path d="M2 13.5h12" />
  </Icon>
);

export const IconCopy = (p: IconProps) => (
  <Icon {...p}>
    <rect x="5.5" y="5.5" width="9" height="9" />
    <path d="M10.5 5.5v-4h-9v9h4" />
  </Icon>
);

export const IconCheck = (p: IconProps) => (
  <Icon {...p}>
    <path d="M2.5 8.5 6 12l7.5-8" />
  </Icon>
);

export const IconCross = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />
  </Icon>
);

export const IconWarning = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 1.5 15 14H1L8 1.5Z" />
    <path d="M8 6v3.6" />
    <path d="M8 11.4v.9" />
  </Icon>
);

export const IconInfo = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="6.5" />
    <path d="M8 7.2v4" />
    <path d="M8 4.6v.9" />
  </Icon>
);

export const IconArrowRight = (p: IconProps) => (
  <Icon {...p}>
    <path d="M1.5 8h12" />
    <path d="M10 4.5 13.5 8 10 11.5" />
  </Icon>
);

export const IconChevron = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 3.5 9.5 8 5 12.5" />
  </Icon>
);

export const IconTray = (p: IconProps) => (
  <Icon {...p}>
    <path d="M1.5 9.5V13a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1V9.5" />
    <path d="M1.5 9.5h3.2l1 1.8h4.6l1-1.8h3.2" />
    <path d="M8 1.5v5.8" />
    <path d="M5.6 5 8 7.4 10.4 5" />
  </Icon>
);

export const IconLayers = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 1.5 15 5 8 8.5 1 5l7-3.5Z" />
    <path d="M1 8.5 8 12l7-3.5" />
    <path d="M1 11.5 8 15l7-3.5" />
  </Icon>
);

export const IconGrid = (p: IconProps) => (
  <Icon {...p}>
    <rect x="1.5" y="1.5" width="13" height="13" />
    <path d="M1.5 5.8h13M1.5 10.2h13M5.8 1.5v13M10.2 1.5v13" />
  </Icon>
);

export const IconClock = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="6.5" />
    <path d="M8 4.2V8l2.6 1.9" />
  </Icon>
);

export const IconAntenna = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 14.5V8.2" />
    <path d="M4.5 14.5h7" />
    <path d="M3.2 8.2 8 2l4.8 6.2" />
    <path d="M5.4 8.2h5.2" />
  </Icon>
);

export const IconChip = (p: IconProps) => (
  <Icon {...p}>
    <rect x="4" y="4" width="8" height="8" />
    <path d="M6.4 1.5v2.5M9.6 1.5v2.5M6.4 12v2.5M9.6 12v2.5" />
    <path d="M1.5 6.4H4M1.5 9.6H4M12 6.4h2.5M12 9.6h2.5" />
  </Icon>
);

export const IconRefresh = (p: IconProps) => (
  <Icon {...p}>
    <path d="M13.5 8a5.5 5.5 0 1 1-1.7-3.9" />
    <path d="M13.6 1.8v3.1h-3.1" />
  </Icon>
);

export const IconPlus = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 2.5v11M2.5 8h11" />
  </Icon>
);

export const IconPrint = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4.5 6V1.5h7V6" />
    <rect x="1.5" y="6" width="13" height="5.5" />
    <rect x="4.5" y="9.5" width="7" height="5" />
  </Icon>
);

export const IconFile = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 1.5h6L13 5.5v9H3v-13Z" />
    <path d="M9 1.5v4h4" />
  </Icon>
);

export const IconStop = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3.5" y="3.5" width="9" height="9" />
  </Icon>
);

export const IconTarget = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="5.5" />
    <path d="M8 0.8v3.1M8 12.1v3.1M0.8 8h3.1M12.1 8h3.1" />
    <circle cx="8" cy="8" r="1.2" />
  </Icon>
);

export const IconSwipe = (p: IconProps) => (
  <Icon {...p}>
    <rect x="1.5" y="2.5" width="13" height="11" />
    <path d="M8 2.5v11" />
    <path d="M5.4 6.4 3.4 8l2 1.6" />
    <path d="M10.6 6.4 12.6 8l-2 1.6" />
  </Icon>
);

/**
 * Modality marks. Colour is never the only carrier of a verdict here — the
 * optical mark is a lens, the SAR mark is a beam, and both are drawn.
 */
export const IconOptical = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="6.2" />
    <circle cx="8" cy="8" r="2.6" />
    <path d="M3.6 3.6 6.2 6.2M12.4 3.6 9.8 6.2M3.6 12.4l2.6-2.6M12.4 12.4 9.8 9.8" />
  </Icon>
);

export const IconSar = (p: IconProps) => (
  <Icon {...p}>
    <path d="M2 13.5h12" />
    <path d="M3.4 10.6c1.4-1.6 3-2.4 4.6-2.4s3.2.8 4.6 2.4" />
    <path d="M4.8 7.4C5.8 6.3 6.9 5.7 8 5.7s2.2.6 3.2 1.7" />
    <path d="M6.3 4.2c.6-.6 1.1-.9 1.7-.9s1.1.3 1.7.9" />
  </Icon>
);
