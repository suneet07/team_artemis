# Design

Visual system for the SatQuery AI console. Written from the built frontend in
`frontend/src`, not ahead of it. `PRODUCT.md` owns product truth; this file
owns durable visual decisions.

---

## 1. The world

**An ISRO mission-operations document under a spacecraft equipment plate.**

The audience is a judge at a folding table in a bright, fluorescently lit
convention hall, reading a laptop at an angle for eight minutes, with no
internet. The surface has to look like *operated equipment* — a thing with a
serial number and a maintenance record — rather than a web app about
satellites.

Two consequences follow, and neither is negotiable:

- **The panel is light.** A dark UI mirrors a bright hall. The one dark field
  is the imagery window, set into the panel behind glass, exactly as it is on
  real ground-segment hardware.
- **The interface is achromatic.** Saturated colour appears only where it
  means something: an evidence layer whose colour the backend chose, a hazard
  state, a verdict. The text field never competes with the map.

What this refuses: the dark "AI dashboard" of glass cards and one neon accent,
and its pastel-SaaS opposite. Neither would survive the room.

---

## 2. Colour

Tokens live in `frontend/src/styles/theme.css` under `@theme`. Strategy is
**restrained**: neutrals plus one accent.

| Role | Token | Value | Use |
|---|---|---|---|
| Panel ground | `--color-panel-0` | `#ecefe9` | The enclosure. Page background. |
| Panel raised | `--color-panel-1` / `-2` | `#f5f7f2` / `#fbfcf9` | Pane grounds, sheets. |
| Panel recess | `--color-panel-sunk` | `#e2e6dd` | Milled wells: inputs, drop bay. |
| Anodised plate | `--color-plate-0…2`, `-edge` | `#c3c8bd`…`#8f968a` | Identity chrome, key caps. |
| Rules | `--color-rule-hair`, `-rule`, `-heavy` | `#cdd1c6`, `#b2b8aa`, `#7e857a` | Three weights, used as three weights. |
| Ink | `--color-ink-0…3` | `#131819`…`#7d8681` | Silkscreen black through label grey. |
| Window | `--color-window-0…2`, `-ink`, `-ink-2` | `#0a0f11`… | The imagery instrument and its own chrome. |
| Signal | `--color-signal`, `-ink`, `-wash` | `#dc4a1e`, `#ab3512`, `#fbe9e3` | Stencil vermilion. |
| Pass | `--color-pass`, `-ink`, `-wash` | `#2f6b4a`… | Verified, within manifest, coregistered. |
| Caution | `--color-caution`, `-wash` | `#8a6410`, `#f6efdc` | Warnings that are not failures. |
| Advisory | `--color-advisory`, `-wash` | `#1f5670`, `#e5eef2` | **Refusals.** Informational, never red. |

**Signal is stencil vermilion, not a taste choice.** It is the hazard and
equipment-stencil colour of the world, and it sits as far as possible from
every evidence-layer colour the backend sends (blues, cyans, greens), so a
warning and a mask can never be confused for one another.

`--color-ink-2` (`#5a6360`) is the lightest ink permitted for body text on
`panel-0`; anything lighter fails 4.5:1 and belongs to `ink-3` label duty only.

### Evidence colour is the backend's

Mask colours come from `AssetRef.colour` and are baked into the tiles the API
serves. The legend swatch reads the same field, so the swatch and the pixels
cannot drift. Colour never carries a verdict alone — see §5.

---

## 3. Type

One family across its full width axis, plus one mono for measurement.

- **Archivo Variable** (`wght` 100–900, `wdth` 62–125), self-hosted via
  `@fontsource-variable/archivo`.
- **Azeret Mono Variable**, self-hosted.

No CDN. The venue is offline.

| Role | Class | Setting |
|---|---|---|
| Plate lettering | `.t-plate` | `wdth` 116, `wght` 620, uppercase, `+0.06em`. Identity and headings only. |
| Equipment code | `.t-code` | `wdth` 79, `wght` 640, uppercase, `+0.11em`, 10px. The label voice — every zone, field and control. |
| Small code | `.t-code-sm` | As above at 9px, `+0.12em`. |
| Measurement | `.t-data` / `.t-data-strong` | Azeret Mono, `tnum` + `zero`, `-0.02em`. |
| Document | `.t-doc` | Archivo `wdth` 100, 1.6, max 68ch. |

**Monospace here is not costume.** `.t-data` is applied only to numbers,
parameters, identifiers, coordinates and file names — things that are
measurements. Prose never takes it.

The width axis is the system's signature: the same family reads as an engraved
nameplate at 116 and as a dense equipment stencil at 79. Reaching for a second
family would dilute that.

---

## 4. Materials

CSS classes in `theme.css`, each modelling a real surface:

- `.m-plate` — brushed anodised aluminium: vertical brush hairlines, a
  top-to-bottom light fall, and an inset highlight. Identity chrome only.
- `.m-engraved` — lettering cut into the plate; light catches the lower lip.
- `.m-sheet` — a sheet of the operations document resting on the panel. Offset
  shadow with a soft blur, never a zero-offset halo.
- `.m-sunk` — a milled recess. Inputs, wells, the receiving bay.
- `.m-window` — the imagery instrument, set behind glass. Inner shadow plus a
  highlight on the lower lip.
- `.m-hazard` / `.m-hazard-soft` — diagonal striping. **Reserved** for states
  that genuinely warn: degraded serving, rejected parameters, an uncalibrated
  confidence badge. Never decorative.

Rivets appear once, on the identity plate. A second use would make them a
motif rather than a fastening.

---

## 5. State without relying on colour

`StatusLamp` draws a **shape**, not just a hue:

| State | Shape |
|---|---|
| pass / active | filled circle |
| idle | open circle |
| fail | filled square |
| caution | filled triangle |
| skipped | dash |

Every verdict, agreement result and parameter check pairs its colour with a
lamp shape and a word. A colour-blind judge reads the same thing.

---

## 6. Composition

- **The map is the dominant field.** In the workspace the panels are
  subordinate instruments around it: `288px | 1fr | 408px` at `lg`, stacking
  map-first below that.
- **Every pane carries a stencilled zone code** — `S4·C`, `S1·A`, `S7·E`. Not
  decoration: it is what a presenter says out loud while pointing.
- **Records, not cards.** A bundle renders as a specimen record: dark imagery
  window left, printed marginalia right. Same-size icon-heading-text cards are
  the pattern this world exists to refuse.
- **The registration overlay is a real state**, toggled from the plate. It
  reveals the layout armature and the map's actual tiling partition (from
  `bundle.tiles`, not a decorative grid).
- **Size against the container, not the viewport.** The trace panel is 408px
  inside a 1440px page; viewport breakpoints there produce wrong layouts. Use
  `@container` for anything living in a pane.
- **Never truncate a graded value.** CRS, tool names, parameters and
  identifiers wrap. `EPSG:326…` is precisely the thing the reader came to check.

---

## 7. Motion

**One authored moment: the line printer.** The trace prints itself row by row
as the SSE stream arrives — `clip-path` reveal plus a 3px rise, 420ms on an
exponential ease-out (`--ease-print`). It is the native motion of the world
(a plotter emitting a record) and it happens exactly where the product's
argument lives.

Supporting, and deliberately minimal: `.carriage` for a running stage,
`.pulse-mark` for a live indicator and the streaming caret. Nothing else
animates. Everything is disabled under `prefers-reduced-motion`.

---

## 8. Browser surfaces

Themed, because they are part of the design: text selection (vermilion),
caret, focus ring (2px vermilion, 2px offset), scrollbars — including a
`.on-window` variant for the dark imagery pane — and tabular numerals on every
measurement. MapLibre's own scale bar and attribution are restyled into the
window's chrome rather than left at library defaults.

---

## 9. Print

The print stylesheet is a shipped deliverable, not a nicety: PDF generation is
on the plan's cut list, so `Ctrl+P` on a query view must produce the same
document. Materials flatten, `.no-print` chrome drops out, `.print-break`
keeps evidence and trace sections whole.

---

## 10. Synthetic imagery

`frontend/mocks/synthetic.ts` generates every raster from one seeded, continuous
world so tiles stitch seamlessly across zooms and the optical scene, the SAR
scene and the masks all describe the same ground.

Two rules make it honest, and both are load-bearing:

1. **Masks match their statistics.** Thresholds are calibrated by
   `scripts/calibrate-masks.mjs` so a mask claiming 3.4 km² paints 3.4 km² of
   the 674.6 km² footprint. A mask that contradicts its own number is exactly
   what a judge catches.
2. **The render matches the compatibility report.** A panchromatic scene
   renders greyscale, because the system refuses spectral questions about it.
   Showing it in colour would contradict the refusal.

Outside the scene footprint tiles are transparent, so the dark canvas reads as
"no data" rather than as a failed render.

**Every synthetic raster is labelled.** `alt` text says "synthetic imagery, not
observed data", and the identity plate carries a `MOCK DATA` stencil while
`VITE_USE_MOCKS` is on. Replace with real tiles by pointing at the live API;
nothing outside that one file knows the rasters are fabricated.

---

## 11. Standing rules

- Light or dark comes from the use scene, never the category.
- No CDN, no external tile provider, no telemetry on the demo path.
- The frontend computes nothing. It renders what the backend asserts, including
  keys it does not recognise.
- A refusal is styled advisory, never as an error.
- A rounded number always carries its raw value on hover.
- Icons are authored SVG on one 16-unit grid at 1.5 stroke
  (`src/components/icons.tsx`). No emoji, no icon font.
