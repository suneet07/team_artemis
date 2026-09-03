/**
 * Capture the shipped screens at the target sizes.
 *
 * Runs against the dev server with mocks, drives the real demo flow (ask a
 * question, wait for `done`), and writes one PNG per viewport so the layouts
 * can be reviewed as built rather than as intended.
 *
 *   node scripts/shoot.mjs [baseUrl] [outDir]
 */
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import process from "node:process";

const BASE = process.argv[2] ?? "http://localhost:5173";
const OUT = resolve(process.argv[3] ?? "../.impeccable/review");
mkdirSync(OUT, { recursive: true });

const VIEWPORTS = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "mobile", width: 390, height: 844 },
];

const BUNDLE = "bn_c3f81a";
const NO_CRS_BUNDLE = "bn_benchmark01";

async function settle(page, ms = 900) {
  await page.waitForTimeout(ms);
}

async function shoot(page, name, outDir, full = true) {
  await page.screenshot({
    path: join(outDir, `${name}.png`),
    fullPage: full,
  });
  process.stdout.write(`  ${name}.png\n`);
}

async function runViewport(browser, viewport) {
  const context = await browser.newContext({
    viewport: { width: viewport.width, height: viewport.height },
    deviceScaleFactor: 1,
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  const prefix = viewport.name;
  process.stdout.write(`${prefix} ${viewport.width}x${viewport.height}\n`);

  // S1 — landing
  await page.goto(BASE, { waitUntil: "networkidle" });
  await settle(page, 1600);
  await shoot(page, prefix, OUT);
  await shoot(page, `${prefix}-s1-landing`, OUT);

  // S2 — receiving
  await page.goto(`${BASE}/upload`, { waitUntil: "networkidle" });
  await settle(page);
  await shoot(page, `${prefix}-s2-upload`, OUT);

  // S4 + S5 — workspace, with a real query driven to completion
  await page.goto(`${BASE}/workspace/${BUNDLE}`, { waitUntil: "networkidle" });
  await settle(page, 2200);
  await shoot(page, `${prefix}-s4-workspace-empty`, OUT, false);

  const chip = page.getByRole("button", {
    name: /built-up area is under water/i,
  });
  if (await chip.count()) {
    await chip.first().click();
    await page
      .getByRole("button", { name: /view trace/i })
      .first()
      .waitFor({ timeout: 60_000 })
      .catch(() => {});
    await settle(page, 900);
    await shoot(page, `${prefix}-s4-workspace-answer`, OUT, false);
    await shoot(page, `${prefix}-s4-workspace-answer-full`, OUT);

    await page
      .getByRole("button", { name: /view trace/i })
      .first()
      .click()
      .catch(() => {});
    await settle(page, 900);
    await shoot(page, `${prefix}-s5-trace`, OUT);
  }

  // The refusal path
  const composer = page.locator("#composer");
  if (await composer.count()) {
    await page.getByRole("tab", { name: /chat/i }).click().catch(() => {});
    await settle(page, 400);
    await composer.fill("What changed between the two dates?");
    await page.getByRole("button", { name: /^Ask$/ }).click();
    await settle(page, 8000);
    await shoot(page, `${prefix}-s4-refusal`, OUT, false);
  }

  // §8.4 — the non-georeferenced viewer
  await page.goto(`${BASE}/workspace/${NO_CRS_BUNDLE}`, {
    waitUntil: "networkidle",
  });
  await settle(page, 1600);
  await shoot(page, `${prefix}-s4-no-crs`, OUT, false);

  // S7 — system
  await page.goto(`${BASE}/system`, { waitUntil: "networkidle" });
  await settle(page, 1400);
  await shoot(page, `${prefix}-s7-system`, OUT);

  await context.close();
}

const browser = await chromium.launch();
for (const viewport of VIEWPORTS) {
  await runViewport(browser, viewport);
}
await browser.close();
process.stdout.write(`\nwritten to ${OUT}\n`);
