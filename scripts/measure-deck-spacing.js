#!/usr/bin/env node
// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//
// measure-deck-spacing.js — Vertical spacing diagnostics for MLFP decks.
//
// The overflow checker (check-deck-overflow.js) only catches slides TALLER
// than the 720px viewport. It is blind to the opposite failure modes:
//   - SPARSE: content fills <50% of the slide, lots of dead space.
//   - CRAMPED: content fills >88% — fits, but visually tight, one edit from
//     overflowing.
//   - TOP-HEAVY / BOTTOM-HEAVY: content not vertically balanced (Reveal
//     centers by default, so large asymmetry signals a layout quirk).
//
// For each <section> it measures the content block (union of direct children,
// excluding aside.notes and .slide-footer) in the section's own unscaled
// layout coordinates (offsetTop/offsetHeight), then reports fill ratio and
// top/bottom whitespace.
//
// Usage: node scripts/measure-deck-spacing.js [module-dir ...] [--json]

const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const puppeteer = require("puppeteer");

const REPO_ROOT = path.resolve(__dirname, "..");
const VIEWPORT_W = 1280;
const VIEWPORT_H = 720;
const SPARSE = 0.5; // fill below this = too much dead space
const CRAMPED = 0.88; // fill above this = visually tight
const ALL = [
  "modules/mlfp01",
  "modules/mlfp02",
  "modules/mlfp03",
  "modules/mlfp04",
  "modules/mlfp05",
  "modules/mlfp06",
];

function startServer(port) {
  const mime = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
  };
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const urlPath = decodeURIComponent(req.url.split("?")[0]);
      const filePath = path.join(REPO_ROOT, urlPath);
      if (!filePath.startsWith(REPO_ROOT)) {
        res.writeHead(403);
        res.end("forbidden");
        return;
      }
      fs.stat(filePath, (err, stat) => {
        if (err || !stat.isFile()) {
          res.writeHead(404);
          res.end("not found");
          return;
        }
        res.writeHead(200, {
          "Content-Type":
            mime[path.extname(filePath).toLowerCase()] ||
            "application/octet-stream",
        });
        fs.createReadStream(filePath).pipe(res);
      });
    });
    server.on("error", reject);
    server.listen(port, "127.0.0.1", () => resolve(server));
  });
}

async function measureDeck(browser, baseUrl, deckRel) {
  const page = await browser.newPage();
  await page.setViewport({
    width: VIEWPORT_W,
    height: VIEWPORT_H,
    deviceScaleFactor: 1,
  });
  const out = { deck: deckRel, slides: [], error: null };
  try {
    await page.goto(`${baseUrl}/${deckRel}`, {
      waitUntil: "networkidle0",
      timeout: 30_000,
    });
    await page.waitForFunction(
      () =>
        typeof window.Reveal !== "undefined" &&
        window.Reveal.isReady &&
        window.Reveal.isReady(),
      { timeout: 15_000 },
    );
    await new Promise((r) => setTimeout(r, 500));
    // Reveal sets off-screen horizontal slides to display:none, so each slide
    // must be activated before its content can be measured.
    const count = await page.evaluate(
      () => document.querySelectorAll(".reveal .slides > section").length,
    );
    for (let i = 0; i < count; i++) {
      const slide = await page.evaluate(
        async (idx, VH) => {
          window.Reveal.slide(idx);
          await new Promise((r) =>
            requestAnimationFrame(() => requestAnimationFrame(r)),
          );
          const sections = document.querySelectorAll(
            ".reveal .slides > section",
          );
          const s = sections[idx];
          const kids = Array.from(s.children).filter(
            (c) =>
              c.tagName !== "ASIDE" && !c.classList.contains("slide-footer"),
          );
          let top = Infinity,
            bottom = -Infinity;
          for (const c of kids) {
            const r = c.getBoundingClientRect();
            if (r.height === 0 && r.width === 0) continue;
            top = Math.min(top, c.offsetTop);
            bottom = Math.max(bottom, c.offsetTop + c.offsetHeight);
          }
          if (!isFinite(top)) {
            top = 0;
            bottom = 0;
          }
          const contentH = bottom - top;
          const titleEl = s.querySelector("h1, h2, h3");
          return {
            idx: idx + 1,
            fill: +(contentH / VH).toFixed(2),
            contentH: Math.round(contentH),
            topGap: Math.round(top),
            botGap: Math.round(VH - bottom),
            cls: s.className || "",
            title: (titleEl?.textContent || "(no title)").trim().slice(0, 54),
          };
        },
        i,
        VIEWPORT_H,
      );
      out.slides.push(slide);
    }
  } catch (err) {
    out.error = err.message || String(err);
  } finally {
    await page.close();
  }
  return out;
}

async function main() {
  const argv = process.argv.slice(2);
  const json = argv.includes("--json");
  let targets = argv.filter((a) => !a.startsWith("--"));
  if (targets.length === 0) targets = ALL;
  const decks = targets.map((t) =>
    t.endsWith(".html") ? t : path.join(t, "deck.html"),
  );

  const port = 8799;
  const server = await startServer(port);
  const browser = await puppeteer.launch({ headless: "new" });
  const baseUrl = `http://127.0.0.1:${port}`;
  const results = [];
  for (const d of decks) {
    results.push(await measureDeck(browser, baseUrl, d));
  }
  await browser.close();
  server.close();

  if (json) {
    console.log(JSON.stringify(results, null, 2));
    return;
  }

  let totSparse = 0,
    totCramped = 0,
    totSlides = 0;
  for (const r of results) {
    console.log("\n" + "=".repeat(64));
    console.log("  " + r.deck);
    console.log("=".repeat(64));
    if (r.error) {
      console.log("  [ERROR] " + r.error);
      continue;
    }
    totSlides += r.slides.length;
    const sparse = r.slides.filter((s) => s.fill < SPARSE);
    const cramped = r.slides.filter((s) => s.fill >= CRAMPED);
    totSparse += sparse.length;
    totCramped += cramped.length;
    const fills = r.slides.map((s) => s.fill);
    const avg = (fills.reduce((a, b) => a + b, 0) / fills.length).toFixed(2);
    console.log(
      `  ${r.slides.length} slides | avg fill ${avg} | sparse(<${SPARSE}) ${sparse.length} | cramped(>=${CRAMPED}) ${cramped.length}`,
    );
    if (sparse.length) {
      console.log("  -- SPARSE (dead space) --");
      for (const s of sparse)
        console.log(
          `     #${String(s.idx).padStart(3)} fill ${s.fill}  top ${String(s.topGap).padStart(3)} bot ${String(s.botGap).padStart(3)}  ${s.title}`,
        );
    }
    if (cramped.length) {
      console.log("  -- CRAMPED (tight) --");
      for (const s of cramped)
        console.log(
          `     #${String(s.idx).padStart(3)} fill ${s.fill}  top ${String(s.topGap).padStart(3)} bot ${String(s.botGap).padStart(3)}  ${s.title}`,
        );
    }
  }
  console.log("\n" + "=".repeat(64));
  console.log(
    `  TOTAL ${totSlides} slides | sparse ${totSparse} | cramped ${totCramped}`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
