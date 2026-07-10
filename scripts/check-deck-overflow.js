#!/usr/bin/env node
// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//
// check-deck-overflow.js — Canonical slide-overflow / content-clip detector.
//
// This is the SINGLE source of truth for "does a deck slide lose content".
// It supersedes the old scrollHeight>720 heuristic, which was blind to the
// dominant failure mode: content CLIPPED *inside* the 1280×720 frame by an
// `overflow:hidden` / `max-height` cap (fixed-height sections, capped tables,
// scrollable <pre>). Clipped content never extends past the slide edge, so an
// extent-only check reports "clean" while a table row or half a code block is
// silently cut off (verified against rendered PDFs — e.g. M5 "Reparameterisation
// Trick" loses its Step-4 box; "Five Algorithms" loses its PPO row).
//
// Algorithm (per slide — the deck is NAVIGATED slide-by-slide via Reveal.slide,
// because Reveal only lays out the current slide):
//   (A) EXTENT   — natural extent of visible children exceeds the 1280×720
//                  frame (auto-height sections that spill past the edge).
//   (B) CLIP     — any element (incl. the section) whose scrollHeight/Width
//                  exceeds its clientHeight/Width under overflow hidden/auto/
//                  scroll, i.e. content is cut off or forced to scroll.
// False-positive filters: KaTeX's clipped `.katex-mathml` a11y subtree is
// ignored; KaTeX struts add ~5-8px so a `.katex`/`.katex-display` vertical clip
// must exceed 15px; other vertical clips must exceed 12px; horizontal clips 10px.
//
// Usage:
//   node scripts/check-deck-overflow.js                       # all 6 modules
//   node scripts/check-deck-overflow.js modules/mlfp05        # one module
//   node scripts/check-deck-overflow.js modules/mlfp05/deck.html
//   node scripts/check-deck-overflow.js --json                # JSON output
//   node scripts/check-deck-overflow.js --screenshots         # save .png of clipped slides
//
// Exit codes:
//   0 — every slide fits within 1280×720 with no clipped content
//   1 — at least one slide overflows/clips (or a deck failed to load)
//   2 — invocation/setup error

const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const puppeteer = require("puppeteer");

const REPO_ROOT = path.resolve(__dirname, "..");
const VIEWPORT_W = 1280;
const VIEWPORT_H = 720;
const EXT_TOL = 6; // px past the frame edge before EXTENT counts
const ALL_MODULES = [
  "modules/mlfp01",
  "modules/mlfp02",
  "modules/mlfp03",
  "modules/mlfp04",
  "modules/mlfp05",
  "modules/mlfp06",
];

function parseArgs(argv) {
  const args = { paths: [], json: false, screenshots: false, port: 8765 };
  for (const a of argv.slice(2)) {
    if (a === "--json") args.json = true;
    else if (a === "--screenshots") args.screenshots = true;
    else if (a.startsWith("--port=")) args.port = parseInt(a.slice(7), 10);
    else if (a === "-h" || a === "--help") {
      console.log(
        "Usage: check-deck-overflow.js [paths...] [--json] [--screenshots] [--port=N]\n" +
          "  paths: deck.html files or module dirs (default: all 6 modules)\n" +
          "  --json: machine-readable output\n" +
          "  --screenshots: save .png of every clipped slide to ./pdf/overflow-screenshots/\n" +
          "  --port=N: HTTP server port (default 8765)\n",
      );
      process.exit(0);
    } else args.paths.push(a);
  }
  return args;
}

function resolveDeckPaths(inputPaths) {
  const out = [];
  const inputs = inputPaths.length ? inputPaths : ALL_MODULES;
  for (const input of inputs) {
    let p = path.resolve(REPO_ROOT, input);
    if (fs.existsSync(p) && fs.statSync(p).isDirectory())
      p = path.join(p, "deck.html");
    if (!p.endsWith(".html")) p = path.join(p, "deck.html");
    if (!fs.existsSync(p)) {
      console.error(`[error] deck not found: ${p}`);
      continue;
    }
    out.push(p);
  }
  return out;
}

function startStaticServer(port) {
  return new Promise((resolve, reject) => {
    const mime = {
      ".html": "text/html",
      ".css": "text/css",
      ".js": "application/javascript",
      ".json": "application/json",
      ".svg": "image/svg+xml",
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".woff": "font/woff",
      ".woff2": "font/woff2",
      ".ttf": "font/ttf",
    };
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
        const ext = path.extname(filePath).toLowerCase();
        res.writeHead(200, {
          "Content-Type": mime[ext] || "application/octet-stream",
        });
        fs.createReadStream(filePath).pipe(res);
      });
    });
    server.on("error", reject);
    server.listen(port, "127.0.0.1", () => resolve(server));
  });
}

// The per-slide measurement, evaluated in the page. Returns null (fits) or a
// finding {ext:{v,h}, clips:[{tag,vo,ho}], title}. Kept as a string so it can
// be passed to page.evaluate with the slide index.
const MEASURE_FN = `(FW, FH, EXT_TOL) => {
  const s = window.Reveal.getCurrentSlide();
  if (!s) return null;
  const scale = Reveal.getScale ? Reveal.getScale() : 1;
  const srect = s.getBoundingClientRect();
  // (A) natural extent of visible children (layout position ignores clipping)
  let maxR = 0, maxB = 0;
  s.querySelectorAll("*").forEach((el) => {
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none") return;
    if (cs.position === "absolute" && cs.clip && cs.clip !== "auto" && cs.clip !== "") return;
    if (el.closest(".katex-mathml")) return;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    const right = (r.right - srect.left) / scale;
    const bottom = (r.bottom - srect.top) / scale;
    if (right > 2400) return; // pathological measuring node
    if (right > maxR) maxR = right;
    if (bottom > maxB) maxB = bottom;
  });
  // (B) elements clipping content under overflow hidden/auto/scroll
  const clips = [];
  [s, ...s.querySelectorAll("*")].forEach((el) => {
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") return;
    if (el.closest(".katex-mathml")) return;
    const isKatex = el.classList.contains("katex-display") || el.classList.contains("katex");
    const vMin = isKatex ? 15 : 12; // katex struts add ~5-8px false vertical
    const vClip = (cs.overflowY === "hidden" || cs.overflowY === "auto" || cs.overflowY === "scroll") &&
      el.scrollHeight - el.clientHeight > vMin && el.clientHeight > 0;
    const hClip = (cs.overflowX === "hidden" || cs.overflowX === "auto" || cs.overflowX === "scroll") &&
      el.scrollWidth - el.clientWidth > 10 && el.clientWidth > 0;
    if (vClip || hClip) {
      const cls = (el.className || "").toString().trim().split(/\\s+/)[0] || "";
      clips.push({ tag: el.tagName + (cls ? "." + cls : ""),
        vo: vClip ? el.scrollHeight - el.clientHeight : 0,
        ho: hClip ? el.scrollWidth - el.clientWidth : 0 });
    }
  });
  const vExt = Math.round(maxB) - FH;
  const hExt = Math.round(maxR) - FW;
  const hasExt = vExt > EXT_TOL || hExt > EXT_TOL;
  if (!hasExt && clips.length === 0) return null;
  const t = s.querySelector("h1, h2, h3");
  const title = (t ? t.textContent : "").trim().slice(0, 80);
  return {
    ext: { v: vExt > EXT_TOL ? vExt : 0, h: hExt > EXT_TOL ? hExt : 0 },
    clips: clips.sort((a, b) => (b.vo + b.ho) - (a.vo + a.ho)).slice(0, 4),
    title,
  };
}`;

async function checkOneDeck(browser, baseUrl, deckPath, opts) {
  const relPath = path.relative(REPO_ROOT, deckPath);
  const url = `${baseUrl}/${relPath}`;
  const page = await browser.newPage();
  await page.setViewport({
    width: VIEWPORT_W,
    height: VIEWPORT_H,
    deviceScaleFactor: 1,
  });

  const result = {
    deckPath: relPath,
    totalSlides: 0,
    overflowing: [],
    error: null,
  };

  try {
    // `load` (not networkidle0): some decks keep long-poll/font connections
    // open and never reach network-idle (M5 timed out under networkidle0).
    await page.goto(url, { waitUntil: "load", timeout: 30_000 });
    await page.waitForFunction(
      () =>
        typeof window.Reveal !== "undefined" &&
        window.Reveal.isReady &&
        window.Reveal.isReady(),
      { timeout: 15_000 },
    );
    await page
      .waitForFunction(
        () => document.fonts && document.fonts.status === "loaded",
        { timeout: 8_000 },
      )
      .catch(() => {});
    await new Promise((r) => setTimeout(r, 800));

    // Enumerate every slide (horizontal + vertical stacks).
    const indices = await page.evaluate(() => {
      const out = [];
      window.Reveal.getHorizontalSlides().forEach((h, i) => {
        const v = h.querySelectorAll("section");
        if (v.length) for (let j = 0; j < v.length; j++) out.push([i, j]);
        else out.push([i, 0]);
      });
      return out;
    });
    result.totalSlides = indices.length;

    for (const [h, v] of indices) {
      await page.evaluate((hh, vv) => window.Reveal.slide(hh, vv), h, v);
      await new Promise((r) => setTimeout(r, 90));
      const finding = await page.evaluate(
        `(${MEASURE_FN})(${VIEWPORT_W}, ${VIEWPORT_H}, ${EXT_TOL})`,
      );
      if (finding) {
        result.overflowing.push({
          h,
          v,
          displayedIdx: result.overflowing.length,
          ...finding,
        });
      }
    }

    if (opts.screenshots && result.overflowing.length > 0) {
      const outDir = path.join(REPO_ROOT, "pdf", "overflow-screenshots");
      fs.mkdirSync(outDir, { recursive: true });
      const moduleName = relPath.split(path.sep)[1] || "deck";
      for (const slide of result.overflowing) {
        await page.evaluate(
          (hh, vv) => window.Reveal.slide(hh, vv),
          slide.h,
          slide.v,
        );
        await new Promise((r) => setTimeout(r, 250));
        const fname = `${moduleName}-h${String(slide.h).padStart(3, "0")}-v${slide.v}.png`;
        await page.screenshot({
          path: path.join(outDir, fname),
          fullPage: false,
        });
        slide.screenshot = path.relative(REPO_ROOT, path.join(outDir, fname));
      }
    }
  } catch (err) {
    result.error = err.message || String(err);
  } finally {
    await page.close();
  }
  return result;
}

function describe(o) {
  const tags = [];
  if (o.ext.v) tags.push(`EXT-V+${o.ext.v}`);
  if (o.ext.h) tags.push(`EXT-H+${o.ext.h}`);
  for (const c of o.clips) {
    if (c.vo) tags.push(`CLIP-V${c.vo}[${c.tag}]`);
    if (c.ho) tags.push(`CLIP-H${c.ho}[${c.tag}]`);
  }
  return tags.join("  ");
}

function printHumanReport(results) {
  let totalOverflow = 0,
    totalSlides = 0,
    failedDecks = 0;
  for (const r of results) {
    console.log("\n" + "=".repeat(60) + `\n  ${r.deckPath}\n` + "=".repeat(60));
    if (r.error) {
      console.log(`  [ERROR] ${r.error}`);
      failedDecks++;
      continue;
    }
    totalSlides += r.totalSlides;
    if (r.overflowing.length === 0) {
      console.log(`  [PASS] ${r.totalSlides} slides, 0 clipped`);
    } else {
      totalOverflow += r.overflowing.length;
      console.log(
        `  [FAIL] ${r.totalSlides} slides, ${r.overflowing.length} clipped:`,
      );
      for (const s of r.overflowing) {
        const shot = s.screenshot ? `  → ${s.screenshot}` : "";
        console.log(
          `    h=${s.h} v=${s.v}  ${describe(s)}  | ${s.title}${shot}`,
        );
      }
    }
  }
  console.log("\n" + "=".repeat(60) + "\n  SUMMARY\n" + "=".repeat(60));
  console.log(`  Decks checked: ${results.length}`);
  console.log(`  Total slides:  ${totalSlides}`);
  console.log(`  Clipped:       ${totalOverflow}`);
  if (failedDecks > 0) console.log(`  Failed loads:  ${failedDecks}`);
  console.log(
    totalOverflow === 0 && failedDecks === 0
      ? "\n  ✓ All decks pass overflow check"
      : "\n  ✗ Overflow/clip detected — fix before shipping",
  );
}

async function main() {
  const args = parseArgs(process.argv);
  const decks = resolveDeckPaths(args.paths);
  if (decks.length === 0) {
    console.error("[error] no decks found to check");
    process.exit(2);
  }
  const server = await startStaticServer(args.port);
  const baseUrl = `http://127.0.0.1:${args.port}`;
  let browser;
  try {
    browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-dev-shm-usage"],
    });
    const results = [];
    for (const deck of decks)
      results.push(await checkOneDeck(browser, baseUrl, deck, args));
    if (args.json) console.log(JSON.stringify(results, null, 2));
    else printHumanReport(results);
    const anyOverflow = results.some(
      (r) => r.error || r.overflowing.length > 0,
    );
    process.exitCode = anyOverflow ? 1 : 0;
  } catch (err) {
    console.error(`[error] ${err.message || err}`);
    process.exitCode = 2;
  } finally {
    if (browser) await browser.close();
    server.close();
  }
}

main();
