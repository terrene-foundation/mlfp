#!/usr/bin/env node
// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//
// check-deck-overflow.js — Visual overflow detection for MLFP decks.
//
// Loads each Reveal.js deck in headless Chrome (puppeteer), waits for KaTeX
// rendering to settle, then walks every <section> and reports any slide
// whose scrollHeight exceeds the 720px viewport. This catches the failure
// mode where a slide *renders* but content is clipped at the bottom edge.
//
// Usage:
//   node scripts/check-deck-overflow.js                       # all 6 modules
//   node scripts/check-deck-overflow.js modules/mlfp05        # one module
//   node scripts/check-deck-overflow.js modules/mlfp05/deck.html
//   node scripts/check-deck-overflow.js --json                # JSON output
//   node scripts/check-deck-overflow.js --screenshots         # save .png of overflowing slides
//
// Exit codes:
//   0 — every slide fits within 1280x720
//   1 — at least one slide overflows (or a deck failed to load)
//   2 — invocation/setup error

const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const { spawn } = require("node:child_process");
const puppeteer = require("puppeteer");

const REPO_ROOT = path.resolve(__dirname, "..");
const VIEWPORT_W = 1280;
const VIEWPORT_H = 720;
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
          "  --screenshots: save .png of every overflowing slide to ./pdf/overflow-screenshots/\n" +
          "  --port=N: HTTP server port (default 8765)\n",
      );
      process.exit(0);
    } else args.paths.push(a);
  }
  return args;
}

function resolveDeckPaths(inputPaths) {
  // Convert each input to an absolute deck.html path.
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
  // Minimal static file server rooted at REPO_ROOT. Reveal decks need same-origin
  // for relative asset loading (CSS, JS, fonts) — file:// works in Chrome but
  // some module resolution and cross-resource fetches break, so we serve over HTTP.
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
      // Strip query string, decode, normalise.
      const urlPath = decodeURIComponent(req.url.split("?")[0]);
      const filePath = path.join(REPO_ROOT, urlPath);
      // Containment check — never serve outside REPO_ROOT.
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

async function checkOneDeck(browser, baseUrl, deckPath, opts) {
  // Returns { deckPath, totalSlides, overflowing: [{idx, displayedIdx, scrollHeight, title}] }.
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
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30_000 });

    // Wait for Reveal.js to be ready and for KaTeX rendering to settle.
    await page.waitForFunction(
      () =>
        typeof window.Reveal !== "undefined" &&
        window.Reveal.isReady &&
        window.Reveal.isReady(),
      { timeout: 15_000 },
    );
    // Allow one paint cycle + a small buffer for any deferred KaTeX rendering.
    await new Promise((r) => setTimeout(r, 500));

    const data = await page.evaluate((viewportH) => {
      const slides = document.querySelectorAll(".reveal .slides > section");
      const overflowing = [];
      slides.forEach((s, i) => {
        const sh = s.scrollHeight;
        if (sh > viewportH) {
          const titleEl = s.querySelector("h1, h2, h3");
          const title = (titleEl?.textContent || "(no title)")
            .trim()
            .slice(0, 80);
          overflowing.push({
            idx: i,
            displayedIdx: i + 1,
            scrollHeight: sh,
            overflowPx: sh - viewportH,
            title,
          });
        }
      });
      return { totalSlides: slides.length, overflowing };
    }, VIEWPORT_H);

    result.totalSlides = data.totalSlides;
    result.overflowing = data.overflowing;

    if (opts.screenshots && result.overflowing.length > 0) {
      // Save a screenshot of each overflowing slide for diagnosis.
      const outDir = path.join(REPO_ROOT, "pdf", "overflow-screenshots");
      fs.mkdirSync(outDir, { recursive: true });
      const moduleName = relPath.split(path.sep)[1] || "deck";
      for (const slide of result.overflowing) {
        // Deep-link to the specific slide.
        await page.goto(`${url}#/${slide.idx}`, { waitUntil: "networkidle0" });
        await new Promise((r) => setTimeout(r, 300));
        const fname = `${moduleName}-slide${String(slide.displayedIdx).padStart(3, "0")}.png`;
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

function printHumanReport(results) {
  let totalOverflow = 0;
  let totalSlides = 0;
  let failedDecks = 0;

  for (const r of results) {
    console.log("");
    console.log("=".repeat(60));
    console.log(`  ${r.deckPath}`);
    console.log("=".repeat(60));
    if (r.error) {
      console.log(`  [ERROR] ${r.error}`);
      failedDecks++;
      continue;
    }
    totalSlides += r.totalSlides;
    if (r.overflowing.length === 0) {
      console.log(`  [PASS] ${r.totalSlides} slides, 0 overflowing`);
    } else {
      totalOverflow += r.overflowing.length;
      console.log(
        `  [FAIL] ${r.totalSlides} slides, ${r.overflowing.length} overflowing:`,
      );
      for (const s of r.overflowing) {
        const shot = s.screenshot ? `  → ${s.screenshot}` : "";
        console.log(
          `    slide ${s.displayedIdx} (idx ${s.idx}, +${s.overflowPx}px): ${s.title}${shot}`,
        );
      }
    }
  }

  console.log("");
  console.log("=".repeat(60));
  console.log("  SUMMARY");
  console.log("=".repeat(60));
  console.log(`  Decks checked: ${results.length}`);
  console.log(`  Total slides:  ${totalSlides}`);
  console.log(`  Overflowing:   ${totalOverflow}`);
  if (failedDecks > 0) console.log(`  Failed loads:  ${failedDecks}`);
  console.log(
    totalOverflow === 0 && failedDecks === 0
      ? "\n  ✓ All decks pass overflow check"
      : "\n  ✗ Overflow detected — fix before shipping",
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
    for (const deck of decks) {
      results.push(await checkOneDeck(browser, baseUrl, deck, args));
    }

    if (args.json) {
      console.log(JSON.stringify(results, null, 2));
    } else {
      printHumanReport(results);
    }

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
