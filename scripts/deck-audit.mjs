#!/usr/bin/env node
/**
 * Deck Overflow Audit — Uses Playwright to inspect every slide for overflow issues.
 * Reports: elements exceeding viewport bounds (right/bottom), formula overflow, content clipping.
 *
 * Usage: node scripts/deck-audit.mjs [module]
 *   e.g.: node scripts/deck-audit.mjs ascent01
 *         node scripts/deck-audit.mjs all
 */
import { chromium } from 'playwright';
import { execSync } from 'child_process';
import http from 'http';
import fs from 'fs';
import path from 'path';

const VIEWPORT = { width: 1280, height: 720 };
const DECKS_DIR = path.resolve(import.meta.dirname, '..', 'decks');

// Simple static file server
function startServer(root, port) {
  const server = http.createServer((req, res) => {
    const filePath = path.join(root, decodeURIComponent(req.url.split('?')[0]));
    if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      const indexPath = path.join(filePath, 'index.html');
      if (fs.existsSync(indexPath)) {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(fs.readFileSync(indexPath));
        return;
      }
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath);
    const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.pdf': 'application/pdf' };
    res.writeHead(200, { 'Content-Type': types[ext] || 'application/octet-stream' });
    res.end(fs.readFileSync(filePath));
  });
  server.listen(port);
  return server;
}

async function auditDeck(module, port) {
  const url = `http://localhost:${port}/${module}/deck.html`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: VIEWPORT });

  console.log(`\n${'='.repeat(60)}`);
  console.log(`  Auditing ${module} — ${url}`);
  console.log(`${'='.repeat(60)}\n`);

  await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000); // Let KaTeX render

  // Get total slide count
  const totalSlides = await page.evaluate(() => {
    return Reveal.getTotalSlides();
  });
  console.log(`Total slides: ${totalSlides}`);

  const issues = [];

  for (let i = 0; i < totalSlides; i++) {
    // Navigate to slide
    await page.evaluate((idx) => {
      const slides = Reveal.getSlides();
      if (slides[idx]) {
        const indices = Reveal.getIndices(slides[idx]);
        Reveal.slide(indices.h, indices.v);
      }
    }, i);
    await page.waitForTimeout(300);

    // Check for overflow
    const slideIssues = await page.evaluate((slideIdx) => {
      const issues = [];
      const currentSlide = Reveal.getCurrentSlide();
      if (!currentSlide) return issues;

      const slideRect = currentSlide.getBoundingClientRect();
      const viewportW = 1280;
      const viewportH = 720;

      // Check all children for overflow
      const children = currentSlide.querySelectorAll('*');
      for (const child of children) {
        const rect = child.getBoundingClientRect();
        const tag = child.tagName.toLowerCase();
        const cls = child.className?.toString?.()?.slice(0, 40) || '';

        // Right overflow
        if (rect.right > viewportW + 2) {
          issues.push({
            type: 'RIGHT_OVERFLOW',
            tag,
            class: cls,
            overflow: Math.round(rect.right - viewportW),
            width: Math.round(rect.width),
            text: child.textContent?.slice(0, 50) || ''
          });
        }

        // Bottom overflow
        if (rect.bottom > viewportH + 2) {
          issues.push({
            type: 'BOTTOM_OVERFLOW',
            tag,
            class: cls,
            overflow: Math.round(rect.bottom - viewportH),
            height: Math.round(rect.height),
            text: child.textContent?.slice(0, 50) || ''
          });
        }
      }

      // Deduplicate (parent/child both overflow)
      const seen = new Set();
      return issues.filter(issue => {
        const key = `${issue.type}-${issue.tag}-${issue.overflow}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }, i);

    if (slideIssues.length > 0) {
      // Filter to significant overflows (>5px)
      const significant = slideIssues.filter(s => s.overflow > 5);
      if (significant.length > 0) {
        console.log(`\n  Slide ${i + 1}: ${significant.length} overflow(s)`);
        for (const issue of significant.slice(0, 5)) { // Top 5 per slide
          console.log(`    ${issue.type}: <${issue.tag} class="${issue.class}"> +${issue.overflow}px`);
          if (issue.text) console.log(`      "${issue.text.trim().slice(0, 60)}..."`);
        }
        issues.push({ slide: i + 1, issues: significant });
      }
    }
  }

  // Take screenshot of first problematic slide
  if (issues.length > 0) {
    const firstBad = issues[0].slide - 1;
    await page.evaluate((idx) => {
      const slides = Reveal.getSlides();
      if (slides[idx]) {
        const indices = Reveal.getIndices(slides[idx]);
        Reveal.slide(indices.h, indices.v);
      }
    }, firstBad);
    await page.waitForTimeout(500);
    const screenshotPath = path.join(DECKS_DIR, module, 'overflow-sample.png');
    await page.screenshot({ path: screenshotPath });
    console.log(`\n  Screenshot saved: ${screenshotPath}`);
  }

  console.log(`\n  Summary: ${issues.length} slides with overflow out of ${totalSlides}`);
  await browser.close();
  return issues;
}

// Main
const targetModule = process.argv[2] || 'ascent01';
const PORT = 8099;
const server = startServer(DECKS_DIR, PORT);

try {
  if (targetModule === 'all') {
    for (const mod of ['ascent01', 'ascent02', 'ascent03', 'ascent04', 'ascent05', 'ascent06']) {
      await auditDeck(mod, PORT);
    }
  } else {
    await auditDeck(targetModule, PORT);
  }
} finally {
  server.close();
}
