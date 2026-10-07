#!/usr/bin/env node
/**
 * Hook: auto-format
 * Event: PostToolUse
 * Matcher: Edit|Write
 * Purpose: Auto-format Python, JavaScript, TypeScript files.
 *
 * Advisory posture (owner directive 2026-10-07): never blocks. When a
 * formatter actually rewrote the file, the agent is told via
 * additionalContext — silent on-disk mutation after a Write looks like
 * a revert to the agent and has caused double-write confusion.
 *
 * Exit Codes:
 *   0 = always (outcome reported as additionalContext)
 */

const fs = require("fs");
const crypto = require("crypto");
const { execFileSync } = require("child_process");
const path = require("path");

function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

// Timeout fallback — prevents hanging the Claude Code session
const TIMEOUT_MS = 10000;
const _timeout = setTimeout(() => {
  console.log(JSON.stringify({ continue: true }));
  process.exit(1);
}, TIMEOUT_MS);

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  try {
    const data = JSON.parse(input);
    const result = autoFormat(data);
    console.log(
      JSON.stringify({
        continue: true,
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          formatted: result.formatted,
          formatter: result.formatter,
          ...(result.changed && {
            additionalContext:
              `auto-format rewrote ${path.basename(data.tool_input?.file_path || "file")} ` +
              `(${result.formatter}) on disk after your write. The on-disk bytes differ ` +
              `from what you wrote — re-read the file before making byte-sensitive edits. ` +
              `This is formatting only; your content was not reverted.`,
          }),
        },
      }),
    );
    process.exit(0);
  } catch (error) {
    console.error(`[HOOK ERROR] ${error.message}`);
    console.log(JSON.stringify({ continue: true }));
    process.exit(1);
  }
});

function autoFormat(data) {
  const filePath = data.tool_input?.file_path;
  const cwd = data.cwd || process.cwd();

  if (!filePath || !fs.existsSync(filePath)) {
    return { formatted: false, formatter: "none" };
  }

  // Validate file is within the project directory to prevent symlink attacks
  const resolvedPath = path.resolve(filePath);
  const resolvedCwd = path.resolve(cwd);
  if (!resolvedPath.startsWith(resolvedCwd)) {
    return { formatted: false, formatter: "path outside project" };
  }

  const ext = path.extname(filePath).toLowerCase();

  try {
    // Python files: black or ruff
    if (ext === ".py") {
      const before = sha256(filePath);
      try {
        execFileSync("black", [filePath], { stdio: "pipe" });
        return { formatted: true, formatter: "black", changed: sha256(filePath) !== before };
      } catch {
        // Try ruff if black not available
        try {
          execFileSync("ruff", ["format", filePath], { stdio: "pipe" });
          return { formatted: true, formatter: "ruff", changed: sha256(filePath) !== before };
        } catch {
          return { formatted: false, formatter: "none (black/ruff not found)" };
        }
      }
    }

    // JavaScript/TypeScript files: prettier
    if ([".js", ".jsx", ".ts", ".tsx", ".json"].includes(ext)) {
      const before = sha256(filePath);
      try {
        execFileSync("npx", ["prettier", "--write", filePath], {
          stdio: "pipe",
        });
        return { formatted: true, formatter: "prettier", changed: sha256(filePath) !== before };
      } catch {
        return { formatted: false, formatter: "none (prettier not found)" };
      }
    }

    // YAML/Markdown: prettier
    if ([".yaml", ".yml", ".md"].includes(ext)) {
      const before = sha256(filePath);
      try {
        execFileSync("npx", ["prettier", "--write", filePath], {
          stdio: "pipe",
        });
        return { formatted: true, formatter: "prettier", changed: sha256(filePath) !== before };
      } catch {
        return { formatted: false, formatter: "none" };
      }
    }

    return { formatted: false, formatter: "unsupported file type" };
  } catch (error) {
    return { formatted: false, formatter: `error: ${error.message}` };
  }
}
