#!/usr/bin/env node
/**
 * Hook: validate-deployment
 * Event: PostToolUse
 * Matcher: Edit|Write
 * Purpose: Detect cloud credentials and secrets in deployment-related files.
 *
 * Framework-agnostic — works with any project.
 *
 * Advisory posture (owner directive 2026-10-07): this hook NEVER blocks.
 * Credential findings are delivered via additionalContext so the main agent
 * reads and resolves them in-session instead of the session halting.
 *
 * Exit Codes:
 *   0 = always (findings reported as additionalContext)
 */

const TIMEOUT_MS = 10000;
const timeout = setTimeout(() => {
  console.error(
    "[HOOK TIMEOUT] validate-deployment exceeded 10s — CREDENTIAL CHECK SKIPPED",
  );
  console.log(
    JSON.stringify({
      continue: true,
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        validation:
          "WARNING: Deployment credential check timed out. Manual review required.",
      },
    }),
  );
  process.exit(1);
}, TIMEOUT_MS);

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  clearTimeout(timeout);
  try {
    const data = JSON.parse(input);
    const result = validateDeployment(data);
    console.log(
      JSON.stringify({
        continue: true,
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          validation: result.message,
          ...(result.critical && {
            additionalContext:
              "ADVISORY from validate-deployment (non-blocking per owner directive): " +
              result.message +
              " — MAIN AGENT: a credential/secret pattern was detected in a deployment file. Resolve now (remove the secret, rotate if real), do not defer.",
          }),
        },
      }),
    );
    process.exit(0);
  } catch (error) {
    console.error(`[HOOK ERROR] ${error.message}`);
    console.log(
      JSON.stringify({
        continue: true,
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          validation:
            "WARNING: Deployment validation hook errored. Manual credential review required.",
        },
      }),
    );
    process.exit(1);
  }
});

function validateDeployment(data) {
  const filePath = data.tool_input?.file_path || "";
  const content = data.tool_input?.content || data.tool_input?.new_string || "";

  // Only check deployment-related files
  const deploymentFiles = [
    /deploy\//,
    /Dockerfile/i,
    /docker-compose/i,
    /\.ya?ml$/,
    /terraform/i,
    /\.tf$/,
    /k8s\//,
    /kubernetes\//,
    /\.github\/workflows\//,
    /Makefile/i,
  ];

  const isDeploymentFile = deploymentFiles.some((p) => p.test(filePath));
  if (!isDeploymentFile) {
    return { continue: true, exitCode: 0, message: "Not a deployment file" };
  }

  // CHECK 1: Cloud credential patterns — ADVISORY (was BLOCK pre-2026-10-07)
  const credentialPatterns = [
    // AWS Access Key ID
    {
      pattern: /AKIA[0-9A-Z]{16}/,
      message: "SECRET DETECTED: AWS Access Key ID detected",
    },
    // AWS Secret Access Key (broad context match)
    {
      pattern: /[0-9a-zA-Z/+]{40}(?=\s|"|'|$)/,
      context:
        /aws_secret|AWS_SECRET|secret_access_key|SecretAccessKey|AKIA[0-9A-Z]{16}/i,
      message: "SECRET DETECTED: Possible AWS Secret Access Key detected",
    },
    // Azure Storage Account Key
    {
      pattern: /AccountKey=[^;]{20,}/,
      message: "SECRET DETECTED: Azure Storage Account Key detected",
    },
    // Azure Client Secret
    {
      pattern:
        /AZURE_CLIENT_SECRET\s*[:=]\s*["'][^"']+["']|client_secret\s*[:=]\s*["'][0-9a-zA-Z~._-]{30,}["']/,
      message: "SECRET DETECTED: Azure Client Secret detected",
    },
    // GCP Service Account JSON
    {
      pattern: /"type"\s*:\s*"service_account"/,
      message: "SECRET DETECTED: GCP Service Account JSON detected",
    },
    // Private keys
    {
      pattern: /-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----/,
      message: "SECRET DETECTED: Private key detected in deployment file",
    },
    // GitHub PATs (classic and fine-grained)
    {
      pattern: /ghp_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z_]{22,}/,
      message: "SECRET DETECTED: GitHub Personal Access Token detected",
    },
    // PyPI API tokens
    {
      pattern: /pypi-[0-9a-zA-Z_-]{50,}/,
      message: "SECRET DETECTED: PyPI API token detected",
    },
    // Docker Hub tokens
    {
      pattern: /dckr_pat_[0-9a-zA-Z_-]{20,}/,
      message: "SECRET DETECTED: Docker Hub token detected",
    },
    // Generic API secret keys (OpenAI, Anthropic, Stripe, etc.)
    {
      pattern: /sk-[a-zA-Z0-9]{20,}/,
      message: "SECRET DETECTED: API secret key pattern detected",
    },
  ];

  for (const { pattern, context, message } of credentialPatterns) {
    if (pattern.test(content)) {
      if (context && !context.test(content)) continue;
      return { continue: true, exitCode: 0, message, critical: true };
    }
  }

  // CHECK 2: Plaintext passwords in config — WARN
  const passwordPatterns = [
    /password\s*[:=]\s*["'][^"']{3,}["']/i,
    /POSTGRES_PASSWORD\s*[:=]\s*["'][^"']{3,}["']/i,
    /REDIS_PASSWORD\s*[:=]\s*["'][^"']{3,}["']/i,
    /DB_PASSWORD\s*[:=]\s*["'][^"']{3,}["']/i,
  ];

  const warnings = [];
  for (const pattern of passwordPatterns) {
    if (pattern.test(content)) {
      warnings.push(
        "WARNING: Plaintext password detected. Use secrets manager in production.",
      );
      break;
    }
  }

  // CHECK 3: Dockerfile best practices — WARN
  if (/Dockerfile/i.test(filePath)) {
    // Skip COPY . . check for multi-stage COPY --from patterns
    if (
      /COPY\s+\.\s+\./.test(content) &&
      !content.includes(".dockerignore") &&
      !/COPY\s+--from=/.test(content)
    ) {
      warnings.push(
        "WARNING: COPY . . without .dockerignore may include secrets or unnecessary files.",
      );
    }
    if (!/USER\s+\w+/.test(content) && !/FROM\s+scratch/i.test(content)) {
      warnings.push("WARNING: No USER directive — container will run as root.");
    }
    if (!/HEALTHCHECK/.test(content)) {
      warnings.push("WARNING: No HEALTHCHECK directive.");
    }
  }

  if (warnings.length > 0) {
    return {
      continue: true,
      exitCode: 0,
      message: warnings.join(" | "),
    };
  }

  return { continue: true, exitCode: 0, message: "Deployment file validated" };
}
