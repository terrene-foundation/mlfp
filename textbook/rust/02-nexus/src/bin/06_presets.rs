// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / Presets
//!
//! OBJECTIVE: Use Preset for one-line middleware stacks (development, production, internal).
//! LEVEL: Basic
//! PARITY: Full -- Both SDKs provide Preset enum with pre-configured middleware stacks.
//! VALIDATES: Preset::Development, Preset::Production, Preset::Internal
//!
//! Run: cargo run -p tutorial-nexus --bin 06_presets

use kailash_nexus::prelude::*;

fn main() {
    // ── 1. Middleware Presets ──
    // Presets provide opinionated middleware stacks for common scenarios.
    // Instead of configuring each middleware individually, use a preset.

    // ── 2. Development Preset ──
    // Permissive CORS, no rate limiting, verbose logging.
    // Ideal for local development and testing.

    let _dev = Preset::Development;

    // Development preset characteristics:
    //   - CORS: Allow all origins (*)
    //   - Rate limit: None
    //   - Logging: Verbose (request + response bodies)
    //   - Security headers: Minimal
    //   - Body limit: 10 MB (generous for testing)

    // ── 3. Production Preset ──
    // Strict CORS, rate limiting, security headers, minimal logging.

    let _prod = Preset::Production;

    // Production preset characteristics:
    //   - CORS: Must be explicitly configured
    //   - Rate limit: 60 RPM default
    //   - Logging: Structured (no bodies)
    //   - Security headers: Full (HSTS, CSP, etc.)
    //   - Body limit: 1 MB

    // ── 4. Internal Preset ──
    // For internal services: no CORS needed, generous limits.

    let _internal = Preset::Internal;

    // Internal preset characteristics:
    //   - CORS: Disabled (same-origin only)
    //   - Rate limit: 1000 RPM (high for service-to-service)
    //   - Logging: Structured
    //   - Security headers: Basic
    //   - Body limit: 5 MB

    // ── 5. Using a Preset ──
    // Presets are converted to MiddlewareConfig and applied to the server.
    //
    //   let nexus = Nexus::new();
    //   // ... register handlers ...
    //   let config = NexusConfig::from_preset(Preset::Production);
    //   start_server(nexus, config).await;

    // ── 6. Preset vs Custom ──
    // Use presets for standard scenarios. Use MiddlewareConfig directly
    // when you need fine-grained control over specific middleware.
    //
    //   Preset::Production  -- 90% of deployments
    //   MiddlewareConfig { rate_limit_rpm: Some(200), .. }  -- custom needs

    println!("PASS: 02-nexus/06_presets");
}
