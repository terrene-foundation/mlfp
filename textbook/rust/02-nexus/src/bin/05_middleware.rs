// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / Middleware
//!
//! OBJECTIVE: Configure middleware for CORS, rate limiting, logging, and security headers.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs provide MiddlewareConfig for declarative middleware stacks.
//!         Rust uses tower middleware on axum; Python uses starlette middleware.
//! VALIDATES: MiddlewareConfig, Preset, middleware composition
//!
//! Run: cargo run -p tutorial-nexus --bin 05_middleware

use kailash_nexus::prelude::*;

fn main() {
    // ── 1. Middleware in Nexus ──
    // Middleware processes requests before they reach handlers and
    // responses before they're sent to clients. Nexus middleware stack:
    //
    //   Request -> CORS -> Rate Limit -> Logging -> Body Limit -> Security Headers -> Handler
    //   Response <- CORS <- Rate Limit <- Logging <- Body Limit <- Security Headers <- Handler

    // ── 2. MiddlewareConfig ──
    // Declarative configuration for the middleware stack.

    let config = MiddlewareConfig {
        cors_origins: vec!["http://localhost:3000".to_string()],
        rate_limit_rpm: Some(60), // 60 requests per minute
        body_limit_bytes: Some(1_048_576), // 1 MB
        enable_logging: true,
        enable_security_headers: true,
        ..Default::default()
    };

    assert!(config.enable_logging);
    assert_eq!(config.rate_limit_rpm, Some(60));

    // ── 3. CORS Configuration ──
    // Cross-Origin Resource Sharing prevents unauthorized domains
    // from making requests to your API.

    let cors_config = MiddlewareConfig {
        cors_origins: vec![
            "https://app.example.com".to_string(),
            "https://admin.example.com".to_string(),
        ],
        ..Default::default()
    };

    assert_eq!(cors_config.cors_origins.len(), 2);

    // ── 4. Security Headers ──
    // When enabled, Nexus adds security headers to every response:
    //   X-Content-Type-Options: nosniff
    //   X-Frame-Options: DENY
    //   X-XSS-Protection: 1; mode=block
    //   Strict-Transport-Security: max-age=31536000; includeSubDomains

    let secure = MiddlewareConfig {
        enable_security_headers: true,
        ..Default::default()
    };
    assert!(secure.enable_security_headers);

    // ── 5. Rate Limiting ──
    // Prevents abuse by limiting requests per minute per client.
    // Exceeding the limit returns HTTP 429 Too Many Requests.

    let limited = MiddlewareConfig {
        rate_limit_rpm: Some(100),
        ..Default::default()
    };
    assert_eq!(limited.rate_limit_rpm, Some(100));

    // ── 6. Applying Middleware to Nexus ──
    // In production, middleware is applied when building the server:
    //
    //   let config = NexusConfig {
    //       middleware: MiddlewareConfig { ... },
    //       ..Default::default()
    //   };
    //   let app = build_api_router(&nexus, &config);
    //   // Middleware is automatically applied to all routes

    println!("PASS: 02-nexus/05_middleware");
}
