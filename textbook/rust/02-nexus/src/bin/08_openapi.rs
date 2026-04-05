// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / OpenAPI Spec Generation
//!
//! OBJECTIVE: Auto-generate OpenAPI documentation from registered Nexus handlers.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python uses OpenApiGenerator with add_handler/add_workflow;
//!         Rust uses the same OpenApiGenerator builder with typed info and server entries.
//! VALIDATES: OpenApiGenerator, OpenApiInfo, add_handler, add_workflow, generate, generate_json
//!
//! Run: cargo run -p tutorial-nexus --bin 08_openapi

use kailash_nexus::prelude::*;

fn main() {
    // ── 1. OpenApiInfo ──
    // OpenApiInfo configures the info section of the OpenAPI spec.
    // Fields mirror the OpenAPI 3.0.3 Info Object.

    let info = OpenApiInfo::new("My ML Platform API", "2.0.0")
        .description("Production ML inference endpoints")
        .contact_name("Platform Team")
        .contact_email("platform@example.com")
        .license("Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0")
        .terms_of_service("https://example.com/terms");

    assert_eq!(info.title(), "My ML Platform API");
    assert_eq!(info.version(), "2.0.0");

    // ── 2. Default OpenApiInfo ──
    // Defaults are Kailash-branded with Apache-2.0 license.

    let defaults = OpenApiInfo::default();
    assert_eq!(defaults.title(), "Kailash Nexus API");
    assert_eq!(defaults.version(), "1.0.0");

    // ── 3. OpenApiGenerator with servers ──
    // The generator builds an OpenAPI 3.0.3 spec from registered handlers.

    let mut gen = OpenApiGenerator::new(info)
        .server("https://api.example.com", "Production")
        .server("http://localhost:8000", "Development");

    // ── 4. Register a handler ──
    // add_handler() registers a Nexus handler as an OpenAPI endpoint.
    // Parameters are described as JSON Schema properties with types.

    gen.add_handler(
        "predict",
        "Run model inference",
        &["inference"],
        &[
            ("model_name", "string", true),
            ("threshold", "number", false),
        ],
    );

    // ── 5. Register another handler ──

    gen.add_handler(
        "health",
        "Check service health",
        &["system"],
        &[("service", "string", false)],
    );

    // ── 6. Register a workflow ──
    // add_workflow() adds both execute and info endpoints.

    gen.add_workflow(
        "greet",
        "Greet a user by name",
        &["greetings"],
        &[("name", "string", true)],
    );

    // ── 7. Generate the spec ──
    // generate() returns the spec as a serde_json::Value.

    let spec = gen.generate();

    assert_eq!(spec["openapi"], "3.0.3");
    assert_eq!(spec["info"]["title"], "My ML Platform API");
    assert_eq!(spec["info"]["version"], "2.0.0");
    assert_eq!(spec["info"]["license"]["name"], "Apache-2.0");
    assert_eq!(spec["info"]["contact"]["name"], "Platform Team");

    // Servers are included
    let servers = spec["servers"].as_array().unwrap();
    assert_eq!(servers.len(), 2);
    assert_eq!(servers[0]["url"], "https://api.example.com");

    // ── 8. Verify handler paths ──
    // Each handler creates a POST endpoint at /workflows/{name}/execute.

    let paths = &spec["paths"];
    assert!(paths["/workflows/predict/execute"]["post"].is_object());
    assert!(paths["/workflows/health/execute"]["post"].is_object());

    let predict_op = &paths["/workflows/predict/execute"]["post"];
    assert_eq!(predict_op["operationId"], "execute_predict");

    // ── 9. Verify workflow paths ──
    // Workflows get both execute and info endpoints.

    assert!(paths["/workflows/greet/execute"]["post"].is_object());
    assert!(paths["/workflows/greet/workflow/info"]["get"].is_object());

    let greet_info = &paths["/workflows/greet/workflow/info"]["get"];
    assert_eq!(greet_info["operationId"], "info_greet");

    // ── 10. Verify schemas ──
    // Handler schemas list properties and required fields.

    let schemas = &spec["components"]["schemas"];
    let predict_schema = &schemas["predict_input"];
    assert_eq!(predict_schema["type"], "object");
    assert_eq!(predict_schema["properties"]["model_name"]["type"], "string");
    assert_eq!(predict_schema["properties"]["threshold"]["type"], "number");

    // model_name is required; threshold is optional
    let required = predict_schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(required.contains(&"model_name"));
    assert!(!required.contains(&"threshold"));

    // ── 11. generate_json() ──
    // Convenience method returns formatted JSON string.

    let json_str = gen.generate_json(2);
    assert!(json_str.contains("\"openapi\": \"3.0.3\""));
    assert!(json_str.contains("My ML Platform API"));

    // ── 12. Type mapping (Rust perspective) ──
    // Rust types map to OpenAPI types:
    //   String / &str -> string
    //   i32 / i64     -> integer
    //   f32 / f64     -> number
    //   bool          -> boolean
    //   Vec<T>        -> array
    //   HashMap / struct -> object
    //   Vec<u8>       -> string (format: binary)

    // ── 13. Key concepts ──
    // - OpenApiInfo: title, version, description, contact, license
    // - OpenApiGenerator::new(info).server(url, desc): builds spec
    // - add_handler(): register handler with parameter schema
    // - add_workflow(): register workflow with execute + info endpoints
    // - generate() -> serde_json::Value
    // - generate_json(indent) -> String
    // - Handlers create POST /workflows/{name}/execute
    // - Workflows also get GET /workflows/{name}/workflow/info

    println!("PASS: 02-nexus/08_openapi");
}
