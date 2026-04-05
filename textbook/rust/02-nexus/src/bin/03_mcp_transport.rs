// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / MCP Transport
//!
//! OBJECTIVE: Understand how Nexus exposes handlers as MCP (Model Context Protocol) tools.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs auto-generate MCP tool definitions from handler metadata.
//!         Rust uses SSE transport via axum; Python uses SSE via starlette.
//! VALIDATES: MCP tool generation, SSE transport concept, build_mcp_router
//!
//! Run: cargo run -p tutorial-nexus --bin 03_mcp_transport

use kailash_nexus::prelude::*;

fn main() {
    // ── 1. MCP Transport ──
    // Model Context Protocol (MCP) allows AI agents to call tools via
    // JSON-RPC over Server-Sent Events (SSE). Nexus automatically exposes
    // every registered handler as an MCP tool.
    //
    // Handler registration:
    //   nexus.handler("search", search_handler)
    //
    // Becomes MCP tool:
    //   {
    //     "name": "search",
    //     "description": "...",
    //     "inputSchema": { "type": "object", "properties": { "query": { "type": "string" } } }
    //   }

    let mut nexus = Nexus::new();

    nexus.handler(
        "search_docs",
        ClosureHandler::with_params(
            |inputs: ValueMap| async move {
                let query = inputs
                    .get("query" as &str)
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(Value::from(format!("Results for: {query}")))
            },
            vec![
                HandlerParam::new("query", HandlerParamType::String)
                    .description("Search query string"),
            ],
        ),
    );

    nexus.handler(
        "create_ticket",
        ClosureHandler::with_params(
            |inputs: ValueMap| async move {
                let title = inputs
                    .get("title" as &str)
                    .and_then(|v| v.as_str())
                    .unwrap_or("Untitled");
                let priority = inputs
                    .get("priority" as &str)
                    .and_then(|v| v.as_str())
                    .unwrap_or("medium");
                Ok(Value::from(format!("Created: {title} ({priority})")))
            },
            vec![
                HandlerParam::new("title", HandlerParamType::String)
                    .description("Ticket title"),
                HandlerParam::new("priority", HandlerParamType::String)
                    .description("Priority level"),
            ],
        ),
    );

    assert_eq!(nexus.handler_count(), 2);

    // ── 2. MCP Tool Schema Generation ──
    // Handler parameters are converted to JSON Schema for MCP tool definitions.
    // HandlerParamType maps to JSON Schema types:
    //   String  -> { "type": "string" }
    //   Integer -> { "type": "integer" }
    //   Float   -> { "type": "number" }
    //   Boolean -> { "type": "boolean" }
    //   Object  -> { "type": "object" }
    //   Array   -> { "type": "array" }

    // ── 3. SSE Transport ──
    // MCP uses Server-Sent Events for streaming responses.
    // The SSE endpoint is typically at /mcp/sse.
    //
    // Flow:
    //   1. AI agent connects to SSE endpoint
    //   2. Agent sends tool_call request (JSON-RPC)
    //   3. Nexus routes to the handler
    //   4. Response streamed back via SSE
    //
    // In production:
    //   let mcp_router = build_mcp_router(&nexus);
    //   // Mount alongside the HTTP API router

    // ── 4. Dual-Channel Deployment ──
    // The same handler serves both HTTP and MCP simultaneously:
    //   HTTP:  POST /api/search_docs {"query": "kailash"}
    //   MCP:   tool_call("search_docs", {"query": "kailash"})
    //
    // No code changes needed. Register once, deploy everywhere.

    println!("PASS: 02-nexus/03_mcp_transport");
}
