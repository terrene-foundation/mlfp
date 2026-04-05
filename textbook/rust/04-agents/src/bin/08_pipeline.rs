// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / Pipeline Routing
//!
//! OBJECTIVE: Build agent pipelines that route tasks through sequential stages.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python has Pipeline.router() and Pipeline.sequential();
//!         Rust uses PipelineConfig with the same routing and chaining patterns.
//! VALIDATES: PipelineConfig, PipelineStage, router, sequential, parallel fan-out
//!
//! Run: cargo run -p tutorial-agents --bin 08_pipeline

use kaizen_agents::pipeline::{
    PipelineConfig, PipelineStage, RoutingStrategy,
};

fn main() {
    // ── 1. Pipeline pattern ──
    // Pipelines compose multiple agents into a processing chain.
    // Each stage transforms or enriches the data before passing
    // it to the next stage.

    // ── 2. Pipeline stages ──
    // Each stage has a name, agent reference, and optional transform.

    let classify = PipelineStage::new("classify")
        .agent("classifier-agent")
        .description("Classify the input into a category");

    let enrich = PipelineStage::new("enrich")
        .agent("enrichment-agent")
        .description("Add metadata and context to the classified input");

    let respond = PipelineStage::new("respond")
        .agent("response-agent")
        .description("Generate the final response");

    assert_eq!(classify.name(), "classify");
    assert_eq!(enrich.name(), "enrich");
    assert_eq!(respond.name(), "respond");

    // ── 3. Sequential pipeline ──
    // Stages execute in order. Output of one feeds into the next.

    let sequential = PipelineConfig::new("support-pipeline")
        .strategy(RoutingStrategy::Sequential)
        .stage(classify)
        .stage(enrich)
        .stage(respond);

    assert_eq!(sequential.name(), "support-pipeline");
    assert_eq!(sequential.stage_count(), 3);
    assert!(matches!(
        sequential.strategy(),
        RoutingStrategy::Sequential
    ));

    // ── 4. Router pipeline ──
    // The router uses LLM reasoning to select which stage to execute.
    // Only ONE stage runs per invocation.

    let billing_stage = PipelineStage::new("billing")
        .agent("billing-agent")
        .description("Handle billing inquiries and refunds");

    let technical_stage = PipelineStage::new("technical")
        .agent("tech-agent")
        .description("Handle technical support issues");

    let sales_stage = PipelineStage::new("sales")
        .agent("sales-agent")
        .description("Handle sales inquiries and upselling");

    let router = PipelineConfig::new("support-router")
        .strategy(RoutingStrategy::Router)
        .stage(billing_stage)
        .stage(technical_stage)
        .stage(sales_stage);

    assert_eq!(router.stage_count(), 3);
    assert!(matches!(router.strategy(), RoutingStrategy::Router));

    // CRITICAL: Router uses LLM-based routing, NOT keyword matching.
    // The LLM examines stage descriptions to decide routing.

    // ── 5. Parallel fan-out ──
    // All stages execute concurrently. Results are aggregated.

    let sentiment = PipelineStage::new("sentiment")
        .agent("sentiment-agent")
        .description("Analyze sentiment of the input");

    let entities = PipelineStage::new("entities")
        .agent("entity-agent")
        .description("Extract named entities");

    let summary = PipelineStage::new("summary")
        .agent("summary-agent")
        .description("Produce a brief summary");

    let parallel = PipelineConfig::new("analysis-pipeline")
        .strategy(RoutingStrategy::Parallel)
        .stage(sentiment)
        .stage(entities)
        .stage(summary);

    assert_eq!(parallel.stage_count(), 3);
    assert!(matches!(parallel.strategy(), RoutingStrategy::Parallel));

    // ── 6. Pipeline execution pattern ──
    //
    //   let pipeline = Pipeline::new(sequential_config, agents).await;
    //   let result = pipeline.run("My order hasn't arrived").await;
    //   // classify -> enrich -> respond (sequential)
    //
    //   let router_pipeline = Pipeline::new(router_config, agents).await;
    //   let result = router_pipeline.run("I need a refund").await;
    //   // LLM routes to billing-agent
    //
    // NOTE: We do not call run() here as it requires an LLM API key.

    // ── 7. Key concepts ──
    // - Pipeline: composes agents into processing chains
    // - PipelineStage: individual step with agent and description
    // - RoutingStrategy::Sequential: stages run in order
    // - RoutingStrategy::Router: LLM picks one stage
    // - RoutingStrategy::Parallel: all stages run concurrently
    // - Router uses LLM reasoning, NOT keyword/dispatch tables
    // - Stage descriptions are capability cards for routing decisions

    println!("PASS: 04-agents/08_pipeline");
}
