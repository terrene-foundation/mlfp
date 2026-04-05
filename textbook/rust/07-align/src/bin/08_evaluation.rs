// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Align / Evaluation Strategies
//!
//! OBJECTIVE: Evaluate fine-tuned model quality — automated metrics, LLM-as-judge,
//!            and human evaluation frameworks.
//! LEVEL: Intermediate
//! PARITY: Pending kailash-rs implementation — demonstrates evaluation concepts
//!         with Rust idioms. Python kailash-align provides EvaluationPipeline.
//! VALIDATES: Perplexity, ROUGE, LLM-as-judge, win rate, evaluation rubrics
//!
//! Run: cargo run -p tutorial-align --bin 08_evaluation

fn main() {
    // ── 1. Perplexity ──
    // Perplexity measures how well the model predicts a held-out test set.
    // Lower perplexity = better language modeling.
    //
    //   PPL = exp(-1/N * sum(log P(token_i | context)))
    //
    // Useful for comparing base vs fine-tuned on domain text.

    let log_probs = vec![-2.0_f64, -1.5, -1.8, -2.2, -1.3, -1.9, -1.6, -2.1];
    let n = log_probs.len() as f64;
    let avg_neg_log_prob = -log_probs.iter().sum::<f64>() / n;
    let perplexity = avg_neg_log_prob.exp();

    assert!(perplexity > 1.0, "Perplexity is always >= 1");
    assert!(perplexity < 20.0, "Good domain model should have low perplexity");

    // Compare base vs fine-tuned
    let base_ppl = 15.3;
    let finetuned_ppl = perplexity;
    assert!(finetuned_ppl < base_ppl, "Fine-tuned should have lower perplexity");

    // ── 2. ROUGE scores ──
    // Measures overlap between generated and reference text.
    //   ROUGE-1: unigram overlap
    //   ROUGE-2: bigram overlap
    //   ROUGE-L: longest common subsequence

    let reference = "The quarterly revenue increased by fifteen percent year over year";
    let generated = "Revenue grew fifteen percent compared to last year";

    let rouge1 = rouge_1_f1(reference, generated);
    assert!(rouge1 > 0.0 && rouge1 <= 1.0, "ROUGE-1 is in [0, 1]");

    // Higher ROUGE = more overlap with reference
    let good_gen = "The quarterly revenue increased fifteen percent year over year";
    let good_rouge = rouge_1_f1(reference, good_gen);
    assert!(good_rouge > rouge1, "Better match = higher ROUGE");

    // ── 3. LLM-as-judge ──
    // Use a strong LLM to evaluate response quality.
    // The judge scores or ranks responses according to a rubric.

    let eval_prompt = EvalPrompt {
        system: "You are an impartial judge. Score the response on a scale of 1-5.".into(),
        rubric: vec![
            "1: Incorrect or irrelevant",
            "2: Partially correct with major issues",
            "3: Correct but could be improved",
            "4: Good, clear, and accurate",
            "5: Excellent, comprehensive, and well-structured",
        ],
        question: "What is compound interest?".into(),
        response: "Interest earned on both principal and previously earned interest.".into(),
    };

    assert!(!eval_prompt.rubric.is_empty());
    assert_eq!(eval_prompt.rubric.len(), 5);

    // ── 4. Pairwise comparison (win rate) ──
    // Compare two models head-to-head on the same prompts.
    // A judge picks the winner for each prompt.

    let comparisons = vec![
        Comparison { prompt: "Explain gravity".into(), winner: "model_b".into() },
        Comparison { prompt: "Summarize article".into(), winner: "model_a".into() },
        Comparison { prompt: "Write a poem".into(), winner: "model_b".into() },
        Comparison { prompt: "Debug this code".into(), winner: "model_a".into() },
        Comparison { prompt: "Translate text".into(), winner: "model_b".into() },
        Comparison { prompt: "Analyze data".into(), winner: "tie".into() },
    ];

    let wins_a = comparisons.iter().filter(|c| c.winner == "model_a").count();
    let wins_b = comparisons.iter().filter(|c| c.winner == "model_b").count();
    let ties = comparisons.iter().filter(|c| c.winner == "tie").count();
    let total = comparisons.len();

    let win_rate_a = wins_a as f64 / total as f64;
    let win_rate_b = wins_b as f64 / total as f64;

    assert_eq!(wins_a + wins_b + ties, total);
    assert!(win_rate_b > win_rate_a, "Model B wins more");

    // ── 5. Multi-dimensional evaluation ──
    // Real evaluation needs multiple dimensions, not just overall quality.

    let eval_dimensions = vec![
        EvalDimension { name: "Accuracy".into(), weight: 0.30, score: 4.2 },
        EvalDimension { name: "Helpfulness".into(), weight: 0.25, score: 4.5 },
        EvalDimension { name: "Safety".into(), weight: 0.20, score: 4.8 },
        EvalDimension { name: "Conciseness".into(), weight: 0.15, score: 3.9 },
        EvalDimension { name: "Formatting".into(), weight: 0.10, score: 4.0 },
    ];

    let total_weight: f64 = eval_dimensions.iter().map(|d| d.weight).sum();
    assert!((total_weight - 1.0).abs() < 1e-10, "Weights must sum to 1.0");

    let weighted_score: f64 = eval_dimensions.iter()
        .map(|d| d.weight * d.score)
        .sum();
    assert!(weighted_score > 3.0 && weighted_score < 5.0);

    // ── 6. Position bias in LLM judges ──
    // LLM judges have known biases:
    //   - Position bias: preference for first or second response
    //   - Verbosity bias: preference for longer responses
    //   - Self-enhancement: preference for own model's style
    //
    // Mitigation: swap response order and average results.

    let score_a_first = 4.2_f64; // Model A shown first
    let score_a_second = 3.8_f64; // Model A shown second
    let debiased_score = (score_a_first + score_a_second) / 2.0;
    let position_bias = (score_a_first - score_a_second).abs();

    assert!((debiased_score - 4.0).abs() < 1e-10);
    assert!(position_bias > 0.0, "Position bias exists");
    assert!(position_bias < 1.0, "Bias should be moderate");

    // ── 7. Statistical significance ──
    // With small sample sizes, results may not be significant.
    // Rule of thumb: need 100+ comparisons per evaluation pair.

    let sample_size = comparisons.len();
    let min_for_significance = 100;
    let is_significant = sample_size >= min_for_significance;
    assert!(!is_significant, "6 comparisons is not statistically significant");

    // Standard error for win rate
    let p = win_rate_b;
    let se = (p * (1.0 - p) / sample_size as f64).sqrt();
    assert!(se > 0.1, "High uncertainty with small sample");

    // ── 8. Evaluation pipeline ──
    // Complete evaluation:
    //   1. Automated metrics (perplexity, ROUGE) — fast, cheap
    //   2. LLM-as-judge on held-out prompts — moderate cost
    //   3. Pairwise comparison against baseline — moderate cost
    //   4. Human evaluation on critical cases — expensive, gold standard
    //
    // Only proceed to expensive stages if automated metrics pass.

    let pipeline = EvalPipeline {
        stages: vec![
            EvalStage { name: "Perplexity".into(), automated: true, pass_threshold: "PPL < 10".into() },
            EvalStage { name: "ROUGE".into(), automated: true, pass_threshold: "ROUGE-L > 0.3".into() },
            EvalStage { name: "LLM Judge".into(), automated: true, pass_threshold: "Score > 3.5/5".into() },
            EvalStage { name: "Win Rate".into(), automated: true, pass_threshold: "Win > 50%".into() },
            EvalStage { name: "Human Eval".into(), automated: false, pass_threshold: "Approval > 80%".into() },
        ],
    };

    assert_eq!(pipeline.stages.len(), 5);
    let automated_count = pipeline.stages.iter().filter(|s| s.automated).count();
    assert_eq!(automated_count, 4);

    // ── 9. Key concepts ──
    // - Perplexity: how well model predicts held-out text (lower = better)
    // - ROUGE: reference-based overlap metric (unigram, bigram, LCS)
    // - LLM-as-judge: use strong LLM to score responses against a rubric
    // - Win rate: pairwise comparison, count which model wins more
    // - Multi-dimensional: evaluate accuracy, helpfulness, safety separately
    // - Position bias: swap order and average to debias LLM judges
    // - Statistical significance: need 100+ comparisons for reliable results
    // - Pipeline: automated metrics first, human eval only for finals

    println!("PASS: 07-align/08_evaluation");
}

struct EvalPrompt {
    system: String,
    rubric: Vec<&'static str>,
    question: String,
    response: String,
}

struct Comparison {
    prompt: String,
    winner: String,
}

struct EvalDimension {
    name: String,
    weight: f64,
    score: f64,
}

struct EvalStage {
    name: String,
    automated: bool,
    pass_threshold: String,
}

struct EvalPipeline {
    stages: Vec<EvalStage>,
}

/// Simple ROUGE-1 F1 score based on unigram overlap.
fn rouge_1_f1(reference: &str, generated: &str) -> f64 {
    let ref_lower = reference.to_lowercase();
    let gen_lower = generated.to_lowercase();

    let ref_words: std::collections::HashSet<&str> = ref_lower
        .split_whitespace()
        .collect();
    let gen_words: std::collections::HashSet<&str> = gen_lower
        .split_whitespace()
        .collect();

    let overlap = ref_words.intersection(&gen_words).count() as f64;
    let precision = if gen_words.is_empty() { 0.0 } else { overlap / gen_words.len() as f64 };
    let recall = if ref_words.is_empty() { 0.0 } else { overlap / ref_words.len() as f64 };

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}
