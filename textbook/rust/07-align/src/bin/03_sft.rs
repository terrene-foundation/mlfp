// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Align / Supervised Fine-Tuning (SFT)
//!
//! OBJECTIVE: Understand the supervised fine-tuning pipeline — data preparation,
//!            training loop, and loss computation for instruction-following models.
//! LEVEL: Intermediate
//! PARITY: Pending kailash-rs implementation — demonstrates SFT concepts with Rust idioms.
//!         Python uses AlignmentPipeline with method="sft".
//! VALIDATES: SFT data format, training pipeline stages, loss masking, chat templates
//!
//! Run: cargo run -p tutorial-align --bin 03_sft

use kailash_align_serving::adapter::{AdapterMetadata, TrainingMethod};
use serde::{Deserialize, Serialize};

fn main() {
    // ── 1. SFT data format ──
    // Supervised fine-tuning uses instruction/response pairs.
    // The model learns to generate the response given the instruction.

    let training_examples = vec![
        SftExample {
            instruction: "Summarize the quarterly earnings report.".into(),
            input: "Revenue grew 15% YoY to $2.3B. Operating margin expanded \
                    200bps to 28%. EPS beat consensus by $0.12.".into(),
            output: "Strong quarter: 15% revenue growth to $2.3B with margin \
                     expansion and an EPS beat.".into(),
        },
        SftExample {
            instruction: "Classify the sentiment of this review.".into(),
            input: "The product works well but shipping took three weeks.".into(),
            output: "Mixed sentiment: positive on product quality, \
                     negative on delivery speed.".into(),
        },
    ];

    assert_eq!(training_examples.len(), 2);
    assert!(!training_examples[0].instruction.is_empty());
    assert!(!training_examples[0].output.is_empty());

    // ── 2. Chat template format ──
    // Modern LLMs use chat templates with role markers.
    // SFT data is converted to this format before training.

    let chat_messages = vec![
        ChatMessage { role: "system".into(), content: "You are a financial analyst.".into() },
        ChatMessage { role: "user".into(), content: "Summarize the earnings report.".into() },
        ChatMessage { role: "assistant".into(), content: "Strong quarter with growth.".into() },
    ];

    let formatted = format_chat_template(&chat_messages);
    assert!(formatted.contains("<|system|>"));
    assert!(formatted.contains("<|user|>"));
    assert!(formatted.contains("<|assistant|>"));

    // ── 3. Loss masking ──
    // During SFT, loss is computed ONLY on the assistant's response tokens.
    // The instruction/system tokens are masked (not penalized).
    //
    // Token: [<|system|>, You, are, ..., <|user|>, Summarize, ..., <|assistant|>, Strong, ...]
    // Mask:  [   -100,  -100,-100,...,   -100,    -100,  ...,     -100,          1,     1, ...]
    //
    // -100 = ignored by cross-entropy loss, 1 = supervised

    let tokens = vec!["<|system|>", "You", "are", "<|user|>", "Summarize",
                      "<|assistant|>", "Strong", "quarter"];
    let mask = create_loss_mask(&tokens, "<|assistant|>");

    // Only tokens after <|assistant|> should be trained on
    assert_eq!(mask, vec![false, false, false, false, false, false, true, true]);
    let trained_tokens = mask.iter().filter(|&&m| m).count();
    assert_eq!(trained_tokens, 2);

    // ── 4. Training pipeline stages ──
    // SFT pipeline:
    //   1. Load base model
    //   2. Attach LoRA adapters to target modules
    //   3. Prepare dataset (tokenize, apply chat template, create masks)
    //   4. Training loop (forward pass, loss on response tokens, backward pass)
    //   5. Save adapter weights

    let pipeline = SftPipeline {
        base_model: "llama-3-8b".into(),
        adapter_config: AdapterMetadata {
            name: "finance-sft-v1".into(),
            method: TrainingMethod::Lora,
            rank: 16,
            alpha: 32.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            base_model: "llama-3-8b".into(),
            ..Default::default()
        },
        learning_rate: 2e-4,
        num_epochs: 3,
        batch_size: 4,
        max_seq_length: 2048,
        warmup_ratio: 0.03,
    };

    assert_eq!(pipeline.base_model, "llama-3-8b");
    assert_eq!(pipeline.num_epochs, 3);
    assert_eq!(pipeline.adapter_config.rank, 16);

    // ── 5. Training metrics ──
    // Track loss per step to verify convergence.

    let step_losses = vec![2.8, 2.5, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.25, 1.2];
    let early_loss: f64 = step_losses[..3].iter().sum::<f64>() / 3.0;
    let late_loss: f64 = step_losses[7..].iter().sum::<f64>() / 3.0;

    assert!(late_loss < early_loss, "Loss should decrease during training");
    assert!(late_loss < 1.5, "Final loss should be reasonable");

    // ── 6. Sequence packing ──
    // Short examples waste GPU compute on padding. Sequence packing
    // concatenates multiple examples into one sequence (separated by
    // special tokens), maximizing GPU utilization.
    //
    //   Without packing: [ex1, PAD, PAD, PAD] [ex2, PAD, PAD] [ex3, PAD, ...]
    //   With packing:    [ex1, <sep>, ex2, <sep>, ex3, ...]
    //
    // Typical efficiency gain: 2-3x faster training.

    let example_lengths = vec![128, 256, 512, 64, 192, 384];
    let max_seq = 2048;
    let packed_seqs = pack_sequences(&example_lengths, max_seq);
    let total_examples: usize = packed_seqs.iter().map(|p| p.len()).sum();
    assert_eq!(total_examples, example_lengths.len());

    // Packing should use fewer sequences than unpacked
    assert!(packed_seqs.len() < example_lengths.len());

    // ── 7. Data quality checks ──
    // Before training, validate the dataset:
    //   - No empty instructions or outputs
    //   - Reasonable token lengths
    //   - No duplicates
    //   - Output follows instruction format

    for example in &training_examples {
        assert!(!example.instruction.is_empty(), "Instruction must not be empty");
        assert!(!example.output.is_empty(), "Output must not be empty");
        assert!(example.output.len() < 10_000, "Output must be reasonable length");
    }

    // ── 8. Key concepts ──
    // - SFT: train model to follow instructions using (instruction, response) pairs
    // - Chat template: role markers (<|system|>, <|user|>, <|assistant|>)
    // - Loss masking: only compute loss on response tokens, not instruction
    // - Sequence packing: concatenate short examples to maximize GPU utilization
    // - Pipeline: load model -> attach LoRA -> prepare data -> train -> save
    // - Monitor loss per step to verify convergence
    // - Validate data quality before training (no empty/duplicate examples)

    println!("PASS: 07-align/03_sft");
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SftExample {
    instruction: String,
    input: String,
    output: String,
}

#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

struct SftPipeline {
    base_model: String,
    adapter_config: AdapterMetadata,
    learning_rate: f64,
    num_epochs: u32,
    batch_size: u32,
    max_seq_length: u32,
    warmup_ratio: f64,
}

fn format_chat_template(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("<|{}|>\n{}\n", m.role, m.content))
        .collect::<Vec<_>>()
        .join("")
}

fn create_loss_mask(tokens: &[&str], assistant_marker: &str) -> Vec<bool> {
    let mut found_assistant = false;
    tokens
        .iter()
        .map(|&token| {
            if token == assistant_marker {
                found_assistant = true;
                false // The marker itself is not trained on
            } else {
                found_assistant
            }
        })
        .collect()
}

/// Greedy bin-packing: fit examples into sequences of max_length.
fn pack_sequences(lengths: &[usize], max_length: usize) -> Vec<Vec<usize>> {
    let mut bins: Vec<(usize, Vec<usize>)> = Vec::new(); // (remaining, indices)

    for (idx, &len) in lengths.iter().enumerate() {
        // Find first bin with enough space
        let fit = bins.iter_mut().find(|(remaining, _)| *remaining >= len);
        match fit {
            Some((remaining, indices)) => {
                *remaining -= len;
                indices.push(idx);
            }
            None => {
                bins.push((max_length - len, vec![idx]));
            }
        }
    }

    bins.into_iter().map(|(_, indices)| indices).collect()
}
