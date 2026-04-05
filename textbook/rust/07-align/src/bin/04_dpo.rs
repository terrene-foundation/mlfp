// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Align / Direct Preference Optimization (DPO)
//!
//! OBJECTIVE: Understand DPO — aligning models from human preferences without
//!            a separate reward model, using the Bradley-Terry preference model.
//! LEVEL: Advanced
//! PARITY: Pending kailash-rs implementation — demonstrates DPO math with Rust idioms.
//!         Python uses AlignmentPipeline with method="dpo".
//! VALIDATES: Bradley-Terry model, DPO loss derivation, preference data format, beta parameter
//!
//! Run: cargo run -p tutorial-align --bin 04_dpo

use serde::{Deserialize, Serialize};

fn main() {
    // ── 1. Preference data format ──
    // DPO trains on preference pairs: given a prompt, one response is
    // "chosen" (preferred) and one is "rejected" (dispreferred).

    let preferences = vec![
        PreferencePair {
            prompt: "Explain quantum entanglement simply.".into(),
            chosen: "When two particles are entangled, measuring one instantly \
                     reveals information about the other, regardless of distance.".into(),
            rejected: "Quantum entanglement is a non-local quantum mechanical \
                       phenomenon whereby the quantum state of each particle \
                       cannot be described independently.".into(),
        },
        PreferencePair {
            prompt: "What is compound interest?".into(),
            chosen: "Interest earned on both your original amount and previously \
                     earned interest — your money grows faster over time.".into(),
            rejected: "Compound interest is the addition of interest to the \
                       principal sum of a loan or deposit.".into(),
        },
    ];

    assert_eq!(preferences.len(), 2);
    for pair in &preferences {
        assert!(!pair.prompt.is_empty());
        assert!(!pair.chosen.is_empty());
        assert!(!pair.rejected.is_empty());
        assert_ne!(pair.chosen, pair.rejected);
    }

    // ── 2. Bradley-Terry preference model ──
    // The probability that response y_w is preferred over y_l:
    //
    //   P(y_w > y_l) = sigma(r(x, y_w) - r(x, y_l))
    //
    // where sigma is the sigmoid function and r is the reward.
    // Higher reward difference = stronger preference.

    let reward_chosen = 2.5_f64;
    let reward_rejected = 0.8_f64;
    let diff = reward_chosen - reward_rejected;
    let preference_prob = sigmoid(diff);

    assert!(preference_prob > 0.5, "Chosen should be preferred");
    assert!(preference_prob > 0.8, "Strong preference with large gap");

    // Equal rewards = 50/50 preference
    let equal_prob = sigmoid(0.0);
    assert!((equal_prob - 0.5).abs() < 1e-10);

    // ── 3. DPO loss function ──
    // DPO's key insight: the optimal reward function is:
    //   r*(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log Z(x)
    //
    // The DPO loss (substituting into Bradley-Terry):
    //   L_DPO = -log sigma(beta * (log(pi(y_w|x)/pi_ref(y_w|x))
    //                             - log(pi(y_l|x)/pi_ref(y_l|x))))
    //
    // This eliminates the need for a separate reward model.

    let beta = 0.1_f64;

    // Log-probability ratios (policy vs reference)
    let log_ratio_chosen = -0.5_f64;  // pi/pi_ref for chosen
    let log_ratio_rejected = -1.2_f64; // pi/pi_ref for rejected

    let dpo_logit = beta * (log_ratio_chosen - log_ratio_rejected);
    let dpo_loss = -sigmoid(dpo_logit).ln();

    assert!(dpo_loss > 0.0, "Loss is always positive");
    assert!(dpo_loss.is_finite(), "Loss should be finite");

    // ── 4. Beta parameter ──
    // Beta controls how much the model can deviate from the reference:
    //   beta=0.01: Very conservative (stay close to reference)
    //   beta=0.1:  Standard (default, good balance)
    //   beta=0.5:  Aggressive (stronger preference optimization)
    //   beta=1.0:  Maximum (risk of reward hacking)

    let betas = vec![
        BetaConfig { beta: 0.01, description: "conservative" },
        BetaConfig { beta: 0.1,  description: "standard" },
        BetaConfig { beta: 0.5,  description: "aggressive" },
    ];

    // Higher beta amplifies the log-ratio difference
    let losses: Vec<f64> = betas
        .iter()
        .map(|b| {
            let logit = b.beta * (log_ratio_chosen - log_ratio_rejected);
            -sigmoid(logit).ln()
        })
        .collect();

    // Lower beta = loss closer to -log(0.5) = log(2) (less differentiation)
    assert!(losses[0] > losses[1], "Lower beta = weaker signal");
    assert!(losses[1] > losses[2], "Higher beta = stronger signal");

    // ── 5. DPO vs RLHF ──
    // Traditional RLHF pipeline:
    //   1. SFT (supervised fine-tuning)
    //   2. Train reward model on preferences
    //   3. PPO optimization against reward model
    //
    // DPO pipeline (simpler):
    //   1. SFT (supervised fine-tuning)
    //   2. DPO directly on preference pairs (no reward model needed)
    //
    // DPO advantages: simpler, more stable, no reward model to maintain.
    // DPO limitations: needs good reference model, sensitive to data quality.

    let rlhf_steps = vec!["SFT", "Train reward model", "PPO optimization"];
    let dpo_steps = vec!["SFT", "DPO on preferences"];
    assert!(dpo_steps.len() < rlhf_steps.len());

    // ── 6. Implicit reward ──
    // After DPO training, the implicit reward for any response is:
    //   r(x, y) = beta * log(pi_dpo(y|x) / pi_ref(y|x))
    //
    // We can extract this to score new responses without a separate model.

    let implicit_rewards: Vec<f64> = vec![
        beta * 0.8,   // Response A: policy favors over reference
        beta * (-0.3), // Response B: reference slightly better
        beta * 1.5,    // Response C: policy strongly favors
    ];

    // Response C is most aligned with learned preferences
    let best_idx = implicit_rewards
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    assert_eq!(best_idx, 2);

    // ── 7. Data quality for DPO ──
    // DPO is sensitive to preference quality:
    //   - Clear quality gap between chosen/rejected
    //   - Consistent annotation guidelines
    //   - Enough diversity in prompts
    //   - Balanced difficulty (not all trivial preferences)

    for pair in &preferences {
        // Chosen and rejected must differ meaningfully
        let chosen_words: Vec<&str> = pair.chosen.split_whitespace().collect();
        let rejected_words: Vec<&str> = pair.rejected.split_whitespace().collect();
        assert!(chosen_words.len() > 3, "Responses must be substantive");
        assert!(rejected_words.len() > 3, "Responses must be substantive");
    }

    // ── 8. Key concepts ──
    // - DPO: align from preferences without a reward model
    // - Bradley-Terry: P(y_w > y_l) = sigma(r_w - r_l)
    // - DPO loss: uses log-ratio of policy vs reference model
    // - Beta: KL penalty strength (0.1 default, controls deviation)
    // - Implicit reward: beta * log(pi/pi_ref) after training
    // - Simpler than RLHF: no reward model, no PPO instability
    // - Data quality critical: clear preference gaps, diverse prompts

    println!("PASS: 07-align/04_dpo");
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreferencePair {
    prompt: String,
    chosen: String,
    rejected: String,
}

struct BetaConfig {
    beta: f64,
    description: &'static str,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
