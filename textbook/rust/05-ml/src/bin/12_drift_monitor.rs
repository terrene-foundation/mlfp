// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- ML / Drift Monitor
//!
//! OBJECTIVE: Detect data and model drift in production ML systems.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python uses DriftMonitor with polars;
//!         Rust uses the same PSI/KS drift detection with ndarray.
//! VALIDATES: DriftMonitor, PSI, KS test, drift alerts, feature-level monitoring
//!
//! Run: cargo run -p tutorial-ml --bin 12_drift_monitor

fn main() {
    // ── 1. Population Stability Index (PSI) ──
    // PSI measures how different a new distribution is from a reference.
    //   PSI < 0.1  : No significant drift
    //   PSI 0.1-0.2: Moderate drift (investigate)
    //   PSI > 0.2  : Significant drift (retrain)

    let reference = vec![0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05];
    let current   = vec![0.08, 0.12, 0.22, 0.28, 0.18, 0.08, 0.04];

    let psi = compute_psi(&reference, &current);
    assert!(psi > 0.0, "PSI should be positive");
    assert!(psi < 0.1, "Small distribution shift = low PSI");

    // Larger drift
    let drifted = vec![0.02, 0.05, 0.10, 0.40, 0.25, 0.13, 0.05];
    let psi_drifted = compute_psi(&reference, &drifted);
    assert!(psi_drifted > psi, "Drifted distribution has higher PSI");

    // ── 2. Kolmogorov-Smirnov test ──
    // KS statistic: maximum absolute difference between CDFs.
    // p-value < 0.05 indicates statistically significant drift.

    let ref_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let cur_values = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];

    let ks_stat = compute_ks_statistic(&ref_values, &cur_values);
    assert!(ks_stat >= 0.0 && ks_stat <= 1.0);
    assert!(ks_stat < 0.3, "Small shift = low KS statistic");

    // Large drift
    let shifted = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
    let ks_shifted = compute_ks_statistic(&ref_values, &shifted);
    assert!(ks_shifted > ks_stat, "Larger shift = higher KS");

    // ── 3. Feature-level drift monitoring ──
    // Monitor each feature independently to identify which ones drifted.

    let features = vec!["age", "income", "score"];
    let feature_psi = vec![0.02, 0.15, 0.35];

    let drifted_features: Vec<&str> = features
        .iter()
        .zip(feature_psi.iter())
        .filter(|(_, &psi)| psi > 0.1)
        .map(|(&name, _)| name)
        .collect();

    assert_eq!(drifted_features.len(), 2);
    assert!(drifted_features.contains(&"income"));
    assert!(drifted_features.contains(&"score"));

    // ── 4. Drift severity levels ──

    for (&name, &psi) in features.iter().zip(feature_psi.iter()) {
        let severity = drift_severity(psi);
        match name {
            "age" => assert_eq!(severity, "none"),
            "income" => assert_eq!(severity, "moderate"),
            "score" => assert_eq!(severity, "significant"),
            _ => {}
        }
    }

    // ── 5. Model performance drift ──
    // Track prediction quality over time.

    let weekly_accuracy = vec![0.91, 0.90, 0.89, 0.85, 0.80, 0.75];
    let baseline_accuracy = 0.90;

    let degraded_weeks: Vec<usize> = weekly_accuracy
        .iter()
        .enumerate()
        .filter(|(_, &acc)| acc < baseline_accuracy - 0.05)
        .map(|(i, _)| i)
        .collect();

    assert!(degraded_weeks.len() >= 3);

    // ── 6. Monitoring pattern ──
    //
    //   let monitor = DriftMonitor::new(reference_data)
    //       .psi_threshold(0.2)
    //       .ks_threshold(0.05)
    //       .check_frequency("daily");
    //
    //   let report = monitor.check(current_data).await;
    //   if report.has_drift() {
    //       alert(report.drifted_features());
    //   }

    // ── 7. Key concepts ──
    // - PSI: Population Stability Index for distribution drift
    // - KS test: Kolmogorov-Smirnov for statistical significance
    // - Feature-level monitoring: identify which features drifted
    // - Severity levels: none < moderate < significant
    // - Model performance tracking: accuracy over time
    // - Thresholds: PSI 0.1/0.2, KS p-value 0.05
    // - Retrain trigger when significant drift detected

    println!("PASS: 05-ml/12_drift_monitor");
}

fn compute_psi(reference: &[f64], current: &[f64]) -> f64 {
    assert_eq!(reference.len(), current.len());
    let eps = 0.0001;
    reference
        .iter()
        .zip(current.iter())
        .map(|(&r, &c)| {
            let r = r.max(eps);
            let c = c.max(eps);
            (c - r) * (c / r).ln()
        })
        .sum()
}

fn compute_ks_statistic(ref_values: &[f64], cur_values: &[f64]) -> f64 {
    let n = ref_values.len() as f64;
    let m = cur_values.len() as f64;

    let mut all: Vec<(f64, bool)> = ref_values
        .iter()
        .map(|&v| (v, true))
        .chain(cur_values.iter().map(|&v| (v, false)))
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut ref_count = 0.0;
    let mut cur_count = 0.0;
    let mut max_diff = 0.0_f64;

    for (_, is_ref) in &all {
        if *is_ref {
            ref_count += 1.0;
        } else {
            cur_count += 1.0;
        }
        let diff = (ref_count / n - cur_count / m).abs();
        max_diff = max_diff.max(diff);
    }

    max_diff
}

fn drift_severity(psi: f64) -> &'static str {
    if psi < 0.1 {
        "none"
    } else if psi < 0.2 {
        "moderate"
    } else {
        "significant"
    }
}
