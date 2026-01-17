# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:56:10 2026

@author: Firas.Alhawari
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# -----------------------------
# Configuration
# -----------------------------
BASELINE_FILE = "chatbot_eval_review_pure_llms_results.csv"
BEST_FILE = "chatbot_eval_review_hybrid_only_with_trans_results.csv"
MODEL_NAME = "Qwen"

N_BOOTSTRAP = 10000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -----------------------------
# Load data
# -----------------------------
baseline = pd.read_csv(BASELINE_FILE)
best = pd.read_csv(BEST_FILE)

# -----------------------------
# Filter Qwen only
# -----------------------------
baseline = baseline[baseline["model"].str.contains(MODEL_NAME, na=False)]
best = best[best["model"].str.contains(MODEL_NAME, na=False)]

# -----------------------------
# Compute mean human score per row
# -----------------------------
baseline["mean_score"] = baseline[["reviewer1_score", "reviewer2_score"]].mean(axis=1)
best["mean_score"] = best[["reviewer1_score", "reviewer2_score"]].mean(axis=1)

# -----------------------------
# Aggregate across runs per question
# (3 runs → one score per subrun_id)
# -----------------------------
baseline_agg = (
    baseline
    .groupby("subrun_id", as_index=False)["mean_score"]
    .mean()
    .rename(columns={"mean_score": "baseline_score"})
)

best_agg = (
    best
    .groupby("subrun_id", as_index=False)["mean_score"]
    .mean()
    .rename(columns={"mean_score": "best_score"})
)

# -----------------------------
# Pair questions
# -----------------------------
paired = pd.merge(
    baseline_agg,
    best_agg,
    on="subrun_id",
    how="inner"
)

if paired.empty:
    raise ValueError("No paired questions found. Check subrun_id alignment.")

# -----------------------------
# Extract paired arrays
# -----------------------------
baseline_scores = paired["baseline_score"].values
best_scores = paired["best_score"].values
diff = best_scores - baseline_scores

# -----------------------------
# Descriptive statistics
# -----------------------------
mean_baseline = baseline_scores.mean()
mean_best = best_scores.mean()
mean_diff = diff.mean()

# -----------------------------
# Bootstrap CI for mean difference
# -----------------------------
boot_means = []
n = len(diff)

for _ in range(N_BOOTSTRAP):
    sample = np.random.choice(diff, size=n, replace=True)
    boot_means.append(sample.mean())

ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

# -----------------------------
# Statistical tests
# -----------------------------
t_stat, t_p = ttest_rel(best_scores, baseline_scores)
w_stat, w_p = wilcoxon(best_scores, baseline_scores)

# -----------------------------
# Output
# -----------------------------
print("\n=== Paired Evaluation: Best Config vs Baseline (Qwen) ===\n")

print(f"Number of paired questions: {n}\n")

print("Mean human score:")
print(f"  Baseline (Pure Qwen): {mean_baseline:.3f}")
print(f"  Best Config (Hybrid): {mean_best:.3f}\n")

print("Mean difference (Best − Baseline):")
print(f"  Δ = {mean_diff:.3f}")
print(f"  95% Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n")

print("Statistical tests:")
print(f"  Paired t-test: t = {t_stat:.3f}, p = {t_p:.4f}")
print(f"  Wilcoxon signed-rank test: W = {w_stat:.3f}, p = {w_p:.4f}")

# -----------------------------
# Optional: save paired data
# -----------------------------
paired.to_csv("paired_qwen_scores.csv", index=False)

# -----------------------------
# Sanity check: global means (before pairing)
# -----------------------------
pure_global_mean = baseline["mean_score"].mean()
rag_global_mean = best["mean_score"].mean()

print("\n=== SANITY CHECK: GLOBAL MEANS (ALL RUNS, UNPAIRED) ===\n")

print("Pure Qwen:")
print(f"  Rows (runs): {len(baseline)}")
print(f"  Unique questions (subrun_id): {baseline['subrun_id'].nunique()}")
print(f"  Global mean score: {pure_global_mean:.3f}\n")

print("RAG Qwen (Hybrid):")
print(f"  Rows (runs): {len(best)}")
print(f"  Unique questions (subrun_id): {best['subrun_id'].nunique()}")
print(f"  Global mean score: {rag_global_mean:.3f}\n")