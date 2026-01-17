# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 20:35:44 2025

@author: Firas.Alhawari
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import scipy.stats as stats

# -----------------------------
# Configuration
# -----------------------------
input_csv = Path(__file__).parent / "chatbot_eval_review_pure_llms_results.csv"

plots_dir = input_csv.parent / "plots_pure"
plots_dir.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(input_csv)
df.columns = df.columns.str.strip()

# Keep only PURE context
df = df[df["context_type"].str.lower() == "pure"]

# Convert numeric columns safely
for col in ["reviewer1_score", "reviewer2_score", "latency_sec"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Compute average reviewer score per run
df["avg_reviewer_score"] = df[["reviewer1_score", "reviewer2_score"]].mean(axis=1)

# Preserve question order exactly as in CSV
question_order = df["question"].drop_duplicates().tolist()

# Map questions to short labels for plots (Q1, Q2, …)
question_labels = [f"Q{i+1}" for i in range(len(question_order))]
question_label_map = dict(zip(question_order, question_labels))
df["question_label"] = df["question"].map(question_label_map)

# -----------------------------
# Aggregate per model/question
# -----------------------------
agg_df = (
    df.groupby(["question", "question_label", "model"], as_index=False)
    .agg(
        avg_score_mean=("avg_reviewer_score", "mean"),
        avg_score_std=("avg_reviewer_score", "std"),
        latency_mean=("latency_sec", "mean"),
        latency_std=("latency_sec", "std"),
    )
)
agg_df = agg_df.fillna(0)

# -----------------------------
# Safe filename helper
# -----------------------------
def sanitize(s):
    return re.sub(r"[^0-9a-zA-Z\-_.]", "_", str(s))

# -----------------------------
# Reviewer Score Plot (All Models)
# -----------------------------
sns.set(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(20, 8))

sns.barplot(
    data=agg_df,
    x="question_label",
    y="avg_score_mean",
    hue="model",
    errorbar=None,
    capsize=0.2,
    width=0.7,
    ax=ax,
)

# Manual error bars
for bars, (_, row) in zip(ax.containers, agg_df.groupby("model", sort=False)):
    for bar, (_, r) in zip(bars, row.iterrows()):
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            yerr=r["avg_score_std"],
            ecolor="black",
            capsize=3,
            linewidth=1.2,
        )

ax.set_title("Reviewer Score (Mean ± SD) — PURE Context", fontsize=14, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("Average Reviewer Score", fontsize=12)
ax.tick_params(axis="x", rotation=35, labelsize=10)
ax.tick_params(axis="y", labelsize=10)

ax.set_xticks(range(len(question_order)))
ax.set_xticklabels(question_labels, rotation=35, ha="right")

ax.legend(
    title="LLM Model",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
    borderaxespad=0,
    frameon=True,
)

plt.subplots_adjust(left=0.05, right=0.8, top=0.9, bottom=0.25)
plt.tight_layout()
plt.savefig(plots_dir / "reviewer_score_pure_all_models.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Latency Plot (All Models)
# -----------------------------
fig, ax = plt.subplots(figsize=(20, 8))

sns.barplot(
    data=agg_df,
    x="question_label",
    y="latency_mean",
    hue="model",
    errorbar=None,
    capsize=0.2,
    width=0.7,
    ax=ax,
)

# Manual error bars
for bars, (_, row) in zip(ax.containers, agg_df.groupby("model", sort=False)):
    for bar, (_, r) in zip(bars, row.iterrows()):
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            yerr=r["latency_std"],
            ecolor="black",
            capsize=3,
            linewidth=1.2,
        )

ax.set_title("Latency (Mean ± SD) — PURE Context", fontsize=14, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("Average Latency (s)", fontsize=12)
ax.tick_params(axis="x", rotation=35, labelsize=10)
ax.tick_params(axis="y", labelsize=10)

ax.set_xticks(range(len(question_order)))
ax.set_xticklabels(question_labels, rotation=35, ha="right")

ax.legend(
    title="LLM Model",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
    borderaxespad=0,
    frameon=True,
)

plt.subplots_adjust(left=0.05, right=0.8, top=0.9, bottom=0.25)
plt.tight_layout()
plt.savefig(plots_dir / "latency_pure_all_models.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Ranking by Reviewer Score
# -----------------------------
summary = (
    df.groupby(["model"], as_index=False)
    .agg(
        avg_score_mean=("avg_reviewer_score", "mean"),
        avg_score_std=("avg_reviewer_score", "std"),
        latency_mean=("latency_sec", "mean"),
        latency_std=("latency_sec", "std"),
    )
)
summary = summary.fillna(0)

# Compute 95% CI for score and latency per model
summary["score_ci95_lower"] = summary["avg_score_mean"] - 1.96 * summary["avg_score_std"] / np.sqrt(df.groupby("model")["avg_reviewer_score"].count().values)
summary["score_ci95_upper"] = summary["avg_score_mean"] + 1.96 * summary["avg_score_std"] / np.sqrt(df.groupby("model")["avg_reviewer_score"].count().values)

summary["latency_ci95_lower"] = summary["latency_mean"] - 1.96 * summary["latency_std"] / np.sqrt(df.groupby("model")["latency_sec"].count().values)
summary["latency_ci95_upper"] = summary["latency_mean"] + 1.96 * summary["latency_std"] / np.sqrt(df.groupby("model")["latency_sec"].count().values)

summary = summary.sort_values(by="avg_score_mean", ascending=False).reset_index(drop=True)
summary["rank"] = summary.index + 1

ranking_file = plots_dir / "config_ranking_pure.csv"
summary.to_csv(ranking_file, index=False)

# -----------------------------
# Cohen's κ per question (allowed scores only, with CI95)
# -----------------------------
allowed_scores = [0, 2, 5, 8, 10]
kappa_list = []

for q in question_order:
    sub_df = df[df["question"] == q].copy()
    r1 = sub_df["reviewer1_score"].apply(lambda x: x if x in allowed_scores else np.nan)
    r2 = sub_df["reviewer2_score"].apply(lambda x: x if x in allowed_scores else np.nan)
    valid_idx = r1.notna() & r2.notna()
    
    if valid_idx.sum() > 1:
        kappa = cohen_kappa_score(r1[valid_idx].astype(int), r2[valid_idx].astype(int), labels=allowed_scores)
        n = valid_idx.sum()
        se = np.sqrt((1 - kappa**2) / n)
        ci95 = (kappa - 1.96*se, kappa + 1.96*se)
    else:
        kappa = np.nan
        ci95 = (np.nan, np.nan)
    
    kappa_list.append({
        "question": q,
        "question_label": question_label_map[q],
        "cohen_kappa": kappa,
        "ci95_lower": ci95[0],
        "ci95_upper": ci95[1]
    })

kappa_df = pd.DataFrame(kappa_list)
kappa_df.to_csv(plots_dir / "cohens_kappa_per_question.csv", index=False)

# -----------------------------
# Per-question score distribution and 95% CI (allowed scores only)
# -----------------------------
ci_list = []
for q in question_order:
    scores = df[df["question"] == q]["avg_reviewer_score"]
    scores = scores.round().isin(allowed_scores) * df["avg_reviewer_score"]
    scores = df[df["question"] == q]["avg_reviewer_score"]
    scores = scores[scores.round().isin(allowed_scores)]
    mean = scores.mean()
    sem = stats.sem(scores) if len(scores) > 1 else 0
    ci95 = sem * stats.t.ppf((1 + 0.95)/2., len(scores)-1) if len(scores) > 1 else 0
    ci_list.append({"question": q, "question_label": question_label_map[q], "mean": mean, "CI95": ci95})

ci_df = pd.DataFrame(ci_list)
ci_df.to_csv(plots_dir / "question_scores_ci95.csv", index=False)

# -----------------------------
# Boxplot per question
# -----------------------------
plt.figure(figsize=(18,6))
sns.boxplot(
    x="question_label",
    y="avg_reviewer_score",
    data=df[df["avg_reviewer_score"].round().isin(allowed_scores)],
    palette="Set3"
)
plt.xticks(rotation=35)
plt.xlabel("Question")
plt.ylabel("Reviewer Score")
plt.title("Distribution of Reviewer Scores per Question — PURE (allowed scores only)")
plt.tight_layout()
plt.savefig(plots_dir / "score_distribution_pure.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Accuracy vs Latency Overview
# -----------------------------
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=summary,
    x="latency_mean",
    y="avg_score_mean",
    hue="model",
    s=120,
    edgecolor="black",
)
plt.title("Accuracy vs Latency Trade-off (PURE Configurations)")
plt.xlabel("Average Latency (s)")
plt.ylabel("Average Reviewer Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(plots_dir / "accuracy_vs_latency_overview_pure.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Display Results
# -----------------------------
print("\n🏆 BEST OVERALL PURE CONFIGURATION:")
best = summary.iloc[0]
print(
    f"Rank 1 | Model: {best['model']} | "
    f"Score: {best['avg_score_mean']:.3f} ± {best['avg_score_std']:.3f} "
    f"(95% CI: {best['score_ci95_lower']:.3f}–{best['score_ci95_upper']:.3f}) | "
    f"Latency: {best['latency_mean']:.3f}s ± {best['latency_std']:.3f} "
    f"(95% CI: {best['latency_ci95_lower']:.3f}–{best['latency_ci95_upper']:.3f})"
)

print("\n🏅 TOP 10 PURE CONFIGURATIONS (by Reviewer Score):")
print(
    summary.head(10)[
        [
            "rank",
            "model",
            "avg_score_mean",
            "avg_score_std",
            "score_ci95_lower",
            "score_ci95_upper",
            "latency_mean",
            "latency_std",
            "latency_ci95_lower",
            "latency_ci95_upper"
        ]
    ].to_string(index=False)
)

print(f"\n✅ All PURE plots saved in: {plots_dir.resolve()}")
print(f"✅ PURE ranking file saved as: {ranking_file.resolve()}")
print(f"✅ Cohen's κ per question saved as: {plots_dir / 'cohens_kappa_per_question.csv'}")
print(f"✅ Question-wise score CI saved as: {plots_dir / 'question_scores_ci95.csv'}")
