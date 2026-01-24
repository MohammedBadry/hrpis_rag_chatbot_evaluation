# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 23:00:52 2025

@author: Firas.Alhawari
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import scipy.stats as stats
import hashlib

# -----------------------------
# Configuration
# -----------------------------
input_csv = Path(__file__).parent / "chatbot_eval_hybrid_with_trans_q16_review_50_3runs.csv"
plots_dir = input_csv.parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)  # ensure base plots folder exists

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(input_csv)
df.columns = df.columns.str.strip()

# Ignore PURE context
df = df[df["context_type"].str.lower() != "pure"]

# Convert numeric columns
for col in ["reviewer1_score", "reviewer2_score", "latency_sec", "chunk_size", "top_k"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["chunk_size"] = df["chunk_size"].fillna(-1)
df["top_k"] = df["top_k"].fillna(-1)
df["embedding_model"] = df["embedding_model"].fillna("N/A")

# Compute average reviewer score per run/reviewer
df["avg_reviewer_score"] = df[["reviewer1_score", "reviewer2_score"]].mean(axis=1)

# Preserve question order exactly as in the CSV
question_order = df["question"].drop_duplicates().tolist()

# -----------------------------
# Aggregate per model/question
# -----------------------------
agg_df = (
    df.groupby(
        ["context_type", "chunk_size", "top_k", "embedding_model", "model", "question"],
        as_index=False,
    )
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
# Generate plots per combination
# -----------------------------
sns.set(style="whitegrid", font_scale=1.0)
comb_vars = ["context_type", "chunk_size", "top_k", "embedding_model"]
unique_combs = agg_df[comb_vars].drop_duplicates()

for _, comb_row in unique_combs.iterrows():
    ctx, chunk, topk, embed = comb_row

    sub_df = agg_df[
        (agg_df["context_type"] == ctx)
        & (agg_df["chunk_size"] == chunk)
        & (agg_df["top_k"] == topk)
        & (agg_df["embedding_model"] == embed)
    ].copy()

    if sub_df.empty:
        continue

    # Preserve exact question order
    sub_df.loc[:, "question"] = pd.Categorical(
        sub_df["question"], categories=question_order, ordered=True
    )
    sub_df = sub_df.sort_values("question", key=lambda x: x.map({q: i for i, q in enumerate(question_order)}))

    title_suffix = (
        f"{ctx.upper()} | Chunk: {chunk if chunk > 0 else 'N/A'} | "
        f"Top-K: {topk if topk > 0 else 'N/A'} | Embedding: {embed}"
    )

    #safe_name = "_".join([f"{k}-{sanitize(comb_row[k])}" for k in comb_vars])
    
    # With a short hash-based name
    combo_str = "_".join([f"{k}-{str(comb_row[k])}" for k in comb_vars])
    safe_name = hashlib.md5(combo_str.encode()).hexdigest()  # short unique identifier

    # -----------------
    # Reviewer Score Plot
    # -----------------
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=sub_df,
        x="question",
        y="avg_score_mean",
        hue="model",
        errorbar=None,
        capsize=0.2,
    )

    for bars, (_, row) in zip(ax.containers, sub_df.groupby("model", sort=False)):
        for bar, (_, r) in zip(bars, row.iterrows()):
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=r["avg_score_std"],
                ecolor="black",
                capsize=3,
                linewidth=1.2,
            )

    plt.title(f"Reviewer Score (Mean ± SD)\n{title_suffix}", fontsize=12, weight="bold")
    plt.xlabel("Question")
    plt.ylabel("Average Reviewer Score")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="LLM Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    score_plot_file = plots_dir / f"reviewer_score_{safe_name}.png"
    score_plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(score_plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------
    # Latency Plot
    # -----------------
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=sub_df,
        x="question",
        y="latency_mean",
        hue="model",
        errorbar=None,
        capsize=0.2,
    )

    for bars, (_, row) in zip(ax.containers, sub_df.groupby("model", sort=False)):
        for bar, (_, r) in zip(bars, row.iterrows()):
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=r["latency_std"],
                ecolor="black",
                capsize=3,
                linewidth=1.2,
            )

    plt.title(f"Latency (Mean ± SD)\n{title_suffix}", fontsize=12, weight="bold")
    plt.xlabel("Question")
    plt.ylabel("Average Latency (s)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="LLM Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    latency_plot_file = plots_dir / f"latency_{safe_name}.png"
    latency_plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(latency_plot_file, dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------
# Ranking by Reviewer Score
# -----------------------------
summary = (
    df.groupby(["context_type", "chunk_size", "top_k", "embedding_model", "model"], as_index=False)
    .agg(
        avg_score_mean=("avg_reviewer_score", "mean"),
        avg_score_std=("avg_reviewer_score", "std"),
        latency_mean=("latency_sec", "mean"),
        latency_std=("latency_sec", "std"),
    )
)
summary = summary.fillna(0)
summary = summary.sort_values(by="avg_score_mean", ascending=False).reset_index(drop=True)
summary["rank"] = summary.index + 1

ranking_file = plots_dir / "config_ranking.csv"
ranking_file.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(ranking_file, index=False)

# -----------------------------
# Display Results
# -----------------------------
print("\n>> BEST OVERALL CONFIGURATION:")
best = summary.iloc[0]
print(
    f"Rank 1 | Model: {best['model']} | Context: {best['context_type']} | "
    f"Chunk: {best['chunk_size']} | Top-K: {best['top_k']} | Embedding: {best['embedding_model']} | "
    f"Score: {best['avg_score_mean']:.3f} ± {best['avg_score_std']:.3f} | "
    f"Latency: {best['latency_mean']:.3f}s ± {best['latency_std']:.3f}"
)

print("\n>> TOP 10 CONFIGURATIONS (by Reviewer Score):")
print(
    summary.head(10)[
        [
            "rank",
            "model",
            "context_type",
            "embedding_model",
            "chunk_size",
            "top_k",
            "avg_score_mean",
            "avg_score_std",
            "latency_mean",
            "latency_std",
        ]
    ].to_string(index=False)
)

# -----------------------------
# Accuracy vs Latency Overview
# -----------------------------
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=summary,
    x="latency_mean",
    y="avg_score_mean",
    hue="context_type",
    style="model",
    s=120,
    edgecolor="black",
)
plt.title("Accuracy vs Latency Trade-off (All Configurations)")
plt.xlabel("Average Latency (s)")
plt.ylabel("Average Reviewer Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
overview_file = plots_dir / "accuracy_vs_latency_overview.png"
overview_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(overview_file, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Generate winning combination CSV with per-question stats (3 runs, 2 reviewers)
# -----------------------------
winning_comb = summary.iloc[0]
win_mask = (
    (agg_df["context_type"] == winning_comb["context_type"])
    & (agg_df["chunk_size"] == winning_comb["chunk_size"])
    & (agg_df["top_k"] == winning_comb["top_k"])
    & (agg_df["embedding_model"] == winning_comb["embedding_model"])
    & (agg_df["model"] == winning_comb["model"])
)
win_stats = agg_df[win_mask].copy()

# Ensure question order Q1-Q18
win_stats.loc[:, "question"] = pd.Categorical(
    win_stats["question"], categories=question_order, ordered=True
)
win_stats = win_stats.sort_values("question", key=lambda x: x.map({q: i for i, q in enumerate(question_order)}))

# Mapping for question labels
question_label_map = {q: f"Q{i+1}" for i, q in enumerate(question_order)}

# -----------------------------
# Compute per-question stats
# -----------------------------
win_stats_list = []
for q in question_order:
    q_rows = df[
        (df["question"] == q)
        & (df["context_type"] == winning_comb["context_type"])
        & (df["chunk_size"] == winning_comb["chunk_size"])
        & (df["top_k"] == winning_comb["top_k"])
        & (df["embedding_model"] == winning_comb["embedding_model"])
        & (df["model"] == winning_comb["model"])
    ].copy()

    # Reviewer scores
    scores = q_rows[["reviewer1_score", "reviewer2_score"]].to_numpy().flatten()
    allowed_scores = [0, 2, 5, 8, 10]
    scores = np.array([s for s in scores if s in allowed_scores])

    mean_score = scores.mean() if len(scores) > 0 else np.nan
    std_score = scores.std(ddof=1) if len(scores) > 1 else np.nan

    # 95% confidence interval
    if len(scores) > 1:
        if np.all(scores == scores[0]):
            ci95_score = (scores[0], scores[0])
        else:
            ci95_score = stats.t.interval(
                0.95,
                len(scores)-1,
                loc=mean_score,
                scale=stats.sem(scores)
            )
    else:
        ci95_score = (np.nan, np.nan)

    # Latency stats
    latency = q_rows["latency_sec"].to_numpy()
    mean_latency = latency.mean() if len(latency) > 0 else np.nan
    std_latency = latency.std(ddof=1) if len(latency) > 1 else np.nan

    # Answer and citation
    first_run = q_rows.iloc[0]
    answer_text = first_run["answer"] if "answer" in first_run else ""
    if "citation_present" in first_run:
        if isinstance(first_run["citation_present"], str):
            citation_present = first_run["citation_present"].strip().upper() == "TRUE"
        else:
            citation_present = bool(first_run["citation_present"])
    else:
        citation_present = np.nan

    # Cohen's kappa
    y1 = q_rows["reviewer1_score"].to_numpy()
    y2 = q_rows["reviewer2_score"].to_numpy()
    if len(np.unique(y1)) == 1 and len(np.unique(y2)) == 1 and y1[0] == y2[0]:
        kappa = 1.0
        ci95_kappa = (1.0, 1.0)
    elif len(q_rows) >= 2:
        kappa = cohen_kappa_score(y1, y2)
        se_kappa = np.sqrt((1 - kappa**2) / len(q_rows))
        ci95_kappa = (max(-1, kappa - 1.96*se_kappa), min(1, kappa + 1.96*se_kappa))
    else:
        kappa = np.nan
        ci95_kappa = (np.nan, np.nan)

    win_stats_list.append({
        "question": q,
        "question_label": question_label_map[q],
        "answer": answer_text,
        "mean_score": mean_score,
        "score_std": std_score,
        "score_CI95": ci95_score,
        "mean_latency": mean_latency,
        "latency_std": std_latency,
        "citation_present": citation_present,
        "cohen_kappa": kappa,
        "cohen_CI95_lower": ci95_kappa[0],
        "cohen_CI95_upper": ci95_kappa[1]
    })

# -----------------------------
# Save per-question CSV
# -----------------------------
win_stats_per_question = pd.DataFrame(win_stats_list)
winning_csv_file = plots_dir / "winning_combination_per_question.csv"
winning_csv_file.parent.mkdir(parents=True, exist_ok=True)
win_stats_per_question.to_csv(winning_csv_file, index=False)
print(f"\n>> Winning combination per-question CSV saved as: {winning_csv_file.resolve()}")

# -----------------------------
# Compute Cohen's Kappa per Category
# -----------------------------
categories = df["Category"].dropna().unique()
kappa_list = []

for cat in categories:
    cat_rows = df[df["Category"] == cat]

    y1 = cat_rows["reviewer1_score"].to_numpy()
    y2 = cat_rows["reviewer2_score"].to_numpy()

    if len(np.unique(y1)) == 1 and len(np.unique(y2)) == 1 and y1[0] == y2[0]:
        kappa = 1.0
        ci95_kappa = (1.0, 1.0)
    elif len(cat_rows) >= 2:
        kappa = cohen_kappa_score(y1, y2)
        se_kappa = np.sqrt((1 - kappa**2) / len(cat_rows))
        ci95_kappa = (max(-1, kappa - 1.96*se_kappa), min(1, kappa + 1.96*se_kappa))
    else:
        kappa = np.nan
        ci95_kappa = (np.nan, np.nan)

    kappa_list.append({
        "Category": cat,
        "num_questions": cat_rows["question"].nunique(),
        "num_rows": len(cat_rows),
        "cohen_kappa": kappa,
        "cohen_CI95_lower": ci95_kappa[0],
        "cohen_CI95_upper": ci95_kappa[1]
    })

df_kappa_category = pd.DataFrame(kappa_list)
category_csv_file = plots_dir / "cohen_kappa_per_category.csv"
category_csv_file.parent.mkdir(parents=True, exist_ok=True)
df_kappa_category.to_csv(category_csv_file, index=False)
print(f"\n>> Cohen Kappa per Category CSV saved as: {category_csv_file.resolve()}")

# -----------------------------
# Aggregate overall mean, SD, and 95% CI for scores and latency
# -----------------------------
all_scores = df[["reviewer1_score", "reviewer2_score"]].to_numpy().flatten()
all_scores = all_scores[~np.isnan(all_scores)]  # remove NaNs

all_latency = df["latency_sec"].to_numpy()
all_latency = all_latency[~np.isnan(all_latency)]

def compute_mean_ci(values, confidence=0.95):
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0
    if n > 1:
        ci_low, ci_high = stats.t.interval(confidence, n-1, loc=mean, scale=stats.sem(values))
    else:
        ci_low, ci_high = mean, mean
    return mean, std, ci_low, ci_high

score_mean, score_std, score_ci_low, score_ci_high = compute_mean_ci(all_scores)
lat_mean, lat_std, lat_ci_low, lat_ci_high = compute_mean_ci(all_latency)

print("\n>> Overall Reviewer Score Across All Questions:")
print(f"Mean: {score_mean:.3f} | SD: {score_std:.3f} | 95% CI: [{score_ci_low:.3f}, {score_ci_high:.3f}]")

print("\n>> Overall Latency Across All Questions:")
print(f"Mean: {lat_mean:.3f}s | SD: {lat_std:.3f}s | 95% CI: [{lat_ci_low:.3f}s, {lat_ci_high:.3f}s]")
