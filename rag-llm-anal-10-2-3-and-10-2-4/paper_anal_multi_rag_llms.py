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
# -----------------------------
# Configuration
# -----------------------------
input_csv = Path(__file__).parent / "chatbot_eval_review_rag_llms_results.csv"
plots_dir = input_csv.parent / "plots"
plots_dir.mkdir(exist_ok=True, parents=True)

# Category mapping for the Q1-Q18 questions
category = [
    "Regular Allowance", "Regular Allowance", "Regular Allowance", "Regular Allowance",
    "Regular Allowance", "Regular Allowance", "Regular Allowance", "Regular Allowance",
    "Regular Allowance", "Regular Allowance", "Regular Allowance", "Regular Allowance",
    "Income Tax", "Health Insurance", "Health Insurance",
    "Employee Attributes", "Employee Attributes", "Employee Attributes"]
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

    # Preserve the exact question order from CSV
    sub_df.loc[:, "question"] = pd.Categorical(
        sub_df["question"], categories=question_order, ordered=True
    )
    sub_df = sub_df.sort_values("question", key=lambda x: x.map({q: i for i, q in enumerate(question_order)}))

    title_suffix = (
        f"{ctx.upper()} | Chunk: {chunk if chunk > 0 else 'N/A'} | "
        f"Top-K: {topk if topk > 0 else 'N/A'} | Embedding: {embed}"
    )

    # -----------------
    # Reviewer Score Plot (mean ± SD)
    # -----------------
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=sub_df,
        x="question",
        y="avg_score_mean",
        hue="model",
        errorbar=None,  # no warning, no seaborn CI
        capsize=0.2,
    )

    # Manual SD error bars
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

def short_name(ctx, chunk, topk, embed):
    embed_short = "MPNet" if "mpnet" in embed.lower() else "MiniLM"
    return f"{ctx}_c{int(chunk)}_k{int(topk)}_{embed_short}"

    safe_name = short_name(ctx, chunk, topk, embed)
    plt.savefig(plots_dir / f"reviewer_score_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------
    # Latency Plot (mean ± SD)
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

    # Manual SD error bars
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

    plt.savefig(plots_dir / f"latency_{safe_name}.png", dpi=300, bbox_inches="tight")
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

# -------------------------------------------------
# ADD: 95% Confidence Intervals for Score and Latency
# -------------------------------------------------

ci_rows = []

for _, row in summary.iterrows():
    cfg_mask = (
        (df["context_type"] == row["context_type"]) &
        (df["chunk_size"] == row["chunk_size"]) &
        (df["top_k"] == row["top_k"]) &
        (df["embedding_model"] == row["embedding_model"]) &
        (df["model"] == row["model"])
    )

    # Reviewer scores
    score_vals = df.loc[cfg_mask, "avg_reviewer_score"].dropna().to_numpy()
    n = len(score_vals)

    if n < 2:
        score_lo, score_hi = np.nan, np.nan
    else:
        score_mean = score_vals.mean()
        score_std = score_vals.std(ddof=1)
        score_se = score_std / np.sqrt(n)
        score_lo = score_mean - 1.96 * score_se
        score_hi = score_mean + 1.96 * score_se

    # Latency
    latency_vals = df.loc[cfg_mask, "latency_sec"].dropna().to_numpy()

    if len(latency_vals) < 2:
        lat_lo, lat_hi = np.nan, np.nan
    else:
        lat_mean = latency_vals.mean()
        lat_std = latency_vals.std(ddof=1)
        lat_se = lat_std / np.sqrt(len(latency_vals))
        lat_lo = lat_mean - 1.96 * lat_se
        lat_hi = lat_mean + 1.96 * lat_se

    ci_rows.append({
        "context_type": row["context_type"],
        "chunk_size": row["chunk_size"],
        "top_k": row["top_k"],
        "embedding_model": row["embedding_model"],
        "model": row["model"],
        "num_runs": n,
        "score_CI95_lower": score_lo,
        "score_CI95_upper": score_hi,
        "latency_CI95_lower": lat_lo,
        "latency_CI95_upper": lat_hi,
    })

summary = summary.merge(
    pd.DataFrame(ci_rows),
    on=["context_type", "chunk_size", "top_k", "embedding_model", "model"],
    how="left"
)

ranking_file = plots_dir / "config_ranking.csv"
summary.to_csv(ranking_file, index=False)

# -----------------------------
# Display Results
# -----------------------------
print("\n?? BEST OVERALL CONFIGURATION:")
best = summary.iloc[0]
print(
    f"Rank 1 | Model: {best['model']} | Context: {best['context_type']} | "
    f"Chunk: {best['chunk_size']} | Top-K: {best['top_k']} | Embedding: {best['embedding_model']} | "
    f"Score: {best['avg_score_mean']:.3f} ± {best['avg_score_std']:.3f} | "
    f"Latency: {best['latency_mean']:.3f}s ± {best['latency_std']:.3f}"
)

print("\n?? TOP 10 CONFIGURATIONS (by Reviewer Score):")
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
plt.savefig(plots_dir / "accuracy_vs_latency_overview.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Accuracy vs Response Time Scatter: Model Color, Retrieval Type, Top-k, Chunk Size
# -----------------------------
# Make a copy for plotting
summary_plot = summary.copy()

# Abbreviate embeddings and handle N/A
summary_plot['embedding_abbrev'] = summary_plot['embedding_model'].replace({
    'sentence-transformers_all-mpnet-base-v2':'MPNet',
    'sentence-transformers_all-MiniLM-L6-v2':'MiniLM',
    np.nan:'None',
    'N/A':'None'
})

# Map marker shapes to embeddings (include 'None')
shape_map = {'MPNet':'o', 'MiniLM':'s', 'None':'D'}

# Map chunk size to marker size
size_map = {750:800, 500:500, -1:150}  # -1 for N/A

# Border style: solid for FAISS, dashed for BM25
summary_plot['linestyle'] = summary_plot['context_type'].apply(lambda x: 'dashed' if x=='bm25' else 'solid')

# Create the figure
plt.figure(figsize=(14,9)) # widen plot
ax = plt.gca()

for idx, row in summary_plot.iterrows():
    # Determine annotation based on top_k
    annotate_text = 'x' if row['top_k'] == 15 else ''

    ax.scatter(
     row['latency_mean'],
     row['avg_score_mean'],
     s=size_map[row['chunk_size']],
     color=sns.color_palette("Set1")[['Qwen2.5-72B-Instruct','Mistral-7B-Instruct-v0.3','Gemma-2-9b-it'].index(row['model'])],
     marker=shape_map[row['embedding_abbrev']],
     edgecolor='black',
     linewidth=1.5,
     linestyle=row['linestyle'],
     zorder=3
    )   

    # Annotate x for top_k=15
    if annotate_text:
        ax.text(row['latency_mean'], row['avg_score_mean'], annotate_text, 
                color='black', fontsize=12, ha='center', va='center', zorder=4)

# Axes and title
ax.set_xlabel("Average Response Time (s)", fontsize=16, fontweight='bold')
ax.set_ylabel("Average Reviewer Score", fontsize=16, fontweight='bold')
ax.set_title("Accuracy vs Response Time: LLMs with FAISS/BM25 Retrieval", fontsize=16, fontweight='bold')

ax.tick_params(axis='both', which='major', labelsize=14)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')
    
# Legend for models (color)
import matplotlib.patches as mpatches
model_patches = [mpatches.Patch(color=c, label=m) for c, m in zip(
    sns.color_palette("Set1")[:3], ['Qwen','Mistral','Gemma']
)]

# Legend for embeddings (shape)
embedding_patches = [plt.Line2D([0],[0], marker=s, color='w', label=e, 
                                markerfacecolor='grey', markersize=10) 
                     for e,s in shape_map.items()]

# Legend for retrieval (linestyle)
retrieval_patches = [plt.Line2D([0],[0], color='k', linestyle='solid', label='FAISS'),
                     plt.Line2D([0],[0], color='k', linestyle='dashed', label='BM25')]

# Legend for chunk size (size)
# Update size_map to include None / -1
#size_map = {500:350, 750:500, -1:250}  # -1 for N/A chunk size

# When creating chunk size legend, add N/A
chunk_patches = [    
    plt.scatter([],[], s=size_map[750], color='grey', edgecolor='k', label='750'),
    plt.scatter([],[], s=size_map[500], color='grey', edgecolor='k', label='500'),
    plt.scatter([],[], s=size_map[-1], color='grey', edgecolor='k', label='N/A')  # new legend
]

# Legend for top_k annotation
topk_patches = [
    plt.Line2D([0],[0], marker='o', color='w', label='top-k=10', markerfacecolor='grey', markersize=10),
    plt.Line2D([0],[0], marker='x', color='w', label='top-k=15', markerfacecolor='grey', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
]

# Combine all legends
leg1 = ax.legend(handles=model_patches, title='LLM Model', prop={'weight':'bold', 'size':'12'}, loc='upper left', bbox_to_anchor=(1.0,1))
leg1.get_title().set_fontweight('bold')
leg1.get_title().set_fontsize(13)  # optional

leg2 = ax.legend(handles=embedding_patches, title='Embedding', prop={'weight':'bold', 'size':'12'}, loc='upper left', bbox_to_anchor=(1.0,0.75))
leg2.get_title().set_fontweight('bold')
leg2.get_title().set_fontsize(13)  # optional

leg3 = ax.legend(handles=retrieval_patches, title='Retrieval', prop={'weight':'bold', 'size':'12'}, loc='upper left', bbox_to_anchor=(1.0,0.55))
leg3.get_title().set_fontweight('bold')
leg3.get_title().set_fontsize(13)  # optional

leg4 = ax.legend(handles=chunk_patches, title='Chunk Size', prop={'weight':'bold', 'size':'12'}, loc='upper left', bbox_to_anchor=(1.0,0.35))
leg4.get_title().set_fontweight('bold')
leg4.get_title().set_fontsize(13)  # optional

leg5 = ax.legend(handles=topk_patches, title='Top-k', prop={'weight':'bold', 'size':'12'}, loc='upper left', bbox_to_anchor=(1.0,0.20))
leg5.get_title().set_fontweight('bold')
leg5.get_title().set_fontsize(13)  # optional

ax.add_artist(leg1)
ax.add_artist(leg2)
ax.add_artist(leg3)
ax.add_artist(leg4)

plt.tight_layout()
plt.savefig(plots_dir / "accuracy_vs_response_time_detailed.png", dpi=300, bbox_inches="tight")
plt.show()

#---------------------------------------

print(f"\n? All plots saved in: {plots_dir.resolve()}")
print(f"? Ranking file saved as: {ranking_file.resolve()}")

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

# Ensure question order Q1-Q18 as in original CSV
win_stats.loc[:, "question"] = pd.Categorical(
    win_stats["question"], categories=question_order, ordered=True
)
win_stats = win_stats.sort_values("question", key=lambda x: x.map({q: i for i, q in enumerate(question_order)}))

# Mapping for question labels
question_label_map = {q: f"Q{i+1}" for i, q in enumerate(question_order)}

# -----------------------------
# Compute per-question stats across 3 runs and 2 reviewers
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

    # Reviewer scores (both reviewers, all runs)
    scores = q_rows[["reviewer1_score", "reviewer2_score"]].to_numpy().flatten()
    allowed_scores = [0, 2, 5, 8, 10]
    scores = np.array([s for s in scores if s in allowed_scores])

    mean_score = scores.mean() if len(scores) > 0 else np.nan
    std_score = scores.std(ddof=1) if len(scores) > 1 else np.nan

    # 95% confidence interval for scores (handle identical scores)
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

    # Latency stats (all runs)
    latency = q_rows["latency_sec"].to_numpy()
    mean_latency = latency.mean() if len(latency) > 0 else np.nan
    std_latency = latency.std(ddof=1) if len(latency) > 1 else np.nan

    # Answer and citation from first run only
    first_run = q_rows.iloc[0]
    answer_text = first_run["answer"] if "answer" in first_run else ""
    if "citation_present" in first_run:
        if isinstance(first_run["citation_present"], str):
            citation_present = first_run["citation_present"].strip().upper() == "TRUE"
        else:
            citation_present = bool(first_run["citation_present"])
    else:
        citation_present = np.nan

    # Cohen's kappa across reviewers (all runs, 0/2/5/8/10)
    y1 = q_rows["reviewer1_score"].to_numpy()
    y2 = q_rows["reviewer2_score"].to_numpy()
    
    # Handle perfect agreement (all scores identical)
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

    # Append to list with your exact field names
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
# Convert to DataFrame and save CSV
# -----------------------------
win_stats_per_question = pd.DataFrame(win_stats_list)
winning_csv_file = plots_dir / f"winning_combination_per_question.csv"
win_stats_per_question.to_csv(winning_csv_file, index=False)
print(f"\n?? Winning combination per-question CSV saved as: {winning_csv_file.resolve()}")

# -----------------------------
# Top Models Bar Plot (Mean ± SD) aligned with per-question CSV
# -----------------------------

top_configs = [
    {"model": "Qwen2.5-72B-Instruct", "context_type": "faiss", "chunk_size": 750, "top_k": 15, "embedding_model": "sentence-transformers_all-mpnet-base-v2"},
    {"model": "Mistral-7B-Instruct-v0.3", "context_type": "faiss", "chunk_size": 750, "top_k": 15, "embedding_model": "sentence-transformers_all-MiniLM-L6-v2"},
    {"model": "Gemma-2-9b-it", "context_type": "faiss", "chunk_size": 750, "top_k": 10, "embedding_model": "sentence-transformers_all-MiniLM-L6-v2"},
]

plot_rows = []
for cfg in top_configs:
    mask = (
        (df["model"] == cfg["model"]) &
        (df["context_type"] == cfg["context_type"]) &
        (df["chunk_size"] == cfg["chunk_size"]) &
        (df["top_k"] == cfg["top_k"]) &
        (df["embedding_model"] == cfg["embedding_model"])
    )
    df_sub = df.loc[mask, ["question", "reviewer1_score", "reviewer2_score", "latency_sec"]].copy()
    
    # Flatten reviewer scores
    df_scores = df_sub.melt(id_vars=["question"], value_vars=["reviewer1_score","reviewer2_score"],
                         var_name="reviewer", value_name="score")
    
    df_scores = df_scores[df_scores["score"].isin([0,2,5,8,10])]  # filter allowed scores
    
    # Compute mean & std per question for score
    score_stats = df_scores.groupby("question", as_index=False).agg(
        avg_score_mean=("score", "mean"),
        avg_score_std=("score", "std")
    )
    
    # Compute mean & std per question for latency
    latency_stats = df_sub.groupby("question", as_index=False).agg(
        latency_mean=("latency_sec", "mean"),
        latency_std=("latency_sec", "std")
    )
    
    # Merge stats
    stats_df = pd.merge(score_stats, latency_stats, on="question")
    stats_df["model"] = cfg["model"]
    stats_df["context_type"] = cfg["context_type"]
    stats_df["chunk_size"] = cfg["chunk_size"]
    stats_df["top_k"] = cfg["top_k"]
    stats_df["embedding_model"] = cfg["embedding_model"]
    
    plot_rows.append(stats_df)

plot_df = pd.concat(plot_rows, ignore_index=True)

# Preserve question order
plot_df.loc[:, "question"] = pd.Categorical(
    plot_df["question"], categories=question_order, ordered=True
)
plot_df = plot_df.sort_values("question", key=lambda x: x.map({q:i for i,q in enumerate(question_order)}))
plot_df["question_label"] = plot_df["question"].map({q:f"Q{i+1}" for i,q in enumerate(question_order)})

# Save CSV for the plot
top_models_plot_csv_file = plots_dir / "top_models_per_question_plot_data.csv"
plot_df.to_csv(top_models_plot_csv_file, index=False)
print(f"\n? CSV for top-model per-question bar plot saved as: {top_models_plot_csv_file.resolve()}")

# -----------------------------
# Bar plot with legend raised above y=11
# -----------------------------
plt.figure(figsize=(16,9))

# Define custom palette
custom_palette = ["skyblue", "orange", "mediumseagreen"]

sns.set(style="whitegrid", font_scale=1.1)
ax = sns.barplot(
    data=plot_df,
    x="question_label",
    y="avg_score_mean",
    hue="model",
    palette=custom_palette,
    errorbar=None
)

# Add manual SD bars
for bars, (_, row) in zip(ax.containers, plot_df.groupby("model", sort=False)):
    for bar, (_, r) in zip(bars, row.iterrows()):
        ax.errorbar(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            yerr=r["avg_score_std"],
            ecolor='black',
            capsize=3,
            linewidth=1.2
        )

# Tick labels bold and size 16
plt.xticks(rotation=0, ha="center", fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Axes labels and title
plt.xlabel("Question", fontsize=16, fontweight='bold')
plt.ylabel("Average Reviewer Score", fontsize=16, fontweight='bold')
plt.title("Per-Question Scores for the Best Configuration of Each LLM (Mean ± SD)", fontsize=18, fontweight='bold')

# Add empty vertical space above bars
ymax = plot_df["avg_score_mean"].max() + 3   # add ~1.5 units of empty space
plt.ylim(0, ymax)

# Legend inside plot (upper-right), semi-transparent, compact
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title="LLM Model",
    loc='upper center',
    bbox_to_anchor=(0.9, 0.98),   # inside the new empty band
    fontsize=14,
    title_fontsize=14,
    frameon=True,
    fancybox=True,
    framealpha=0.85,
)

plt.tight_layout()

# Save PNG
top_models_bar_file = plots_dir / "top_models_per_question_barplot.png"
plt.savefig(top_models_bar_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n? Bar plot saved at: {top_models_bar_file.resolve()}")
# -----------------------------
# Compute overall accuracy before and after excluding Q16
# -----------------------------

# -----------------------------
# Paths
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

faiss_csv = Path(__file__).parent / "chatbot_eval_review_rag_llms_results.csv"
hybrid_csv = Path(__file__).parent / "chatbot_eval_review_hybrid_only_results.csv"
hybrid_rerank_csv = Path(__file__).parent / "chatbot_eval_review_hybrid_only_with_trans_results.csv"

plots_dir = faiss_csv.parent / "plots"
plots_dir.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Load Data
# -----------------------------
faiss_df = pd.read_csv(faiss_csv)
hybrid_df = pd.read_csv(hybrid_csv)
hybrid_rerank_df = pd.read_csv(hybrid_rerank_csv)

# Strip columns and fix dtypes
for df_ in [faiss_df, hybrid_df, hybrid_rerank_df]:
    df_.columns = df_.columns.str.strip()
    for col in ["reviewer1_score", "reviewer2_score", "latency_sec", "chunk_size", "top_k"]:
        df_[col] = pd.to_numeric(df_[col], errors="coerce")
    df_["embedding_model"] = df_.get("embedding_model", pd.Series(["N/A"]*len(df_))).fillna("N/A")
    df_['context_type'] = df_['context_type'].astype(str).str.lower()

# -----------------------------
# Define top Qwen FAISS config
# -----------------------------
top_cfg = {
    "model": "Qwen2.5-72B-Instruct",
    "context_type": "faiss",
    "chunk_size": 750,
    "top_k": 15,
    "embedding_model": "sentence-transformers_all-mpnet-base-v2"
}

# Adjust hybrid embedding to match
hybrid_df['embedding_model'] = hybrid_df['embedding_model'].str.replace("/", "_")
hybrid_rerank_df['embedding_model'] = hybrid_rerank_df['embedding_model'].str.replace("/", "_")

# -----------------------------
# Filter top Qwen FAISS and Hybrid
# -----------------------------
faiss_qwen = faiss_df[
    (faiss_df["model"]==top_cfg["model"]) &
    (faiss_df["context_type"]=="faiss") &
    (faiss_df["chunk_size"]==top_cfg["chunk_size"]) &
    (faiss_df["top_k"]==top_cfg["top_k"]) &
    (faiss_df["embedding_model"]==top_cfg["embedding_model"])
]

hybrid_qwen = hybrid_df[
    (hybrid_df["model"]==top_cfg["model"]) &
    (hybrid_df["context_type"]=="hybrid") &
    (hybrid_df["chunk_size"]==top_cfg["chunk_size"]) &
    (hybrid_df["top_k"]==top_cfg["top_k"]) &
    (hybrid_df["embedding_model"]==top_cfg["embedding_model"])
]

hybrid_rerank_qwen = hybrid_rerank_df[
    (hybrid_rerank_df["model"]==top_cfg["model"]) &
    (hybrid_rerank_df["context_type"]=="hybrid") &
    (hybrid_rerank_df["chunk_size"]==top_cfg["chunk_size"]) &
    (hybrid_rerank_df["top_k"]==top_cfg["top_k"]) &
    (hybrid_rerank_df["embedding_model"]==top_cfg["embedding_model"])
]

print(f"FAISS rows: {len(faiss_qwen)}, Hybrid rows: {len(hybrid_qwen)}, Hybrid-Trans rows: {len(hybrid_rerank_qwen)}")  # debug

# -----------------------------
# Question order and labels
# -----------------------------
question_order = faiss_df['question'].drop_duplicates().tolist()
question_label_map = {q: f"Q{i+1}" for i, q in enumerate(question_order)}

# -----------------------------
# Handle appended question text in rerank
# -----------------------------
hybrid_rerank_qwen['question_base'] = hybrid_rerank_qwen['question'].str.replace(
    r"\s*\(for academic staff, admin staff, and students, as applicable\)", "", regex=True
)

# -----------------------------
# Per-question stats
# -----------------------------
def per_question_stats(df_input, question_col='question'):
    stats_list = []
    for q in question_order:
        q_rows = df_input[df_input[question_col]==q]
        scores = pd.to_numeric(q_rows[['reviewer1_score','reviewer2_score']].values.flatten(), errors='coerce')
        scores = scores[np.isin(scores,[0,2,5,8,10])]
        mean_score = scores.mean() if len(scores)>0 else np.nan
        std_score = scores.std(ddof=1) if len(scores)>1 else 0
        stats_list.append({
            'question': q,
            'question_label': question_label_map[q],
            'mean_score': mean_score,
            'score_std': std_score
        })
    return pd.DataFrame(stats_list)

faiss_stats = per_question_stats(faiss_qwen)
hybrid_stats = per_question_stats(hybrid_qwen)
hybrid_rerank_stats = per_question_stats(hybrid_rerank_qwen, question_col='question_base')

# -----------------------------
# Merge for side-by-side bars
# -----------------------------
plot_df = faiss_stats.merge(hybrid_stats, on='question', suffixes=('_faiss','_hybrid'))
plot_df = plot_df.merge(hybrid_rerank_stats, on='question')
plot_df = plot_df.rename(columns={'mean_score':'mean_score_rerank', 'score_std':'score_std_rerank'})
plot_df = plot_df.sort_values('question', key=lambda x: x.map({q:i for i,q in enumerate(question_order)}))

# -----------------------------
# Bar plot
# -----------------------------
plt.figure(figsize=(18,10))
sns.set(style="whitegrid", font_scale=1.1)

x = np.arange(len(plot_df))
width = 0.25

plt.bar(x - width, plot_df['mean_score_faiss'], width, yerr=plot_df['score_std_faiss'], capsize=4, label='FAISS', color='skyblue')
plt.bar(x, plot_df['mean_score_hybrid'], width, yerr=plot_df['score_std_hybrid'], capsize=4, label='Hybrid', color='orange')
plt.bar(x + width, plot_df['mean_score_rerank'], width, yerr=plot_df['score_std_rerank'], capsize=4, label='Hybrid-Query-Transform', color='mediumseagreen')

plt.xticks(x, plot_df['question_label_faiss'], rotation=0, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel("Question", fontsize=16, fontweight='bold')
plt.ylabel("Average Reviewer Score", fontsize=16, fontweight='bold')
plt.title("Qwen Top Config (MPNet-Base, 750, 15): FAISS vs Hybrid vs Hybrid-Query-Transform (Mean ± SD)", fontsize=18, fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()

plot_file = plots_dir / "qwen_faiss_vs_hybrid_trans_barplot.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.show()
print(f"? Plot saved: {plot_file.resolve()}")

# -----------------------------
# Overall means including/excluding Q16
# -----------------------------
for label, df_ in zip(['FAISS','Hybrid','Hybrid-Trans'], [faiss_stats, hybrid_stats, hybrid_rerank_stats]):
    mean_all = df_['mean_score'].mean()
    mean_excl_q16 = df_[df_['question_label']!='Q16']['mean_score'].mean()
    print(f"\n?? {label} overall mean:")
    print(f" - Mean score: {mean_all:.3f}")

# -----------------------------
# Compute Cohen's Kappa per Question Category for Winning Config
# -----------------------------

# Map each question to its category (from your original category list)
question_to_category = dict(zip(question_order, category))

# Filter winning configuration
df_win = df[
    (df["context_type"] == winning_comb["context_type"])
    & (df["chunk_size"] == winning_comb["chunk_size"])
    & (df["top_k"] == winning_comb["top_k"])
    & (df["embedding_model"] == winning_comb["embedding_model"])
    & (df["model"] == winning_comb["model"])
].copy()

# Assign category to each question
df_win["category"] = df_win["question"].map(question_to_category)

# Allowed scores
allowed_scores = [0, 2, 5, 8, 10]

category_stats_list = []

for cat in df_win["category"].unique():
    cat_rows = df_win[df_win["category"] == cat].copy()
    
    # Reviewer scores across all questions in category
    y1 = cat_rows["reviewer1_score"].to_numpy()
    y2 = cat_rows["reviewer2_score"].to_numpy()
    
    mask = np.isin(y1, allowed_scores) & np.isin(y2, allowed_scores)
    y1 = y1[mask]
    y2 = y2[mask]
    
    # Cohen's kappa
    if len(y1) == 0 or len(y2) == 0:
        kappa = np.nan
        ci95_kappa = (np.nan, np.nan)
    elif np.all(y1 == y1[0]) and np.all(y2 == y2[0]) and y1[0] == y2[0]:
        kappa = 1.0
        ci95_kappa = (1.0, 1.0)
    else:
        kappa = cohen_kappa_score(y1, y2)
        se_kappa = np.sqrt((1 - kappa**2) / max(1, len(y1)))
        ci95_kappa = (max(-1, kappa - 1.96*se_kappa), min(1, kappa + 1.96*se_kappa))
    
    # Mean and SD score
    scores = np.concatenate([cat_rows["reviewer1_score"].to_numpy(),
                             cat_rows["reviewer2_score"].to_numpy()])
    scores = scores[np.isin(scores, allowed_scores)]
    mean_score = scores.mean() if len(scores) > 0 else np.nan
    std_score = scores.std(ddof=1) if len(scores) > 1 else np.nan
    
    # Latency
    latency = cat_rows["latency_sec"].to_numpy()
    mean_latency = latency.mean() if len(latency) > 0 else np.nan
    std_latency = latency.std(ddof=1) if len(latency) > 1 else np.nan
    
    category_stats_list.append({
        "category": cat,
        "mean_score": mean_score,
        "score_std": std_score,
        "mean_latency": mean_latency,
        "latency_std": std_latency,
        "cohen_kappa": kappa,
        "cohen_CI95_lower": ci95_kappa[0],
        "cohen_CI95_upper": ci95_kappa[1],
        "num_questions": len(cat_rows["question"].unique()),
        "num_rows": len(cat_rows)
    })

# Convert to DataFrame and save CSV
df_category_stats = pd.DataFrame(category_stats_list)
category_csv_file = plots_dir / "cohen_kappa_per_category_winning_config.csv"
df_category_stats.to_csv(category_csv_file, index=False)

print(f"\n? Cohen's Kappa per-category CSV saved as: {category_csv_file.resolve()}")