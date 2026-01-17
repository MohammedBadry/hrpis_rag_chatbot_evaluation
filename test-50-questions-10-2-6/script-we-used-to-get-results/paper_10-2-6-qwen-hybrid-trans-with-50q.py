#!/usr/bin/env python3
"""
chatbot_eval_top_configs_hybrid_only.py

Evaluates top-performing LLM configurations over payroll questions with:
- Hybrid (FAISS + BM25) retrieval only.

FAISS-only, BM25-only, and pure LLM evaluation are disabled in this version.

Generates CSV and JSON outputs for later reviewer scoring and analysis.

Modifications:
- citation_present flag
- retrieval-confidence score
- timestamp
- faq_fallback_used placeholder
"""

import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import matplotlib.pyplot as plt

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise ValueError("❌ Hugging Face token not found. Set HUGGINGFACE_API_KEY in .env")

# -----------------------------
# Paths
# -----------------------------
OUTPUT_DIR = Path("./vectordb_multi")
RESULTS_CSV = OUTPUT_DIR / "chatbot_eval_hybrid_only.csv"
RESULTS_JSON = OUTPUT_DIR / "chatbot_eval_hybrid_only.json"

NUM_RUNS = 3 # number of repeated runs per setup for statistics
ENABLE_HYBRID = True  # hybrid only

# -----------------------------
# Questions
# -----------------------------

questions = [
    "Basic salary",
    "Basic salary?",
    "What is the basic salary?",
    "What is the basic salary definition?",
    "Transportation allowance",
    "Transportation allowance?",
    "List the transportation allowance values for all",
    "List the transportation allowance values for all, including dean",
    "Family allowance?",
    "Family bonus?",
    "What is the family bonus?",
    "What is the family bonus for wife?",
    "Tax exemption?",
    "Health insurance?",
    "University contribution to health insurance?",
    "Academic staff grades?",
    "Academic staff categories?",
    "Administrative staff categories?",
    "Is membership in health insurance optional for all employees at GJU?",
    "Are children under the age of 18 eligible for inclusion in the health insurance?",
    "Can single females and unemployed divorced females be included as health insurance beneficiaries?",
    "Who can a member include as beneficiaries in the health insurance plan?",
    "Are cosmetic treatments and vaccinations covered under the health insurance regulation?",
    "What dental treatments are included under the health insurance according to this regulation?",
    "Who recommends instructions to the University Council for executing the health insurance regulation?",
    "What are the main tasks and authorities of the health insurance committee?",
    "Can a member who resigned due to health problems continue their health insurance membership?",
    "Does the University contribute to health insurance expenses?",
    "Does the category 'Practicing Professor' exist among the University’s academic staff?",
    "What are the categories of academic staff at the University?",
    "Is a Teaching Assistant considered part of the academic staff?",
    "What are the four Administrative Job Categories at the University?",
    "Which two main employee groups exist at GJU?",
    "Can part-time lecturers be assigned to teach at the University?",
    "What are the three main types of employees at GJU?",
    "Are temporary assignments considered permanent University employment?",
    "What is the main matter the Personnel Affairs Committee deals with for administrative employees?",
    "What is the personal exemption for a resident natural person from 2020 onwards?",
    "What is the tax rate on the first 5,000 JOD of taxable income for a natural person?",
    "Explain the additional exemptions for medical, educational, and housing expenses for resident individuals",
    "How are joint tax return exemptions divided between spouses?",
    "How is taxable income for a resident natural person calculated in Jordan, and what exemptions reduce it?",
    "Are end-of-service benefits fully taxable?",
    "What is the maximum exemption for children’s medical, educational, or housing expenses?",
    "How much is the exemption for the taxpayer himself/herself for medical, educational, and housing expenses?",
    "What is the tax rate on income exceeding 1 million JOD for individuals?",
    "Are donations to public institutions or approved charities deductible?",
    "What is income tax?",
    "How is the transportation bonus determined for faculty members on contracts?",
    "What is the salary range for Fourth Class B staff, and what is their annual increase?"
]

# Category mapping for the 50 questions
category = [
    "Regular Allowance", "Regular Allowance", "Regular Allowance", "Regular Allowance",
    "Regular Allowance", "Regular Allowance", "Regular Allowance", "Regular Allowance",
    "Regular Allowance", "Regular Allowance", "Regular Allowance", "Regular Allowance",
    "Regular Allowance", "Income Tax", "Health Insurance", "Health Insurance",
    "Employee Attributes", "Employee Attributes", "Employee Attributes", "Health Insurance",
    "Health Insurance", "Health Insurance", "Health Insurance", "Health Insurance",
    "Health Insurance", "Health Insurance", "Health Insurance", "Health Insurance",
    "Health Insurance", "Employee Attributes", "Employee Attributes", "Employee Attributes",
    "Employee Attributes", "Employee Attributes", "Employee Attributes", "Employee Attributes",
    "Employee Attributes", "Employee Attributes", "Income Tax", "Income Tax", "Income Tax",
    "Income Tax", "Income Tax", "Income Tax", "Income Tax", "Income Tax", "Income Tax",
    "Income Tax", "Income Tax", "Regular Allowance", "Regular Allowance"
]

# Difficulty mapping for the 50 questions
difficulty = [
    "Difficult", "Difficult", "Difficult", "Medium", "Difficult", "Difficult", "Medium",
    "Medium", "Difficult", "Difficult", "Difficult", "Easy", "Difficult", "Difficult",
    "Difficult", "Difficult", "Medium", "Medium", "Easy", "Easy", "Easy", "Medium", "Easy",
    "Medium", "Easy", "Medium", "Medium", "Easy", "Easy", "Medium", "Easy", "Easy", "Medium",
    "Easy", "Medium", "Easy", "Medium", "Easy", "Easy", "Medium", "Medium", "Difficult", 
    "Medium", "Easy", "Easy", "Medium", "Medium", "Medium", "Medium"
]

# Append globally -> query transformation
questions = [q + " (for academic staff, admin staff, and students, as applicable)" for q in questions]

# -----------------------------
# Top LLM configurations
# -----------------------------
TOP_CONFIGS = [
    {
        "model_name": "Qwen2.5-72B-Instruct",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "chunk_size": 750,
        "top_k": 15,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    }
]

# -----------------------------
# Load FAISS indexes
# -----------------------------
faiss_indexes = {}
for cfg in TOP_CONFIGS:
    emb_name_safe = cfg["embedding_model"].replace("/", "_").replace(":", "_")
    embedding_model = HuggingFaceEmbeddings(model_name=cfg["embedding_model"])
    chunk_size = cfg["chunk_size"]
    dir_pattern = f"vectordb_faiss_{chunk_size}_{emb_name_safe}"
    faiss_dir = OUTPUT_DIR / dir_pattern

    if not faiss_dir.exists():
        alt_pattern = dir_pattern.replace("sentence-transformers_", "")
        faiss_dir_alt = OUTPUT_DIR / alt_pattern
        if faiss_dir_alt.exists():
            faiss_dir = faiss_dir_alt
        else:
            print(f"⚠ Skipping missing index: {faiss_dir}")
            continue

    print(f"Loading FAISS index: {faiss_dir}")
    faiss_key = f"{chunk_size}_{emb_name_safe}"
    faiss_indexes[faiss_key] = FAISS.load_local(
        str(faiss_dir),
        embedding_model,
        allow_dangerous_deserialization=True
    )

# -----------------------------
# Load BM25 baseline
# -----------------------------
bm25 = None
chunk_text_map = []
bm25_dir = OUTPUT_DIR / "vectordb_bm25"
tokenized_file = bm25_dir / "tokenized_corpus.jsonl"
if tokenized_file.exists():
    chunk_texts = [json.loads(line)["tokens"] for line in open(tokenized_file, "r", encoding="utf-8")]
    bm25 = BM25Okapi(chunk_texts)
    chunk_text_map = [" ".join(t) for t in chunk_texts]
else:
    print("⚠ BM25 tokenized corpus not found. Hybrid cannot run without BM25.")
    ENABLE_HYBRID = False

# -----------------------------
# Query function
# -----------------------------
def query_model(client, question, context=None, max_tokens=1024, temperature=0.3):
    prompt = (
        "You are a payroll assistant. ONLY answer using the information provided in the context.\n"
        "Always include citations using the document names shown in brackets (e.g., [Payroll_Regulations_2024]).\n"
        "If the answer is not explicitly in the context, say 'I don't know based on the documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    try:
        response = client.chat_completion(
            model=client.model,
            messages=[
                {"role": "system", "content": "You are a helpful payroll assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        response = client.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature)
        if isinstance(response, dict):
            return response.get("generated_text", "").strip()
        else:
            return str(response).strip()

# -----------------------------
# Evaluation Loop (Hybrid only)
# -----------------------------
results = []
current_iter = 0
total_iters = len(TOP_CONFIGS) * len(questions) * NUM_RUNS
print(f"DEBUG: total_iters = {total_iters}")

for run_id in range(1, NUM_RUNS + 1):
    for cfg in TOP_CONFIGS:
        print(f"\n🔹 Evaluating model: {cfg['model_name']}")
        client = InferenceClient(model=cfg["model_id"], token=HF_TOKEN)
        subrun_id = 1

        emb_name_safe = cfg["embedding_model"].replace("/", "_").replace(":", "_")
        faiss_key = f"{cfg['chunk_size']}_{emb_name_safe}"
        db = faiss_indexes.get(faiss_key)
        if db is None or bm25 is None:
            print(f"⚠ Skipping hybrid for {cfg['model_name']} due to missing FAISS or BM25.")
            continue

        for q in questions:
            k = cfg["top_k"]
            # ---------- Hybrid ----------
            docs = db.similarity_search(q.lower(), k=k)
            context_faiss = "\n\n".join([
                f"[{doc.metadata.get('source', f'Doc{i+1}')}] {doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            token_query = q.lower().split()
            bm25_results = bm25.get_top_n(token_query, chunk_text_map, n=k)
            context_bm25 = "\n\n".join([f"[BM25_Doc_{i+1}] {chunk}" for i, chunk in enumerate(bm25_results)])
            context_hybrid = context_faiss + "\n\n" + context_bm25

            start = time.time()
            answer_hybrid = query_model(client, q, context=context_hybrid, max_tokens=1024)
            latency = time.time() - start

            citation_present = any(f"[{doc.metadata.get('source', f'Doc{i+1}')}]" in answer_hybrid for i, doc in enumerate(docs)) \
                               or any(f"[BM25_Doc_{i+1}]" in answer_hybrid for i, chunk in enumerate(bm25_results))

            # ---------- Top-1 normalized confidence ----------
            docs_with_scores = db.similarity_search_with_score(q.lower(), k=k)
            faiss_scores = [s for d, s in docs_with_scores]
            if faiss_scores:
                faiss_conf = (max(faiss_scores) - min(faiss_scores)) / (max(faiss_scores) if max(faiss_scores) != 0 else 1)
            else:
                faiss_conf = 0.0

            bm25_scores_all = bm25.get_scores(token_query)
            chunk_index_map = {chunk: i for i, chunk in enumerate(chunk_text_map)}
            bm25_topk_scores = [bm25_scores_all[chunk_index_map[chunk]] for chunk in bm25_results if chunk in chunk_index_map]
            if bm25_topk_scores:
                bm25_conf = (max(bm25_topk_scores) - min(bm25_topk_scores)) / (max(bm25_topk_scores) if max(bm25_topk_scores) != 0 else 1)
            else:
                bm25_conf = 0.0

            confidence_score = (faiss_conf + bm25_conf) / 2

            # Fallback if confidence < 0.5
            faq_fallback_used = confidence_score < 0.5

            timestamp = time.time()

            results.append({
                "run_id": run_id,
                "subrun_id": subrun_id,
                "question": q,
                "context_type": "hybrid",
                "chunk_size": cfg["chunk_size"],
                "top_k": k,
                "model": cfg["model_name"],
                "embedding_model": cfg["embedding_model"],
                "answer": answer_hybrid,
                "latency_sec": latency,
                "reviewer1_score": None,
                "reviewer2_score": None,
                "citation_present": citation_present,
                "confidence_score": confidence_score,
                "timestamp": timestamp,
                "faq_fallback_used": faq_fallback_used
            })

            print(f"[{current_iter+1}/{total_iters}] model={cfg['model_name']} | context=HYBRID | "
                  f"chunk={cfg['chunk_size']} | k={k} | confidence={confidence_score:.3f} | "
                  f"citation={citation_present} | latency={latency:.2f}s | q='{q[:60]}...'")
            current_iter += 1
            subrun_id += 1

# -----------------------------
# Save results
# -----------------------------
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)
print(f"\n✅ CSV results saved to: {RESULTS_CSV}")

safe_results = make_json_safe(results)
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(safe_results, f, ensure_ascii=False, indent=2)
print(f"✅ JSON results saved to: {RESULTS_JSON}")

# -----------------------------
# Plot confidence per question (avg ± std)
# -----------------------------
conf_stats = df.groupby("question")["confidence_score"].agg(
    confidence_mean="mean",
    confidence_std="std"
).reset_index()

# Save per-question confidence stats to CSV
CONF_STATS_CSV = OUTPUT_DIR / "confidence_stats_per_question.csv"
conf_stats.to_csv(CONF_STATS_CSV, index=False)
print(f"✅ Per-question confidence stats saved to: {CONF_STATS_CSV}")

# Plot
plt.figure(figsize=(14, 6))
plt.errorbar(
    conf_stats['question'],
    conf_stats['confidence_mean'],
    yerr=conf_stats['confidence_std'],
    fmt='o',
    ecolor='red',
    capsize=5,
    markersize=6,
    linestyle='',
    color='blue'
)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Question")
plt.ylabel("Confidence Score (Top-1 avg normalized)")
plt.title(f"Hybrid Mode: Confidence per Question (Avg ± Std over {NUM_RUNS} runs)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot as PNG
plot_file = OUTPUT_DIR / "confidence_per_question.png"
plt.savefig(plot_file, dpi=300)
print(f"✅ Plot saved to: {plot_file}")

plt.show()