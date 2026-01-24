1) Summary
This repository provides the experimental artifacts supporting the study “Scrum-Guided Development of an Intelligent Payroll Management System with Rule-Based Automation and RAG Chatbot Support,” to be published in Knowledge-Based Systems.

The repository contains Scrum-related documentation, entity–relationship (ER) diagrams, and multiple experimental modules focused on large language model (LLM) analysis, including pure LLM, RAG-based, and hybrid chatbot approaches. It also includes scripts for chatbot ingestion and invocation, test datasets consisting of structured evaluation questions, red-teaming scenarios, and result files used for quantitative and qualitative evaluation of chatbot performance.

The bulk of the experimental results reported in the paper can be reproduced by executing the analysis Python scripts provided in the corresponding repository directories. All artifacts are publicly available to support transparency, reproducibility, and independent validation.

2) Reproducibility and Execution Instructions
To reproduce the reported plots and results, navigate to any directory in the repository and execute the Python script contained in that directory. Each script generates the results corresponding to its respective experiment.

>> The scripts were executed and tested using the following environment:
    -  IDE: Spyder 5.5.2
    -  Python: 3.8.10 (64-bit)
    -  GUI Framework: Qt 5.15.2 | PyQt5 5.15.10
    -  Operating System: Windows 10 Enterprise LTSC
    -  Hardware: Processor: 12th Gen Intel® Core™ i7-12700H @ 2.30 GHz; Memory: 32 GB RAM.

>> Some scripts require access to subscription-based Hugging Face large language models. If a script includes the line:
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
near the beginning, a valid Hugging Face subscription and API key are required to run that script. These scripts invoke cloud-hosted Hugging Face LLMs via their APIs.

>> Scripts that do not reference a Hugging Face API key can be executed locally without any subscription.

3)  FAISS Index and BM25 Databases
The repository includes prebuilt vector and lexical retrieval databases used in the chatbot experiments:
  - faiss db: .\public_dir\test-red-teaming-10-2-7\vectordb_multi\vectordb_faiss_750_all-mpnet-base-v2
  - faiss db: .\public_dir\chatbot-calls\vectordb_multi\vectordb_faiss_750_all-mpnet-base-v2
  - faiss db: .\public_dir\test-red-teaming-10-2-7\vectordb_multi\vectordb_faiss_750_sentence-transformers_all-MiniLM-L6-v2
  - faiss db: .\public_dir\chatbot-calls\vectordb_multi\vectordb_faiss_750_sentence-transformers_all-MiniLM-L6-v2
  - bm25 db: .\public_dir\test-red-teaming-10-2-7\vectordb_multi\vectordb_bm25

4)  Document Corpus and Chatbot Scripts
>> Document Corpus: PDF documents containing German Jordanian University (GJU) regulations and Jordanian tax law are provided in
.\public_dir\chatbot-ingest\docs.
These documents constitute the knowledge source used for document ingestion and retrieval in the RAG-based chatbot experiments.

>> Chatbot Ingestion Script:
The document ingestion pipeline used to preprocess and index the PDF corpus is located in:
.\public_dir\chatbot-ingest.

>> Chatbot Invocation Script:
The base script used to invoke and evaluate chatbot interactions (chatbot core), including pure LLM, RAG-based, and hybrid configurations, is located in:
.\public_dir\chatbot-calls.

5)  Reviewer Evaluation Results
The repository includes reviewer evaluation results for chatbot responses across multiple configurations and models. For each configuration, answers were generated over three independent runs and evaluated by two reviewers using a fixed set of 18 evaluation questions. The corresponding result files are:
  - Pure LLM model results: chatbot_eval_review_pure_llms_results.csv
  - RAG-based LLM model results: chatbot_eval_review_rag_llms_results.csv
  - Hybrid retrieval RAG model results: chatbot_eval_review_hybrid_only_results.csv
  - Hybrid retrieval RAG with query transformation results: chatbot_eval_review_hybrid_only_with_trans_results.csv

These CSV files contain the aggregated reviewer assessments used for comparative analysis of model performance across pure LLM, RAG-based, and hybrid retrieval configurations

>> The expanded set of 50 evaluation questions, along with their corresponding answer keys, is provided in the file 50_questions_with_ans_keys.xlsx. The experimental results generated using this question set are available in the directory test-50-questions-10-2-6. The Python script and input files required to reproduce these results are also included in this directory.

6) How to Cite
Repository Citation (BibTeX):
@misc{alhawari2026hrpis,
  author = {Feras Al-Hawari and Anoud Alufeishat and Mohammad Habahbeh and Ahmad Alfalayleh},
  title = {HRPIS and RAG-Based Chatbot Evaluation Artifacts},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/firasalhawari/hrpis_rag_chatbot_evaluation}
}

>> Example Citation (APA style):
Feras Al-Hawari, Anoud Alufeishat, Mohammad Habahbeh, & Ahmad Alfalayleh. (2026). HRPIS and RAG-Based Chatbot Evaluation Artifacts [Computer software]. GitHub. https://github.com/firasalhawari/hrpis_rag_chatbot_evaluation

>> Corresponding Paper:
Feras Al-Hawari, Anoud Alufeishat, Mohammad Habahbeh, & Ahmad Alfalayleh. (2026). Scrum-Guided Development of an Intelligent Payroll Management System with Rule-Based Automation and RAG Chatbot Support. Knowledge-Based Systems.

