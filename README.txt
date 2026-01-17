*** Go to any dir, and run the Python script in that dir to see related plots and results. 

*** To run the scripts, we used:
   - Spyder IDE version 5.5.2
   - Python 3.8.10 64-bit | Qt 5.15.2 | PyQt5 5.15.10 
   - OS: Windows 10 Enterprise LTSC. 
   - Hardware: Processor-12th Gen Intel(R) Core(TM) i7-12700H  2.30 GHz; RAM-32.0 GB.

*** If a script contains "HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")" in the begining, then you need HUGGINGFACE subscription to run it! In that case, the script uses subscription-based Hugging Face LLMs via cloud APIs. Otherweise, a script should run fine locally.

*** faiss index files and bm25 db
- faiss db: .\public_dir\test-red-teaming-10-2-7\vectordb_multi\vectordb_faiss_750_all-mpnet-base-v2
- faiss db: .\public_dir\chatbot-calls\vectordb_multi\vectordb_faiss_750_all-mpnet-base-v2
- faiss db: .\public_dir\test-red-teaming-10-2-7\vectordb_multi\vectordb_faiss_750_sentence-transformers_all-MiniLM-L6-v2
- faiss db: .\public_dir\chatbot-calls\vectordb_multi\vectordb_faiss_750_sentence-transformers_all-MiniLM-L6-v2
- bm25 db: .\public_dir\test-red-teaming-10-2-7\vectordb_multi\vectordb_bm25

*** docs (pdf file for GJU regulations and Jordan Tax law): .\public_dir\chatbot-ingest\docs
*** chatbot-ingest script: .\public_dir\chatbot-ingest
*** chatbot-calls script: .\public_dir\chatbot-calls