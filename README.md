# RAG: Baseline Bi-Encoder vs. Cross-Encoder Re-Ranking

All code is in **[`RAG_Baseline_vs_Reranking.ipynb`](RAG_Baseline_vs_Reranking.ipynb)**. It was executed end-to-end, so all outputs are saved in it.

- **Document:** official Monopoly rulebook (Hasbro, 8 pages), `data/rulebook.pdf`. It is split into 46 chunks of about 500 characters with 100-character overlap.
- **Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2` (cosine similarity, top 10).
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (re-ranks the top 10, keeps the top 3).
- **LLM:** Azure OpenAI chat deployment (OpenAI and Anthropic also supported), with an identical prompt for both pipelines.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env      # fill in your API credentials
jupyter notebook RAG_Baseline_vs_Reranking.ipynb
```

## Notebook contents
1. PDF loading and chunking
2. Bi-encoder indexing and baseline top-10 retrieval
3. Cross-encoder re-ranking
4. LLM answer generation for both pipelines
5. Failure-mode stress tests: keyword traps, negation/constraints, exceptions, plus control cases
6. Rank-shift summary table and Answer 1 vs. Answer 2 comparison
7. Latency profiling and a scaling estimate for running the cross-encoder over the whole corpus
