# DocSage

Agentic RAG over long PDFs, with page-level citations and measured quality. The default corpus is the FY26 annual reports of **TCS, Infosys and Wipro** (1,291 pages). Every claim cites the exact page, and when the documents don't contain the answer, DocSage says so instead of guessing.

> **Status: work in progress.** Retrieval, answering, the API, the UI and the eval harness are built and a first full evaluation has been run (see [Results](#results)). **No PRD quality target is met yet**: cross-document questions are the main gap, and the query agent that addresses them is not built. See [Status](#status).

**Live demo:** https://docsage-rag.streamlit.app/ (free hosting, so the first visit can take a minute to wake and load the search model; each visitor gets a limited number of questions).

## Problem

Annual reports run to 200–550 pages each. A simple comparison such as "how did attrition change at each company this year?" means opening three PDFs and hunting through each. General chatbots answer fast but often invent figures or can't say where a number came from, which makes them unusable for anything you'd act on.

DocSage answers from the documents only, shows the source passage for every citation, and refuses when the evidence is weak.

## Example

```
Q: What was TCS's employee attrition rate in FY26?
A: TCS's employee attrition rate in FY26 was 13.7% [TCS AR, p.55].
   (page 55 of the PDF: "voluntary IT services' attrition at 13.7% for FY 2026")

Q: How many employees does Wipro have in total?
A: Wipro has a total of 242,156 employees. [Wipro AR, p.17]

Q: What is the best pizza topping?
A: Not in these documents.      (refused by the evidence check, no LLM call)
```

## Quickstart

Requires Python 3.11+ and an API key for one LLM provider: [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) (the default; free daily allowance), [Groq](https://console.groq.com), [DeepSeek](https://platform.deepseek.com) or [OpenRouter](https://openrouter.ai).

```bash
pip install -e ".[dev]"
cp .env.example .env            # then set CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID (or another provider's key)

python scripts/fetch_reports.py # downloads TCS and Wipro; see "Corpus" for Infosys
make index                      # builds both indexes (slow on CPU, see below)
make run                        # API on http://localhost:8000  (docs at /docs)
make ui                         # Streamlit UI on http://localhost:8501
make test && make lint
make eval                       # runs every config on the 25 dev questions (about 10 minutes; resumable if interrupted)
```

```bash
curl -X POST localhost:8000/ask -H "content-type: application/json" \
  -d '{"question": "What was TCS attrition in FY26?"}'
```

The response contains `answer`, `refused`, `citations` (with the source passage text), `latency_ms`, `usage` (tokens and cost in USD) and a Langfuse `trace_id`.

`make index` embeds about 4,700 chunks locally. Measured on an 8-core CPU, the fixed index takes about 9 minutes and the heading-aware index about 31 minutes, so build once and reuse.

## How it works

```
OFFLINE   PDFs ──► PyMuPDF parse ──► strip headers/footers ──► chunk ──► embed (local) ──► data/index/
                   (keeps font sizes)  (repeated margin text)  (fixed or heading-aware)

ONLINE    question ──► vector search ─┐
                       BM25 search ───┴─► reciprocal rank fusion ─► [optional cross-encoder re-rank]
                                                   │
                                      best cosine < threshold? ──yes──► "Not in these documents."
                                                   │ no
                                       top-5 passages ──► LLM (Cloudflare / Groq / DeepSeek / OpenRouter) ──► answer with [DOC, p.N]
                                                                                  │
                                              citations checked against the retrieved passages
```

- **Ingestion** ([src/docsage/ingest](src/docsage/ingest)): PyMuPDF keeps the document name and page number on every chunk. Chunks never span pages, so every citation points at one exact page. Running headers and footers (page numbers, "WIPRO INTEGRATED ANNUAL REPORT 2025-26", section nav bars) are stripped, because they match almost every query term and add noise.
- **Two chunkers behind a flag** (`CHUNKING=fixed|heading`): fixed 512-token windows with 64 overlap, or heading-aware splitting on large-font lines, with tiny sections merged into a neighbour.
- **Hybrid retrieval** ([src/docsage/retrieval](src/docsage/retrieval)): BM25 and cosine similarity over local `bge-small-en-v1.5` embeddings, merged with reciprocal rank fusion (k=60). An optional cross-encoder re-ranker (`jina-reranker-v1-turbo-en`, top 30, passages truncated to 1,000 chars) is behind `USE_RERANKER`.
- **Refusal**: if the best vector cosine is below `REFUSAL_THRESHOLD` (0.55), the system refuses without calling the LLM. Cosine is used instead of the fused score because fused and re-ranker scores aren't comparable across queries.
- **Answering** ([src/docsage/generation](src/docsage/generation)): the LLM must put a `[DOC, p.N]` citation on every factual sentence and ignore instructions found inside the passages. Each citation is mapped back to a retrieved passage, and any that don't match are dropped and reported in `invalid_citations`.
- **Tracing** ([src/docsage/observability](src/docsage/observability)): one Langfuse trace per question, with spans for retrieval, re-ranking, the evidence check and the LLM call (model, tokens, billed cost), plus a `citations_valid` score. Every trace carries the session, tags (`api`, `streamlit`, `eval`), environment and release; provider failures are recorded at level ERROR; reasoning-model thinking is kept on the generation. API keys are redacted before export. It does nothing unless Langfuse keys are set. Questions typed into a public demo are sent to Langfuse, so disclose that to visitors or leave tracing off for the hosted app.

## Corpus

| Document | File in `data/raw/` | Pages | Source |
|---|---|---|---|
| TCS Annual Report 2025-26 | `tcs_ar_fy26.pdf` | 360 | [ar.tcs.com](https://www.ar.tcs.com/) |
| Infosys Integrated Annual Report 2025-26 | `infosys_ar_fy26.pdf` | 383 | [infosys.com](https://www.infosys.com/investors/reports-filings/annual-report/annual-reports/ar-2025-26.html) |
| Wipro Integrated Annual Report 2025-26 | `wipro_ar_fy26.pdf` | 548 | [wipro.com](https://www.wipro.com/investors/annual-reports/) |

The PDFs are not committed to this repo (they are publicly published reports; follow the links above for the originals). `scripts/fetch_reports.py` downloads TCS and Wipro. **Infosys blocks scripted downloads (HTTP 403)**, so download its report in a browser and save it as `data/raw/infosys_ar_fy26.pdf`.

Citation page numbers are PDF page indexes, which can differ from the page numbers printed on the page.

| Index | Chunks | Median chunk | Chunks under 30 words |
|---|---|---|---|
| fixed | 2,159 | 303 words | 3% |
| heading-aware | 2,510 | 243 words | 3% |

## Results

First full evaluation on the **25 dev questions** of the golden set (13 single-fact, 8 cross-document, 4 unanswerable). Answers by `@cf/meta/llama-3.3-70b-instruct-fp8-fast` (Cloudflare Workers AI), judged by `openai/gpt-oss-120b` (Groq), a different model family. Run on 2026-10-05, temperature 0, one run per config.

| Configuration | hit@5 (page) | hit@5 (text) | Correct | Faithful | Cite cov. | Unans. refused | False refusals | p50 / p95 | $ / 1,000 q |
|---|---|---|---|---|---|---|---|---|---|
| 1. Fixed chunks, vector only | 33% | 43% | 52% | 82% | 93% | 4/4 | 10 | 2.3 s / 9.2 s | $0.89 |
| 2. + heading-aware chunking | 48% | 62% | **64%** | 82% | 91% | 4/4 | 4 | 2.2 s / 16.5 s | $0.82 |
| 3. + hybrid (BM25 + vector) | 48% | 52% | 52% | 81% | **96%** | 4/4 | 5 | 2.4 s / 8.3 s | $0.82 |
| 4. + re-ranker (Jina turbo, top 30) | **57%** | **67%** | 56% | 74% | 93% | 4/4 | **2** | 6.3 s / 9.5 s | $0.78 |
| 5. + query agent | not built | | | | | | | | |

| | Single-fact correct (13) | Cross-document correct (8) |
|---|---|---|
| Fixed, vector | 8 | 1 |
| + heading-aware | 10 | 2 |
| + hybrid | 8 | 1 |
| + re-ranker | 9 | 1 |

`hit@5 (page)` is the PRD definition: an expected page is among the top 5 chunks (for cross-document questions, every expected document must be covered). `hit@5 (text)` also counts a chunk that contains the expected figure, because the same number often appears on several pages. Cost is the answering LLM at Cloudflare list prices (actual spend was $0 inside the free daily allowance); judge cost is excluded. Latency excludes provider rate-limit waits.

**How to read this**

- **Targets** (hit@5 >= 85%, correctness >= 80%, faithfulness >= 90%, citation coverage 100%, <= 1 false refusal, p50 < 4 s): **none is met** apart from refusing the unanswerable questions and p50 latency for configs 1-3.
- **The clearest signal** is false refusals (10, 4, 5, 2) and hit@5 by text (43%, 62%, 52%, 67%). Heading-aware chunking is the biggest single gain; the re-ranker gives the best retrieval and fewest false refusals.
- **Correctness differences are within noise.** With 25 questions one question is 4 points, and the gaps between configs are 1-3 questions. Do not read the ordering of 52% / 64% / 52% / 56% as a ranking.
- **Hybrid search did not beat heading-aware chunking alone** on this set (hit@5 by text 52% vs 62%).
- **The re-ranker costs latency**: p50 6.3 s against about 2.3 s, plus occasional 10-20 s calls from the provider in every config. Its faithfulness (74%) was the lowest.
- **All 4 unanswerable questions were refused in every config**, including the 3 topical ones such as Wipro's FY27 guidance. I did not separate refusals by the evidence check from refusals by the LLM.

**Caveats.** Small sample (n = 25). The golden set was corrected after a first run on a different model (see [eval/GOLDEN_CHANGELOG.md](eval/GOLDEN_CHANGELOG.md)), so earlier numbers are not comparable. The 5 held-out questions have not been run. The judge is an LLM, and the PRD's hand-check of 10 judgements has not been done: one case in this run (the exact vs rounded Infosys headcount below) looks judge-strict. Every expected answer was checked against the PDFs (`python eval/validate_golden.py`).

## Failure cases

Eight of the 25 questions fail in all four configs. What actually happens:

1. **Cross-document questions (1-2 of 8 correct in every config).** The top 5 passages rarely cover three companies. The model typically answers one company and then writes "...not in these documents" for the rest (e.g. employee counts: Wipro's 242,156 only), and the re-ranked config once claimed TCS grew revenue fastest while saying it had no data for the others. "How does TCS's revenue compare with Infosys's?" was refused in all four configs. This is what the planned query agent (one sub-query per company) is for.
2. **Table questions.** TCS's employee cost as a percentage of revenue (one row of a multi-year table on p.70) was refused in all four configs: the table is never retrieved.
3. **Vocabulary and wording gaps.** Infosys's revenue growth (9.6%) was refused or answered wrongly in all configs; one config answered with revenue figures and said growth was unavailable. Earlier, "headcount" missed because the reports say "Total Number of Employees"; query rewriting is the likely fix.
4. **Rounded vs exact figures.** For Infosys's employees the model answered "over 3,25,000" (a figure that appears in the report on p.10) while the expected exact value 3,28,594 is on the same page; the judge failed it. Arguably a judge-strictness issue rather than a system error.
5. **Topical but unanswerable questions slip past the threshold.** "What is Wipro's FY27 guidance?" scores a cosine of about 0.75, so only the LLM's refusal instruction catches it (it did, in every config). Off-topic questions (about 0.45) are caught by the threshold.

## Deploy the demo (Streamlit Community Cloud)

Deployed at https://docsage-rag.streamlit.app/. To deploy your own copy:

1. Push this repo to GitHub (the prebuilt heading index in `data/index/heading/` is committed on purpose; `.env`, the PDFs and the fixed index are not).
2. On [share.streamlit.io](https://share.streamlit.io) click **Create app**, choose this repo, branch `main`, main file `app/streamlit_app.py`.
3. In **Advanced settings** pick Python 3.12 and paste your secrets (TOML), for example:

```toml
LLM_PROVIDER = "cloudflare"
LLM_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
CLOUDFLARE_API_TOKEN = "..."
CLOUDFLARE_ACCOUNT_ID = "..."
DEMO_MODE = "1"             # hides developer controls and hides cost
DEMO_MAX_QUESTIONS = "20"   # per visitor, protects the free LLM quota
# optional tracing:
LANGFUSE_PUBLIC_KEY = "pk-lf-..."
LANGFUSE_SECRET_KEY = "sk-lf-..."
LANGFUSE_HOST = "https://cloud.langfuse.com"
LANGFUSE_ENVIRONMENT = "production"
```

`requirements.txt` holds the runtime dependencies only. In a simulated hosted run (no `.env`, secrets from `secrets.toml`) the app used about 310 MB of memory against the free tier's 690 MB guarantee.

## Tradeoffs

- **Local embeddings and re-ranker** (no API key, free) vs speed: indexing is slow on CPU, so the index is built once and shipped, not built at deploy time.
- **Chunks never span pages**: exact citations, at the cost of splitting a table or section that crosses a page break.
- **Refusal on cosine similarity**: simple and free (no LLM call), but blind to topical unanswerable questions (failure case 2).
- **Provider choice**: free tiers shaped the project. Cloudflare Workers AI (default) gives a free daily allowance and reports neurons rather than dollars, so cost is computed at list price; Groq's free tier caps tokens per minute and per day (too tight for the full eval); OpenRouter and DeepSeek need paid credits but report usage cleanly. Citation-format compliance varies by model, so `invalid_citations` is reported on every response.
- **Tables are not parsed specially.** A figure can be split across chunks.

## Status

| Area | State |
|---|---|
| Ingestion, both chunkers, header/footer stripping | Done |
| Hybrid retrieval, optional re-ranker | Done |
| Cited answers, refusal, `/ask` API | Done, tested with a stubbed LLM and a few live questions |
| Streamlit UI | Done and deployed on Streamlit Community Cloud (question box, clickable sources, hosted mode with a per-visitor question cap) |
| Langfuse tracing | Done; real traces fetched back from Langfuse and audited against its best-practices checklist (nesting, types, usage and cost, scores, ERROR levels) |
| Tests and lint | 34 tests passing, ruff clean; CI workflow written, not yet run on GitHub |
| Docker | Dockerfile and `.dockerignore` written, not tested |
| Golden set (30 questions) and eval harness (`make eval`) | Done; first run on the 25 dev questions complete; 5 holdout questions not yet run; judge hand-check not done |
| Query agent (comparison questions) | Not started |
| Prompt-injection tests, streaming, deploy to Hugging Face Spaces | Not started |

## Project layout

```
src/docsage/
  ingest/          pdf_loader.py, chunking.py, build_index.py
  retrieval/       embedder.py, vector.py, bm25.py, hybrid.py, reranker.py
  generation/      prompts.py, answer.py, refusal.py
  api/             main.py, schemas.py            (FastAPI /ask)
  observability/   tracing.py                     (Langfuse)
  pipeline.py      question -> retrieve -> refuse-or-answer -> cited result
app/streamlit_app.py
scripts/fetch_reports.py
eval/              golden_set.jsonl, GOLDEN_CHANGELOG.md, validate_golden.py, run_eval.py, metrics.py, judge.py, configs.yaml, results/
tests/
```

Configuration lives in `.env` (see `.env.example`): `LLM_PROVIDER` (`cloudflare`, `groq`, `deepseek` or `openrouter`) with that provider's key, `LLM_MODEL`, `JUDGE_PROVIDER` / `JUDGE_MODEL` for the eval judge, the optional `LANGFUSE_*` keys, and flags such as `CHUNKING`, `USE_BM25`, `USE_RERANKER` and `REFUSAL_THRESHOLD`.

## Limitations

DocSage reads text and tables only, not charts or images inside the PDFs. It does not support user accounts or private uploads, and it only reports what the documents say; it is not financial advice.

## License

MIT. The annual reports belong to their publishers; link to the originals rather than redistributing them.
