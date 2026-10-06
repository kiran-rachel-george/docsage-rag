| Configuration | n | hit@5 (page) | hit@5 (text) | Correct | Faithful | Cite cov. | Refused (unans.) | False refusals | p50 / p95 | $/1,000 q |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 25 | 33% | 43% | 52% | 82% | 93% | 4/4 | 10 | 2.3s / 9.2s | $0.89 |
| heading-aware | 25 | 48% | 62% | 64% | 82% | 91% | 4/4 | 4 | 2.2s / 16.5s | $0.82 |
| hybrid | 25 | 48% | 52% | 52% | 81% | 96% | 4/4 | 5 | 2.4s / 8.3s | $0.82 |
| reranker | 25 | 57% | 67% | 56% | 74% | 93% | 4/4 | 2 | 6.3s / 9.5s | $0.78 |

LLM `@cf/meta/llama-3.3-70b-instruct-fp8-fast`, judge `openai/gpt-oss-120b`, refusal threshold 0.55, run 2026-10-05. Latency excludes provider rate-limit waits. Cost is the answering LLM only (judge excluded).
