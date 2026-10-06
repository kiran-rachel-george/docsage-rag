| Configuration | n | hit@5 | Correct | Faithful | Cite cov. | Refused (unans.) | False refusals | p50 / p95 | $/1,000 q |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 25 | 29% | 44% | 80% | 95% | 4/4 | 11 | 2.4s / 3.0s | $0.00 |
| heading-aware | 25 | 38% | 56% | 75% | 92% | 4/4 | 5 | 2.7s / 3.7s | $0.00 |
| hybrid | 25 | 33% | 52% | 88% | 94% | 4/4 | 5 | 1.7s / 2.0s | $0.00 |

LLM `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b`, refusal threshold 0.55, run 2026-10-04. Latency excludes provider rate-limit waits. Cost is the answering LLM only (judge excluded).
