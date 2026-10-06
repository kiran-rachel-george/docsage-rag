"""Run the golden set through the real question-answering path (pipeline.ask, the same
function POST /ask calls) for each experiment config, judge the answers, and write results.

    python eval/run_eval.py --configs baseline heading-aware hybrid    # dev questions only
    python eval/run_eval.py --all                                  # every runnable config
    python eval/run_eval.py --all --holdout                        # final run incl. 5 held-out
    python eval/run_eval.py --report                               # rebuild summary table
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent / "src"))

from judge import judge_correctness, judge_faithfulness
from metrics import aggregate, citation_coverage, doc_coverage, hit_at_k, hit_text_at_k
from validate_golden import load_golden

from docsage import pipeline
from docsage.config import settings

RESULTS_DIR = EVAL_DIR / "results"


def load_configs() -> dict[str, dict]:
    cfg = yaml.safe_load((EVAL_DIR / "configs.yaml").read_text(encoding="utf-8"))
    return {c["name"]: c for c in cfg["experiments"]}


def run_question(row: dict, cfg: dict, session: str, _attempt: int = 1) -> dict:
    try:
        a = pipeline.ask(
            row["question"],
            strategy=cfg["chunking"],
            use_bm25=cfg["use_bm25"],
            use_reranker=cfg["use_reranker"],
            session_id=session,
            tags=["eval", cfg["name"]],
        )
    except Exception as e:
        if "error 429" in str(e) and _attempt < 3:
            # rate-limit buckets reset within a minute; wait it out and retry this question
            time.sleep(65)
            return run_question(row, cfg, session, _attempt + 1)
        if any(code in str(e) for code in ("error 401", "error 402", "error 403")):
            # billing/auth problems affect every remaining question: stop, don't score them
            raise SystemExit(f"Aborting: {e}. Fix the OpenRouter key/credits and re-run.") from e
        return {
            "id": row["id"], "type": row["type"], "question": row["question"],
            "error": f"{type(e).__name__}: {e}", "hit": None, "cov": None, "correct": False,
            "faithful": None, "refused": False, "citation_coverage": None,
            "latency_ms": 0.0, "cost_usd": 0.0, "judge_cost": 0.0,
        }

    judge_cost = 0.0
    if row["type"] == "unanswerable":
        correct, reason = a.refused, "refused as expected" if a.refused else "answered instead of refusing"
        faithful = None
    else:
        if a.refused:
            correct, reason = False, "false refusal"
            faithful = None
        else:
            j = judge_correctness(row["question"], row["expected_answer"], a.answer)
            correct, reason, judge_cost = j["pass"], j["reason"], j["cost"]
            cited = [{"doc": c["doc"], "page": c["page"], "text": c["text"]} for c in a.citations]
            f = judge_faithfulness(a.answer, cited or a.hits)
            faithful, judge_cost = f["pass"], judge_cost + f["cost"]
            reason += f" | faithfulness: {f['reason']}"
    return {
        "id": row["id"], "type": row["type"], "question": row["question"],
        "expected": row["expected_answer"], "answer": a.answer, "refused": a.refused,
        "citations": [f"{c['doc']} p.{c['page']}" for c in a.citations],
        "invalid_citations": a.invalid_citations,
        "retrieved": [f"{h['doc']} p.{h['page']}" for h in a.hits],
        "hit": hit_at_k(row, a.hits), "hit_text": hit_text_at_k(row, a.hits),
        "cov": doc_coverage(row, a.hits),
        "correct": correct, "faithful": faithful, "judge_reason": reason,
        "citation_coverage": citation_coverage(a.answer, a.refused),
        "latency_ms": a.latency_ms, "throttle_ms": a.throttle_ms, "cost_usd": a.cost_usd, "judge_cost": judge_cost,
        "trace_id": a.trace_id,
    }


def run_config(cfg: dict, rows: list[dict], include_holdout: bool) -> Path:
    session = f"eval-{cfg['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    rows = [r for r in rows if include_holdout or r["split"] == "dev"]
    print(f"\n=== {cfg['name']}: {len(rows)} questions "
          f"(chunking={cfg['chunking']}, bm25={cfg['use_bm25']}, rerank={cfg['use_reranker']}) ===")
    # Checkpoint after every question so an interrupted run (time limit, crash) resumes
    # where it stopped. Only successful answers are kept; errored ones are retried.
    ckpt = RESULTS_DIR / f".partial_{cfg['name']}_{'holdout' if include_holdout else 'dev'}.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    done = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else []
    done = [r for r in done if not r.get("error")]
    if done:
        print(f"  resuming: {len(done)} question(s) already done")
    results = list(done)
    done_ids = {r["id"] for r in done}
    for i, row in enumerate(rows, 1):
        if row["id"] in done_ids:
            continue
        r = run_question(row, cfg, session)
        results.append(r)
        ckpt.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        mark = "ok " if r["correct"] else "BAD"
        print(f"  [{i:2d}/{len(rows)}] {mark} {r['id']} {r['latency_ms'] / 1000:5.1f}s "
              f"hit={r['hit']} {r.get('error', '')}")
    failed = [r["id"] for r in results if r.get("error")]
    if failed:
        # never publish numbers computed from failed calls
        raise SystemExit(
            f"{len(failed)} question(s) errored ({', '.join(failed)}); "
            "results NOT saved. Fix the cause and re-run."
        )
    agg = aggregate(results)
    out = RESULTS_DIR / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": cfg, "llm_model": settings.llm_model,
        "judge_model": settings.judge_model or settings.llm_model,
        "reranker_model": settings.reranker_model if cfg["use_reranker"] else None,
        "refusal_threshold": settings.refusal_threshold, "include_holdout": include_holdout,
        "date": time.strftime("%Y-%m-%d"), "aggregate": agg,
        "judge_cost_usd": sum(r["judge_cost"] for r in results), "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    print(f"  -> {out.name}")
    return out


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def report() -> str:
    """Markdown table from the newest result file of each config."""
    latest: dict[str, dict] = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        latest[d["config"]["name"]] = d
    order = list(load_configs())
    lines = [
        ("| Configuration | n | hit@5 (page) | hit@5 (text) | Correct | Faithful | Cite cov. | Refused (unans.) "
         "| False refusals | p50 / p95 | $/1,000 q |"),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in order:
        d = latest.get(name)
        if not d:
            continue
        a = d["aggregate"]
        lines.append(
            f"| {name} | {a['n']} | {pct(a['hit_at_5'])} | {pct(a.get('hit_text_at_5'))} | {pct(a['correctness'])} | "
            f"{pct(a['faithfulness'])} | {pct(a['citation_coverage'])} | "
            f"{a['refused_unanswerable']}/{a['n_unanswerable']} | {a['false_refusals']} | "
            f"{a['latency_p50_s']:.1f}s / {a['latency_p95_s']:.1f}s | ${a['cost_per_1000']:.2f} |"
        )
    meta = next(iter(latest.values()), None)
    if meta:
        note = (
            f"LLM `{meta['llm_model']}`, judge `{meta['judge_model']}`, "
            f"refusal threshold {meta['refusal_threshold']}, run {meta['date']}. "
            "Latency excludes provider rate-limit waits. Cost is the answering LLM only (judge excluded)."
        )
        lines += ["", note]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every config that is implemented")
    ap.add_argument("--holdout", action="store_true", help="include the 5 held-out questions")
    ap.add_argument("--limit", type=int, help="first N questions only (smoke test)")
    ap.add_argument("--report", action="store_true", help="only rebuild the summary table")
    args = ap.parse_args()

    if not args.report:
        configs = load_configs()
        names = list(configs) if args.all else args.configs
        if not names:
            ap.error("pass --configs NAME... or --all")
        rows = load_golden()
        if args.limit:
            rows = rows[: args.limit]
        for name in names:
            cfg = configs[name]
            if cfg.get("use_agent"):
                print(f"\n=== {name}: skipped (query agent F6 is not implemented yet) ===")
                continue
            run_config(cfg, rows, args.holdout)
    text = report()
    (RESULTS_DIR / "summary.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
