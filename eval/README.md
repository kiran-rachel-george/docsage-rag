# DocSage golden set

`golden_set.jsonl`: 30 hand-written questions over the FY26 (year ended 31 March 2026) integrated annual reports of TCS, Infosys and Wipro. Written on 2 Oct 2026, before any retrieval tuning.

| Type | Count | Notes |
|---|---|---|
| single_fact | 15 | 5 per company; 1 per company is a table question (`"table": true`) |
| comparison | 10 | Cross-document; several deliberately compare differently-defined metrics |
| unanswerable | 5 | 3 topical (plausible but not in the reports), 2 off-topic |

## Fields
- `id`, `type`, `question`, `expected_answer`
- `answerable`: false for the 5 unanswerable questions; the correct behaviour is to refuse
- `sources`: list of `{doc, pdf_page}`. `doc` is `tcs` | `infosys` | `wipro` (the files in `data/`). **`pdf_page` is the 1-based page index in the PDF file, not the printed page number.** Several TCS and Infosys PDF pages are two-page spreads, so the printed numbers differ.
- `table`: true for questions whose answer sits in a table (measures table parsing)
- `topical`: on unanswerable questions, true if the question sounds answerable from the reports
- `notes`: why the question is tricky, where relevant

## Scoring
- Retrieval hit@5: a hit when any expected `{doc, pdf_page}` appears among the top-5 retrieved chunks' metadata. For comparisons, also report per-document recall.
- Unanswerable: pass only if the system refuses.

Every expected page was checked against `pdftotext` output of the source PDFs (all checks passed). Suggested hold-out (one of each kind, no table questions): q04, q09, q13, q20 and q27. Keep them aside until the final run (see the PRD's risks section).
