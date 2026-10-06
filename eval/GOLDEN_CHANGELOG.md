# Golden set changelog

## v2 (2 Oct run reviewed on 5 Oct 2026)

The first full run (v1) was inspected against the PDFs. These label errors were found and
fixed; **no question was added, removed or re-split, and the 5 holdout questions were not
touched.** Fixes were made after seeing v1 results, so v1 and v2 scores are not directly
comparable; v1 results are kept in `results/archive_v1/`.

| id | problem in v1 | fix |
|---|---|---|
| s03 | expected answer demanded "growth of 4.6%", which the question did not ask for; a correct "₹267,021 crore" was judged wrong | expected answer is now just "₹267,021 crore" |
| s06 | the figure also appears on p.27 (client metrics table) | pages `[8, 27]` |
| s15 | the dividend also appears on p.17 (highlights) | pages `[17, 20]` |
| c08 | ambiguous: TCS reports a payout of 80.9% (excl. special dividend) and 99.8% (incl.) | expected answer states both, evidence for both on pp.23-24 |

Metric change: `hit@5 (page)` is unchanged (PRD definition: an expected page is in the top 5).
A second column, `hit@5 (text)`, counts a retrieved chunk as a hit if it contains the expected
evidence string, because the same figure often appears on several pages.

Everything is checked against the PDFs by `python eval/validate_golden.py`.
