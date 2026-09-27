# Origin

This engine was **extracted from a production self-hosted system**, not
written as a library. A larger closed-stack version of the same
recommendation service serves real users daily; this repo is the part
that stood on its own.

## What actually happened

The scorer, the recall channels, and the fitted weights ran for months
inside a private media system before anything was published. Extraction
(squashed into the initial public commit) kept the ranking core
byte-compatible with what production runs, then rebuilt everything
around it so the engine works on *your* database:

| Kept from production | Rebuilt for standalone |
|---|---|
| 22-feature scorer + `weights.json` | split movie/tv schema + compatibility views |
| recall channels (SQL, TMDB graph, KNN) | `ingest/` — daily-export mirror loader |
| saga advancement, MMR, popularity guardrails | personalization layer (events → w_i → user vectors) |
| performance profile (27µs/pair scoring) | `demo/`, `eval/`, benchmarks |

## What stays closed — and why

- **The tagging pipeline** that turns messy channel posts into clean
  titles: it's the accumulation of months of labeled corrections, and
  it's the part competitors can't shortcut.
- **The graded training/eval data** behind the weights. The
  coefficients are published; the 6,650 pairs they were fitted on are
  not — anyone can use the result, nobody gets to replay the fit.
- **The narrative-tag enrichment** (tone/arc/structure features are in
  the public feature space, but the tagger that fills them is closed).

None of that is needed to run this repo at full engine quality — those
features degrade to zero and everything else keeps working.

## Honest gaps

- The public eval set is one curator's judgments on a small pool
  (`eval/`), not the production training data.
- Numbers in the README were measured on the demo catalog; scale the
  catalog and latency holds, recall breadth grows with your mirror.
