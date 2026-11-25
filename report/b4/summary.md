# B4 relevance experiments

All comparisons use the supported-context cohort.
Schema agreement is deferred. Context catalogues are not observed runtime requests.
Seed intervals are 95% t-intervals over independent seeds (not clipped to [0,1]).
Per-run CSVs also contain bootstrap intervals from the shared harness.

Configuration selected on validation only: **relevance**.

| Experiment | Eval set | AUROC mean | Seed SD | 95% seed CI |
|---|---|---:|---:|---|
| pair_only | mcptox | 0.5056 | 0.0122 | [0.4754, 0.5359] |
| pair_only | msb | 0.3348 | 0.1512 | [-0.0409, 0.7105] |
| pair_only | test | 0.9910 | 0.0017 | [0.9869, 0.9952] |
| relevance | mcptox | 0.4536 | 0.0989 | [0.2079, 0.6993] |
| relevance | msb | 0.2573 | 0.0837 | [0.0495, 0.4651] |
| relevance | test | 0.9900 | 0.0018 | [0.9856, 0.9944] |
| relevance_bipia | mcptox | 0.3973 | 0.0614 | [0.2448, 0.5498] |
| relevance_bipia | msb | 0.1750 | 0.0651 | [0.0134, 0.3367] |
| relevance_bipia | test | 0.9268 | 0.0079 | [0.9071, 0.9465] |
| text_control | mcptox | 0.3393 | 0.0488 | [0.2180, 0.4606] |
| text_control | msb | 0.1597 | 0.0216 | [0.1061, 0.2134] |
| text_control | test | 0.9928 | 0.0018 | [0.9884, 0.9972] |
