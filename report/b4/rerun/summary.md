# B4 relevance experiments

All comparisons use the supported-context cohort.
Schema agreement is deferred. Context catalogues are not observed runtime requests.
Seed intervals are 95% t-intervals over independent seeds (not clipped to [0,1]).
Per-run CSVs also contain bootstrap intervals from the shared harness.

Configuration selected on validation only: **relevance**.

| Experiment | Eval set | AUROC mean | Seed SD | 95% seed CI |
|---|---|---:|---:|---|
| pair_only | mcptox | 0.4855 | 0.0822 | [0.2812, 0.6897] |
| pair_only | msb | 0.2639 | 0.1359 | [-0.0738, 0.6016] |
| pair_only | test | 0.9901 | 0.0048 | [0.9780, 1.0021] |
| relevance | mcptox | 0.4246 | 0.0786 | [0.2293, 0.6198] |
| relevance | msb | 0.2472 | 0.1067 | [-0.0180, 0.5123] |
| relevance | test | 0.9901 | 0.0040 | [0.9802, 1.0001] |
| relevance_bipia | mcptox | 0.3858 | 0.0551 | [0.2489, 0.5227] |
| relevance_bipia | msb | 0.1922 | 0.0478 | [0.0735, 0.3109] |
| relevance_bipia | test | 0.9307 | 0.0068 | [0.9138, 0.9477] |
| text_control | mcptox | 0.3597 | 0.0598 | [0.2111, 0.5084] |
| text_control | msb | 0.1543 | 0.0089 | [0.1322, 0.1763] |
| text_control | test | 0.9939 | 0.0003 | [0.9931, 0.9948] |
