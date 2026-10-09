# Capstone Report — Search Visibility Decline Prioritization

- **Author:** Mehdi Dardour
- **Lane:** Refresh / Content Opportunity Scoring (Search Intelligence)
- **Repo:** https://github.com/dardourmehdi/FlyrankAI-ML
- **Date:** October 9, 2026

## 1. Problem framing

FlyRank editors have limited time to investigate pages at risk of losing organic-search visibility. The decision is **which content pages to review first**, as of the end of March 2026. The unit of analysis is one pseudonymous **client–content pair**, aggregated across March. The output is a **ranked risk score**; a human editor would review the highest-ranked pages before deciding whether any content update, technical investigation, or no action is appropriate. A false positive costs editorial review time; a false negative can delay investigation of a page that subsequently loses visibility. Ranking helps focus a fixed review budget and compare many measured signals consistently. This is **decision support**, not an automated refresh recommendation or a causal prediction of refresh benefit.

## 2. Data safety

The private input consists of 158,549 pseudonymized client–content observations from 46 clients, assembled from Google Search Console daily performance records. March 1–31, 2026 supplies features; April 1–30, 2026 supplies the retrospective outcome. Only observations present in both month-level aggregates were retained. The five March predictors are `impressions`, `clicks`, `ctr`, `avg_position`, and `active_days`.

`april_impressions` and `is_future_decline` must never be features: they contain future or target information. Similarly, `trend_direction`, `trend_pct`, or any post-decision refresh action must be excluded if present in other data tables. `client_hash_id` is used **only to create client-disjoint partitions**; `content_hash_id` is used only to check uniqueness. Neither identifier is a model predictor. No client names, URLs, private queries, tokens, raw data, row-level scores, or identifiers belong in the public paper, `docs/`, or committed `work/` files. **Before submission**, inspect `work/` and `git diff --cached` to independently confirm that rule. The private `flyrank_march_april_model_data.csv` must remain excluded from Git tracking.

## 3. Baseline

The deliberately transparent baseline sorts pages by **descending `log(1 + March impressions)`**. It uses information available at the March decision point and tests whether a five-feature ML ranker adds value beyond a simple visibility-based queue. Both baseline and random forest were scored on the **same 23,235 test pages** from 12 held-out clients.

| Held-out metric | Visibility baseline | Random forest |
|---|---:|---:|
| Precision@20 | 55.0% | 75.0% |
| Precision@50 | 36.0% | 80.0% |
| Precision@100 | 35.0% | 72.0% |
| Precision@500 | 37.2% | 70.2% |
| ROC AUC | 0.570 | 0.701 |
| Average precision | 0.492 | 0.636 |

This baseline is **not** the earlier Week 4 refresh-score benchmark, which used a different dataset and target; those historical numbers should not be presented as directly comparable.

## 4. Model / analysis

A **RandomForestClassifier** assigns a probability-like ranking score to each March page. A nonlinear forest is a reasonable exploratory model for the five tabular performance signals; no language model or private identifiers are needed. The exact input features are `impressions`, `clicks`, `ctr`, `avg_position`, and `active_days` (March only). Numeric median imputation is included in the model pipeline. Hyperparameters are `n_estimators=150`, `min_samples_leaf=10`, `max_features='sqrt'`, `random_state=42`, `n_jobs=-1`. They were fixed for this comparison, not optimized on the holdout.

**Proxy target:** `is_future_decline = 1` when **April impressions are strictly less than 80% of March impressions**; otherwise, it is 0. This measures a subsequent drop greater than 20%, not its cause. March features have a clear information cutoff. Pseudonymous IDs, April values, labels, and any label-derived fields are excluded from `X`.

## 5. Evaluation

A single `GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)` separates complete clients: **34 training clients, 135,314 pages** and **12 test clients, 23,235 pages**, with no client overlap. This controls client-level overlap, although it is **not** an out-of-time replication; features refer to March and outcomes to April for both groups.

The **overall observed positive-label rate is 47.8%**, and the **held-out rate is 47.0%**. For context, the held-out majority-class rate is about **53.0%** (not a ranking baseline). The model's Precision@50 is **80.0%**, compared with **36.0%** for the baseline and **47.0%** held-out prevalence: a roughly **1.70× precision lift over prevalence**, with a **44-percentage-point** improvement over the tested baseline. ROC AUC improves from **0.570 to 0.701**, which provides discrimination evidence beyond top-k alone.

**Error analysis and operational caution:** At K=50, **10 of the 50** top-ranked model pages did **not** satisfy the measured decline label (false positive reviews under this proxy); the baseline had **32 such pages**. Precision@50 alone does not identify all future declines or quantify the model's total false negatives, so recall should not be inferred from that figure. Global top-k ranks may favor large clients. Among the **7 held-out clients with at least 50 eligible pages**, macro-averaged per-client Precision@50 is **50.3% for the model versus 23.1% for the baseline**. This spread between global and per-client metrics warns against assuming equal client benefit. No uncertainty intervals, threshold-level confusion matrix, or additional client/time splits were measured here.

## 6. Interpretation

The model provides better retrospective prioritization than the tested visibility-only rule. Random-forest impurity-based importance ranks:

| Feature | Importance |
|---|---:|
| `avg_position` | 0.324 |
| `impressions` | 0.291 |
| `active_days` | 0.190 |
| `ctr` | 0.131 |
| `clicks` | 0.064 |

This suggests reported search position and exposure are useful **within this fitted model**, but impurity importance is not a causal explanation. Correlated features and client differences can affect those rankings. A noteworthy limitation is that strong **global** Precision@50 (80.0%) coexists with materially lower **macro per-client** Precision@50 (50.3%); the aggregate metric is not a guarantee of uniform results. No negative causal effect, refresh uplift, or time-series stability was tested.

## 7. Recommendation

Run an **editor-reviewed pilot** rather than auto-refreshing pages. At the end of a reporting month, build only that month's eligible pre-outcome features, score the page queue, and present top-ranked pages alongside their current impressions, CTR, position, and available history. Editors should first check measurement availability, site changes, query mix, seasonality, and whether a refresh is even appropriate; record the reason for any action. Apply a review budget and monitor outcomes by client rather than claiming the pooled top-k number applies equally to everyone.

Confidence is **moderate for retrospective discrimination on this one client-disjoint March–April test**, and **limited for future deployment**. Important constraints: (1) an April **inner join** excludes March pages without April observations (survivorship bias); (2) small March impression counts make a relative-drop label unstable; (3) only one adjacent month pair and one client split were evaluated; (4) GSC trends alone do not explain the cause of decline; and (5) observing a drop does not prove a refresh will reverse it. Next steps are prospective monthly validation, repeated client-disjoint splits, uncertainty estimates, review-cost-aware metrics, and a randomized or otherwise causally defensible intervention study if business lift is to be claimed.

## 8. Reproducibility

From a fresh clone of the repository, using **Python 3.10+** and a private, authorized local copy of `flyrank_march_april_model_data.csv`:

```bash
git clone https://github.com/dardourmehdi/FlyrankAI-ML.git
cd FlyrankAI-ML
python -m venv .venv
source .venv/bin/activate 
python -m pip install numpy pandas scikit-learn
python run_capstone.py --csv /ABSOLUTE/PRIVATE/PATH/flyrank_march_april_model_data.csv --out outputs
```

The script must be copied into the repository root as `run_capstone.py` (as supplied in the accompanying capstone package). If the script has **not** been committed yet, these commands cannot run from a fresh clone; finish the repository integration first. The tested split and forest use **random seed 42**. Core packages are `numpy`, `pandas`, and `scikit-learn`; record exact versions via `python -m pip freeze > requirements-lock.txt` **in the local environment**, review that output, and commit only approved dependency metadata if required. The generated `outputs/metrics.json` contains aggregate validation results; the input CSV and client/page-level outputs remain private. The paper is in `paper.md`; the static report is `docs/index.html`. Deployment to GitHub Pages and the URL in `submission/paper_url.txt` must be performed and verified separately.

---

**Claims checklist:** All quantitative claims above refer to the measured March→April evaluation; reported outcomes are **observed**, **measured**, and **directional**. This is **decision-support**, not a causal statement, not a reconstruction of Google's algorithm, and not proof of performance on future clients or dates. Before submission, rerun `run_capstone.py`, compare `outputs/metrics.json`, inspect staged files for identifying information, and confirm the published page contains only safe aggregate information.
