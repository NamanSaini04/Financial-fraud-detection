# Fraud Risk Insights — Executive Summary

> **Audience:** fraud-ops lead / risk manager (non-technical).
> **Dataset:** ULB credit-card transactions — 283,726 transactions (after
> de-duplication), 473 confirmed fraud (**0.17%**). European cardholders, 2 days.
> **Bottom line:** a tuned XGBoost model flags ~84% of fraud while sending only
> ~1 in 480 transactions to manual review (119 of 56,746 on the test set),
> cutting estimated fraud cost ~20% vs a naive cutoff. All numbers below are from
> a held-out test set, not training data.

---

## 1. The problem in one chart
Fraud is **0.17%** of transactions. A model that blindly labels everything
"legitimate" would be **99.8% accurate and catch zero fraud**. This is why we
ignore accuracy entirely and measure **recall** (fraud caught) and **precision**
(flags that are real) instead. *(See `reports/figures/01_class_balance.png`.)*

## 2. Three findings that matter

### 🕑 Finding 1 — Fraud concentrates at night
Fraud **rate** peaks in the small hours despite that being the *quietest* traffic:

| Hour | Fraud rate | vs. 0.17% baseline |
|------|-----------|--------------------|
| 2 AM | 1.45% | **~8.7×** |
| 4 AM | 1.04% | **~6.2×** |

**So what:** a transaction at 2–4 AM is far riskier per-dollar. Time-of-day is a
cheap, interpretable risk signal a rules layer could use *today*, independent of
the model.

### 💳 Finding 2 — Fraud hides in small amounts
Median fraud amount is **$9.25** vs **$22.00** for legitimate transactions —
consistent with *card-testing* (criminals validate a stolen card with a tiny
charge before a big one). A few large frauds pull the *mean* up ($122 vs $88),
which is why median, not mean, tells the real story.

**So what:** don't rely on "large amount = fraud." Most fraud here is small. A
high-amount rule alone would miss the bulk of it.

### 📊 Finding 3 — A few behavioural signals separate fraud cleanly
Anonymised behavioural features (V14, V17, V12, V10) separate fraud from
legitimate activity with little overlap, which is *why* the model works at all.
These are PCA-transformed for privacy, so we **cannot** name them in business
terms — an honest limitation, not a gap in the analysis.

## 3. The model & the decision

**Model:** XGBoost (gradient-boosted trees) beat Logistic Regression, Random
Forest, and SVM on PR-AUC (**0.82**). But a model outputs a *probability* — the
business value is in **where we draw the line**.

**The threshold is a business lever, not a technical default:**

| Strategy | Fraud caught | Missed (FN) | False alarms (FP)* | Est. cost** |
|----------|-------------|-------------|--------------------|-------------|
| Default cutoff (0.50) | 79% | 20 | 12 | $3,060 |
| "Catch 90% at any cost" | 91% | 9 | **3,558** | $19,140 |
| **Cost-optimal (0.12)** ✅ | **84%** | 15 | 39 | **$2,445** |

\* counts on the 56,746-transaction held-out test set. \*\* assumes a missed fraud
costs $150, a false alarm $5 — **illustrative; replace with real figures from finance.**

**Recommendation:** operate at the **0.12 threshold**. Versus the default it
catches 5 more frauds for 27 more reviews — worthwhile because a missed fraud
costs ~30× a false alarm. It lowers estimated cost ~20%. Chasing maximum recall
is actively harmful: it sextuples cost via an unmanageable review queue.

## 4. What this would mean operationally
- **Analyst workload:** 119 flagged transactions (80 real fraud + 39 false
  alarms) across the 56,746-transaction test window — a manageable queue,
  versus 3,644 flags from the max-recall setting.
- **Tunable to capacity:** if the review team grows or shrinks, move the
  threshold — no retraining needed.
- **Layer with rules:** combine the model with simple, explainable rules
  (e.g. extra scrutiny on 2–4 AM transactions) for defence in depth.

## 5. Limitations (read before trusting this)
- **Not production-ready.** This is offline analysis on historical data — no
  real-time scoring, monitoring, or retraining loop.
- **Concept drift.** Fraud tactics evolve; a model trained on past data decays.
  Production use requires periodic retraining and performance monitoring.
- **Anonymised features.** PCA columns mean we can rank importance but not
  explain *why* a transaction scored high — a constraint for analyst trust and
  for any regulatory "reason code" requirement.
- **Small fraud sample.** Only ~378 fraud cases in training. SMOTE can rebalance
  counts but cannot invent genuinely new fraud patterns — there is a real
  ceiling on learnable signal.
- **Cost assumptions are placeholders.** The recommended threshold shifts with
  the true FN/FP costs, which must come from the business.
- **One dataset, one region, two days.** Not representative of all fraud, all
  geographies, or seasonal effects.

## 6. Suggested next steps
1. Replace placeholder FN/FP costs with finance's real figures; re-tune threshold.
2. Pilot in *shadow mode* (score live traffic, don't act) to validate before
   any automated decline.
3. Add monitoring for score drift and monthly retraining.
4. Explore richer data (device, geo, velocity) — the PaySim dataset is the
   planned next step for interpretable, behaviour-rich features.
