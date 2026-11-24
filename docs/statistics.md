# Statistical methodology

All numbers below are read from committed result files; rerun with `make analysis`.

## Toolkit (`src/olist/stats/`)

| Module | Contents | Notes |
|---|---|---|
| `core.py` | descriptive summary with normal CI; Wilson interval; two-proportion z-test with Wald CI and Cohen's h; percentile bootstrap (one- and two-sample); permutation test; Bonferroni, Holm and Benjamini-Hochberg; sample size, power and MDE for proportions and means | numpy + `statistics.NormalDist` only, so the in-browser dashboard uses it; each function is tested against SciPy/statsmodels (`tests/test_stats_core.py`) |
| `inference.py` | Welch t, Mann-Whitney U, chi-square; group comparison with bootstrap CI, Cohen's d and Cliff's delta; stratified mean difference with Holm-adjusted per-stratum tests and an inverse-variance pooled estimate | SciPy; pipeline and notebooks |
| `effect_sizes.py`, `abtest.py` | the original effect-size, power, SRM, Beta-Binomial and peeking helpers used by notebook 02 | kept unchanged |

## Late delivery and review score (re-verified)

Source: [`results/analytics/late_vs_on_time.json`](../results/analytics/late_vs_on_time.json),
SQL 25.

- Delivered, reviewed orders: **late 2.27★ (n = 6,381)** against **on time 4.29★
  (n = 89,443)**; difference -2.02 stars, 95% bootstrap CI -2.06 to -1.98; Cohen's d -1.71;
  Cliff's delta -0.64 (large). 53.7% of late orders get 1 star against 6.6% of on-time ones.
- **Reported figure corrected:** an earlier summary of this project quoted "4.29 vs 4.27 for
  late deliveries". The data say 4.29 vs **2.27**.
- **Stratified:** within each customer state with at least 30 late and 30 on-time orders
  (21 of 27 states) the gap is negative and significant after Holm correction in all 21;
  pooled -2.03 (CI -2.07 to -1.99). Within product categories (29 of 74 qualify): negative and
  significant in all 29; pooled -2.02.
- **Not causal.** Lateness is not randomised: late orders cluster in regions, seasons
  (Nov 2017, Feb-Mar 2018), categories and sellers. Stratification removes one confounder at
  a time. Notebook 02 adds an ordinal regression with category, state, basket and value
  controls (odds ratio 0.068 for a higher score). The consistent sign across every stratum
  makes a large effect plausible, but the right test is an experiment (e.g. proactive
  delay messages).

## Uncertainty for rates and rankings

- Rates on small samples use **Wilson** intervals (they stay inside [0, 1] and behave at 0/n).
- **Seller ranking** (`olist/analytics/sellers.py`): sellers need 30 delivered orders to be
  ranked (627 of 3,095 sellers, 76.8% of item revenue). Late, 1-2 star and cancellation rates
  are shrunk with an empirical-Bayes Beta prior (method of moments on ranked sellers: the
  between-seller variance is the observed variance minus the binomial sampling variance);
  mean reviews are shrunk with a normal-normal model. Score = -0.35 z(late) - 0.35 z(bad
  reviews) - 0.10 z(cancellations) + 0.20 z(log revenue). 400 Monte Carlo draws from each
  seller's Beta posteriors give a 90% rank interval and P(top decile). Only 8 of the 25
  largest sellers by revenue make the top decile; the median 90% rank interval spans 337
  places, which is the honest message: most mid-table differences are noise.

## Multiple testing

Per-stratum tests use Holm (family-wise error). Benjamini-Hochberg is available for
exploratory screens (false discovery rate). The A/B analysis has one pre-declared primary
metric; secondary metrics and the guardrail are reported with CIs and are not used to
declare success.

## Customer lifetime value

Source: [`results/analytics/clv.json`](../results/analytics/clv.json).

- **Observed value**: GMV of valid orders before 2018-09-01: R$ 165.71 per customer; the first
  order is 97.8% of all observed GMV.
- **Predicted value**: BG/NBD (Fader, Hardie & Lee 2005) for repeat purchase occasions
  (same-day orders merged, since 30% of second orders are same-day split baskets) and
  Gamma-Gamma for GMV per repeat occasion (fitted on 2,063 repeaters; the frequency-value
  correlation is -0.001, so the independence assumption holds). Both are maximum likelihood
  with SciPy, tested by parameter recovery on simulated data.
- **Temporal validation**: fitted on purchases before 2018-03-01 (56,622 customers, 1,012
  repeaters), predicting Mar-Aug 2018: **739 repeat purchases predicted vs 720 actual**
  (R$ 110.8k vs R$ 107.1k GMV). A constant-rate baseline predicts 1,274. Per customer the
  model ranks returners only modestly (ROC-AUC 0.588, PR-AUC 0.032 vs base rate 0.012).
- **12-month forecast** from 2018-09-01: R$ 2.48 of expected GMV per customer, about 1.5% of
  observed value. Use: aggregate planning of repeat revenue; not per-customer promises.

## Delivery-time and late-delivery model

Source: [`results/analytics/delivery_model.json`](../results/analytics/delivery_model.json).

- Features known at purchase: Olist's promised days, distance, states and lane, category,
  basket, weight, freight, purchase month/weekday/hour, orders placed in the previous 7 days,
  and the seller's and lane's history of deliveries completed **before** the purchase
  (`merge_asof` on delivery time; a test re-derives the seller history by brute force).
- Split by purchase date: train 2017, validation Jan-Mar 2018, test Apr-Jul 2018 (August is
  excluded: late-August orders may not have been delivered before the extract ended).
- Candidates (L2 vs L1 objective; with or without the seasonal/demand features) are compared
  on the validation window only; the chosen setting and the lane baseline are refitted on
  train + validation and the test window is scored once.
- Test results: delivery days MAE **5.20** (95% bootstrap CI 5.14-5.26) for LightGBM, 5.65
  for the lane-history mean, 14.56 for Olist's promise (which overshoots by 13.9 days on
  average). Late-delivery PR-AUC **0.123** (0.109-0.141) vs 0.069 for the lane's historical
  late rate (base rate 4.0%); ROC-AUC 0.73 vs 0.63; the model's top decile catches 35% of late
  orders. Fitted on 2017 only, LightGBM still leads (MAE 4.71 vs 5.16; PR-AUC 0.100 vs 0.062).
- Limits: probabilities are not calibrated after the early-2018 surge (Brier 0.047 vs 0.041
  for the lane rate); the demand feature needs a live feed; the model has never been used to
  change promises, so its business effect is untested.
