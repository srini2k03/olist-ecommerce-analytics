# A/B testing methodology (simulated)

Olist never ran this experiment. The simulator (`src/olist/experiments/`) shows how the
analysis and the decision would work, using **real 2018 baskets** and **labelled
assumptions** for the effects. Results: [`results/analytics/ab_test.json`](../results/analytics/ab_test.json);
interactive version: the dashboard's A/B testing page.

## Design

- **Treatment**: free shipping on orders whose item value (excluding freight) is at least
  R$ 150; the marketplace pays the carrier freight of those orders.
- **Unit and assignment**: orders, randomised 50/50 (visits are not in the data).
- **Primary metric**: share of orders with 2+ items (baseline 9.9% in 2018).
- **Secondary**: customer-paid GMV per order, item value per order, items per order.
- **Guardrail**: mean review score (no more than 0.05 stars worse at the CI lower bound).
- **Economics**: subsidy per order, contribution = take rate x item value - payment cost -
  subsidy, at 100% rollout per month.
- **Sample size**: two-proportion z-test, alpha 0.05, power 0.8, minimum detectable relative
  lift 15% -> 6,763 orders per arm, 62 days at the Jan-Aug 2018 rate of 220 valid orders per
  day (the dashboard recomputes this for whatever period and filters are selected).

## Simulation mechanics

Both arms resample real orders (item value, freight, item count, review). In treatment, a
single-item order becomes a two-item order with the probability that yields the assumed lift
(the extra item is another real item, with its price and freight); qualifying orders get
their freight paid. An assumed volume effect can add treatment orders. The sample-ratio
check uses the assignment counts. Bootstrap CIs (percentile) for every mean difference; a
Beta-Binomial posterior gives P(treatment better) and the expected loss.

## Decision engine

`decision.decide` ships only if all hold:

1. no sample-ratio mismatch (p >= 0.001);
2. the primary metric improved significantly (p < 0.05 and positive);
3. the review guardrail is not breached;
4. P(incremental contribution > 0) >= 80% (bootstrap).

Otherwise "do not ship" (guardrail breached or P(contribution > 0) < 20%, or significant lift
with contribution probably negative) or "inconclusive".

## Result with the default assumptions

Assumed +15% multi-item lift, no volume effect, 10% take rate, 2% payment cost: in this draw
the lift is detected (9.4% -> 11.9%, p < 0.001), but **the subsidy costs R$ 9.26 per
treatment order** because about 24% of orders already qualify and get free shipping without
changing behaviour. Incremental contribution
at rollout: about **-R$ 58.5k per month** (95% CI -R$ 62.6k to -R$ 54.3k), P(> 0) = 0% ->
**do not ship**. The sensitivity grid (lift 0-50% x take rate 5-20%) is negative everywhere
(-R$ 64.7k to -R$ 58.5k per month); in the tornado the threshold (R$ 100-250) swings the
result by R$ 56.6k per month, every other assumption by less than R$ 2.5k. This agrees with notebook 02
(subsidy R$ 9.60 per order vs about R$ 1.44 of extra GMV).

## What would make it worth testing

A higher threshold (fewer already-qualifying orders), a subsidy funded by sellers, or a
target on the multi-item rate large enough that the extra item value times the take rate
covers the subsidy. Pre-register the primary metric, run the full planned duration (notebook
02 shows that peeking 20 times inflates the false-positive rate from 5% to 23%), and check
SRM before reading any metric.
