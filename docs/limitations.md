# Dataset and method limitations

- **Period and coverage.** Orders from Sept 2016 to Oct 2018; 2016 and Sep-Oct 2018 are
  partial, so trend and cohort analyses use Jan 2017 - Aug 2018. The dataset is a sample
  Olist published, not the whole business.
- **No costs, commissions or margins.** Marketplace revenue and contribution are estimates
  from labelled assumptions (take rate, payment cost); they are not observed profit.
- **No traffic data.** Visits and sessions are absent, so conversion cannot be measured; the
  simulator and scenarios express conversion as a change in orders.
- **No returns or refunds.** GMV is gross of returns.
- **One review per order, mostly.** Review scores are post-delivery surveys answered by a
  subset of customers; non-response is not random.
- **Identity.** `customer_unique_id` is Olist's de-duplicated person key; we cannot verify
  it. A person can order from several addresses (state = first order's state).
- **Observational data.** Associations (late deliveries vs reviews, repeat rates after late
  first orders) are not causal effects; scenarios that use them say so and are upper bounds.
- **Low repeat rate.** About 3% of customers ever order again, which limits cohort analysis,
  CLV and churn models; predictions are useful in aggregate and for ranking, not per person.
- **Right-censoring.** Late cohorts and late orders have less follow-up; the retention curve
  and the repeat-gap estimate only use cohorts/orders observed long enough, and the delivery
  model test window stops in July 2018.
- **Simulated experiment.** The A/B test is a simulation on real baskets with assumed effects.
- **Licence.** Data: CC BY-NC-SA 4.0 (non-commercial); derived artefacts are shared under the
  same terms. Code: MIT.
