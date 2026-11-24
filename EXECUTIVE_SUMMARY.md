# Executive summary: Olist marketplace, Jan 2017 to Aug 2018

**Bottom line.** Olist more than doubled in a year (Jan–Aug 2018 GMV +140% on the same months of 2017), but **growth flattened in 2018** (mean month-on-month −1.0%). Almost nobody buys twice: **3.0%** of customers ever do, and the first order is **97.8%** of all customer value. Further growth has to come from keeping customers. **Late delivery** is the controllable problem most strongly tied to how customers rate their order.

*Scope:* 99,441 orders from the public Olist dataset, of which 97,905 valid orders and R$ 15.69M GMV (item revenue R$ 13.45M + freight R$ 2.24M) in Jan 2017 – Aug 2018. Every figure comes from code in this repository run on the checksum-pinned data (`results/analytics`, `results/sql`, the notebooks); the data passed 75 quality checks (11 known source issues handled in staging). The A/B test in recommendation 4 is **simulated** on real 2018 baskets with assumed effects.

## Recommendations

**1. Make "delivered by the promised date" the top customer-experience KPI, and fix it lane by lane.**
- **Finding:** late orders average **2.27★** against **4.29★** for on-time orders (gap −2.02, 95% CI −2.06 to −1.98); 54% of late orders get one star. The gap holds within every one of the 21 states and 29 categories large enough to test. With category, state, basket and value controls, being late cuts the odds of a better review about 15-fold (odds ratio **0.068**).
- **Where:** late rates reach **21.4%** in Alagoas against **4.5%** in São Paulo, and the SP → RJ lane alone has **17.6% of all late deliveries**.
- **Action:** customers tolerate slow delivery but punish late delivery (1–3 days late already drops the mean to 3.29★). Pad promised dates on high-risk lanes, escalate the SP → RJ carrier, and use the late-delivery model to message customers whose order is at risk: its top 10% of orders holds **35%** of late deliveries (PR-AUC 0.123 vs 0.069 for the lane's history, on later months than it was trained on).
- **Size of the prize:** if the 6.7% of late orders were reviewed like on-time ones, the mean review would rise by about **0.13★**. Do not expect much short-term revenue from it: customers whose first order was late return within 180 days at 1.43% against 1.96% after an on-time first order, so halving late deliveries adds only a few repeat orders a month (business simulation). It is a reputation and seller-quality lever first.

**2. Run a win-back programme for "At-risk big spenders" before day 365.**
- **Finding:** this K-Means segment is **12.4% of customers but 38.6% of revenue** (R$ 515 average spend, about 298 days since the last order) and has the lowest mean review of any segment (3.95). The transparent rule-based RFM view agrees on the shape: one-time buyers with top-quintile spend who have gone quiet are 14.7% of customers and 27.5% of revenue.
- **Action:** send a personal offer in the category they bought, starting with an apology or credit where their order was late. This is the only segment where paid reactivation is likely to pay.

**3. Nudge recent buyers to a second order between day 30 and day 90, and plan on aggregate CLV, not individual scores.**
- **Finding:** month-1 retention is **0.48%**; 30% of second orders are same-day split baskets; of real returns, most come within 180 days (83% of all second orders). A BG/NBD + Gamma-Gamma model predicted **739** repeat purchases for Mar–Aug 2018 against **720** actual (a constant-rate baseline said 1,274), so it is reliable for planning repeat revenue in aggregate, but it ranks individual customers only modestly (ROC-AUC 0.59).
- **Action:** trigger a complementary-category bundle for recent buyers inside that window, measured against a holdout. Expected 12-month repeat GMV is about R$ 2.48 per customer: budget retention offers accordingly.

**4. Do not launch a R$ 150 free-shipping threshold as designed.**
- **Finding (simulated):** even with an assumed +15% lift in multi-item orders, the subsidy costs **R$ 9.26 per treatment order** because about a quarter of orders already qualify; the estimated contribution change is **−R$ 58.5k per month** at rollout (95% CI −R$ 62.6k to −R$ 54.3k), so the decision engine says **do not ship**. It stays negative for any lift up to 50% and take rate up to 20%; only the threshold moves it materially. Notebook 02 reached the same conclusion (R$ 9.60 subsidy vs R$ 1.44 extra GMV per order).
- **Action:** if free shipping is tested, use a higher threshold or a seller-funded subsidy, pre-register the primary metric, plan about 9 weeks at 2018 volumes, check sample-ratio mismatch first and **do not peek** (20 looks inflate false positives from 5% to 23%).

**5. Rank sellers on service quality with uncertainty, and use the churn score only for cheap actions.**
- **Finding:** a ranking on shrunk late, bad-review and cancellation rates (sellers with 30+ delivered orders: 627 of 3,095, 77% of item revenue) puts only **8 of the 25 largest sellers** in its top decile, and its 90% rank intervals are wide: most mid-table differences are noise. The repeat-purchase model reaches ROC-AUC 0.581 on a later window; its top 10% captures 19.4% of next-180-day repeat buyers (1.94x lift).
- **Action:** coach or penalise sellers whose *shrunk* late rate is high, not those with a few unlucky orders. Use the churn score for email/push to the top 10–20% and run a randomised holdout before attaching discounts.

## Caveats
- The data is observational: the delivery-to-review link is strongly supported (every stratum, controls) but not experimentally proven.
- Commission, cost and traffic data are absent: marketplace revenue, contribution and conversion are labelled estimates or assumptions.
- Repeat purchase is rare (about 3%), so customer-level predictions are weak by nature.
- Review "sentiment" relies on keyword proxies.

Details: [dashboard](https://srini2k03.github.io/olist-ecommerce-analytics/app/) · [docs](docs/) · [notebooks](notebooks/) · [SQL snapshots](results/sql/) · [analytics results](results/analytics/).
