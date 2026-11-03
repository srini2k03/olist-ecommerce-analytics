# Executive summary: Olist marketplace, Jan 2017 to Aug 2018

**Bottom line.** Olist more than doubled in a year (Jan–Aug 2018 GMV +140% on the same months of 2017), but **growth flattened in 2018** (mean month-on-month −1.0%). Almost nobody buys twice: **3.0%** of customers ever do. Further growth has to come from keeping customers. Our analysis points to **late delivery** as the main controllable reason they do not return.

*Scope:* 99,441 orders from the public Olist dataset. Figures come from executed SQL and notebooks in this repository. The A/B test in recommendation 4 is **simulated**, with inputs calibrated to real data.

## Recommendations

**1. Make "delivered by the promised date" the top customer-experience KPI, and fix it lane by lane.**
- **Finding:** late orders average **2.27★**, against **4.29★** for on-time orders. Controlling for category, state, basket and value, being late cuts the odds of a better review about 15-fold (odds ratio **0.068**, 95% CI 0.063–0.074).
- **Where:** late rates reach **21.4%** in Alagoas against **4.5%** in São Paulo, and the SP → RJ lane alone accounts for **17.6% of all late orders**.
- **Action:** pad promised dates on high-risk north-east lanes. Customers tolerate slow delivery but punish late delivery; arriving 1–3 days late already drops the mean to 3.29★. Renegotiate or switch the SP → RJ carrier.
- **Size of the prize:** if the 6.7% of late orders were reviewed like on-time ones, the platform's mean review would rise by about **0.13★** (6.7% × 2.02★).

**2. Run a win-back programme for "At-risk big spenders" before day 365.**
- **Finding:** this K-Means segment is **12.4% of customers but 38.6% of revenue** (R$ 515 average spend, about 298 days since the last order). It has the **lowest mean review of any segment (3.95)**.
- **Action:** send a personal offer in the category they bought, starting with an apology or credit where their order was late. This is the only segment where paid reactivation clearly pays.

**3. Nudge recent buyers to a second order between day 30 and day 90.**
- **Finding:** month-1 retention is **0.48%**. Of the customers who do return, **68%** place their second order within 90 days and **83%** within 180 days.
- **Action:** trigger a complementary-category bundle for "Recent mid-value buyers" (31.2% of customers, 31.4% of revenue) inside that window, and measure it against a holdout.

**4. Do not launch a R$ 150 free-shipping threshold as designed. Test a better-targeted version, and run tests properly.**
- **Finding (simulated, calibrated to real 2018 data):** the threshold lifts the multi-item rate (+1.43 pp, p = 0.006). However, the freight subsidy costs **R$ 9.60 per order** against about **R$ 1.44** of extra GMV, because **26%** of orders already qualify.
- **Action:** test a threshold just above typical single-item baskets, or a partial subsidy, and use margin as a guardrail.
- **Testing discipline:**
  - at 222 orders per day, plan about **9-week** tests (a 15% minimum detectable effect);
  - **do not peek**: checking 20 times inflates the false-positive rate from 5% to **23%**;
  - read results by week of exposure to strip out novelty effects.

**5. Use the churn score only to prioritise cheap retention actions, and prove uplift first.**
- **Finding:** repeat purchase is hard to predict. On a later, held-out window, LightGBM reaches ROC-AUC **0.583** and PR-AUC **0.020** against a 1.18% base rate. Logistic regression performs about as well. Still, **targeting the top 10% of customers ranked by the model captures 19.6% of next-180-day repeat buyers (1.96x lift)**, against 15.4% for a "most recent first" rule.
- **Action:** use the score for email and push to the top 10–20%. Run a randomised holdout before attaching discounts.

## Caveats
- The data is observational, so the delivery-to-review link is strongly supported but not experimentally proven.
- Repeat-purchase signal is weak, with no browsing or marketing data available.
- Review "sentiment" relies on keyword proxies.

Details: [notebooks](notebooks/) · [SQL snapshots](results/sql/) · dashboard: `app/streamlit_app.py`.
