import numpy as np
import pandas as pd
import pytest

from olist.model.evaluate import gains_table, ranking_metrics, reason_codes, shap_matrix
from olist.model.features import build_training_set
from olist.model.train import make_lgbm, make_logreg, recency_rule_score, to_lgbm_frame
from olist.segmentation import fit_kmeans, k_diagnostics, name_segments, rfm_table, transform


def test_gains_table_and_capture():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.arange(10)[::-1].astype(float)  # positives ranked first
    g = gains_table(y, score, n_bins=5)
    assert g.cumulative_gain.iloc[0] == 1.0
    assert g.cumulative_gain.iloc[-1] == 1.0
    m = ranking_metrics(y, score)
    assert m["roc_auc"] == 1.0 and m["capture_top10"] == 0.5


def test_models_train_and_explain(synthetic_orders):
    tr = build_training_set(synthetic_orders, ["2017-09-01", "2018-01-01"], 120)
    lr = make_logreg().fit(tr, tr.label)
    assert lr.predict_proba(tr)[:, 1].shape == (len(tr),)
    X, cats = to_lgbm_frame(tr)
    gbm = make_lgbm(n_estimators=30, min_child_samples=20, min_data_per_group=20).fit(X, tr.label)
    sv = shap_matrix(gbm, X.head(50))
    assert sv.shape == (50, X.shape[1])
    reasons = reason_codes(sv, X.head(50), top_n=3)
    assert len(reasons) == 50 and all(r.count(";") == 2 for r in reasons)
    assert (recency_rule_score(tr) <= 0).all()
    # Unseen categories at scoring time become NaN, not an error.
    X2, _ = to_lgbm_frame(tr.assign(customer_state="ZZ"), cats)
    assert X2.customer_state.isna().all()


def test_segmentation_pipeline(synthetic_orders):
    rfm = rfm_table(synthetic_orders, "2018-09-01")
    X = transform(rfm)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)
    diag = k_diagnostics(X, range(2, 4), sample=1000)
    assert diag.silhouette.between(-1, 1).all()
    km = fit_kmeans(X, 3)
    assert len(set(km.labels_)) == 3


def test_segment_naming_rules():
    profile = pd.DataFrame(
        {
            "recency_days": [400, 100, 300, 120, 420],
            "frequency": [1.0, 1.0, 2.1, 1.0, 1.0],
            "monetary": [300, 300, 290, 70, 70],
        }
    )
    names = name_segments(profile)
    assert names == {
        0: "At-risk big spenders",
        1: "Recent big spenders",
        2: "Champions",
        3: "Recent low spenders",
        4: "One-and-done (lapsed, low value)",
    }


@pytest.mark.parametrize("bad", [[], None])
def test_ranking_metrics_rejects_empty(bad):
    with pytest.raises((ValueError, TypeError)):
        ranking_metrics(bad, bad)
