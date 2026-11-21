import numpy as np

from graph_fraud.metrics import best_f1_threshold, illicit_metrics, per_timestep_f1


def test_best_threshold_recovers_separable_data():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr = best_f1_threshold(y, s)
    assert 0.3 < thr <= 0.8
    m = illicit_metrics(y, s, thr)
    assert m["illicit_f1"] == 1.0 and m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0


def test_degenerate_labels_do_not_crash():
    assert best_f1_threshold(np.zeros(4), np.random.rand(4)) == 0.5
    m = illicit_metrics(np.zeros(4), np.array([0.1, 0.2, 0.9, 0.95]), 0.5)
    assert m["illicit_f1"] == 0.0 and np.isnan(m["roc_auc"])


def test_per_timestep_f1_shape():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.2, 0.8])
    ts = np.array([1, 1, 2, 2])
    rows = per_timestep_f1(y, s, ts, 0.5)
    assert [r["timestep"] for r in rows] == [1, 2]
    assert rows[0]["illicit_f1"] == 1.0 and rows[1]["illicit_f1"] == 0.0
