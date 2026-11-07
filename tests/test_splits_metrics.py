"""Temporal snapshots (no leakage), cold start, metrics, bootstrap and result files."""
import numpy as np
import pytest

CUTOFFS = [1950, 1960, 1970, 1980]


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_snapshots_leak_free(base, cutoff):
    data, cfg = base["data"], base["CFG"]
    begin = data.exhibitions["begin"]
    val, test = base["eval_snapshots"](data, cutoff, cfg["split"])
    train = base["train_snapshots"](data, cutoff, cfg["split"])
    assert len(train) > 10
    for snap in [val, test, *train]:
        assert (begin.loc[snap.hist_artist[0]] < snap.history_end).all()
        assert (begin.loc[snap.hist_curator[0]] < snap.history_end).all()
        assert (begin.loc[snap.targets] >= snap.history_end).all()
        assert set(snap.target_curator[0]) <= set(snap.targets)
        assert np.array_equal(snap.candidates, np.unique(snap.hist_artist[1]))
    # Training targets end before validation targets, which end before the test window.
    first_val = begin.loc[val.targets].min()
    assert all((begin.loc[s.targets] < first_val).all() for s in train)
    assert begin.loc[val.targets].max() < begin.loc[test.targets].min()
    assert begin.loc[test.targets].dt.year.between(cutoff, cutoff + 9).all()


def test_label_matrix(base):
    _, test = base["eval_snapshots"](base["data"], 1960, base["CFG"]["split"])
    labels = test.label_matrix()
    for i in [0, len(test.targets) // 2]:
        e = test.targets[i]
        expected = np.intersect1d(test.positives[e], test.candidates)
        assert np.array_equal(test.candidates[labels[i]], expected)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_cold_start_snapshot(base, cutoff):
    data = base["data"]
    _, test = base["eval_snapshots"](data, cutoff, base["CFG"]["split"])
    cold = base["cold_start_snapshot"](data, test)
    assert len(np.intersect1d(cold.candidates, test.candidates)) == 0   # no history at all
    appearing = np.concatenate(list(test.positives.values()))
    assert np.isin(cold.candidates, appearing).all()                   # each debuts in window
    assert np.array_equal(np.union1d(cold.candidates, np.intersect1d(appearing, test.candidates)),
                          np.unique(appearing))
    assert np.array_equal(cold.targets, test.targets)
    assert cold.label_matrix().any(axis=1).sum() > 50


def test_metrics_hand_example(base):
    # Ranking: c0(+), c1(-), c2(+), c3(-), c4(-).
    m, ranks = base["ranking_metrics"](np.array([5, 4, 3, 2, 1.]),
                                       np.array([1, 0, 1, 0, 0], bool), [2, 4],
                                       np.random.default_rng(0))
    assert np.isclose(m["recall@2"], 0.5) and np.isclose(m["recall@4"], 1.0)
    assert np.isclose(m["map@2"], 0.5) and np.isclose(m["map@4"], (1 + 2 / 3) / 2)
    assert np.isclose(m["ndcg@4"], (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
    assert np.isclose(m["auc"], 5 / 6) and list(ranks) == [1, 3]


def test_metrics_perfect_and_ties(base):
    rng = np.random.default_rng(1)
    labels = np.zeros(100, bool)
    labels[:5] = True
    m, _ = base["ranking_metrics"](-np.arange(100.0), labels, [10], rng)
    assert m["map@10"] == m["recall@10"] == m["ndcg@10"] == m["auc"] == 1.0
    # All-tied scores must not favour the first candidate.
    tied = [base["ranking_metrics"](np.zeros(10), np.eye(10, dtype=bool)[0], [1], rng)[1][0]
            for _ in range(300)]
    assert 4.5 < np.mean(tied) < 6.5


def test_paired_bootstrap(base):
    rng = np.random.default_rng(0)
    a = rng.random(200)
    same = base["paired_bootstrap"](a, a)
    assert same["mean_diff"] == same["ci_low"] == same["ci_high"] == 0
    better = base["paired_bootstrap"](a + 0.1 + 0.01 * rng.standard_normal(200), a)
    assert better["ci_low"] > 0.05 and better["p_le_0"] == 0
    noise = base["paired_bootstrap"](rng.random(200), rng.random(200))
    assert noise["ci_low"] < 0 < noise["ci_high"]


def test_result_file_names(base):
    match = base["RESULT_FILE"].fullmatch
    m = match("rgcn_full_T1960_s3_per_exh.csv")
    assert (m["tag"], m["cutoff"], m["seed"], m["kind"]) == ("rgcn_full", "1960", "3", "per_exh")
    m = match("recency_popularity_T1980_cold_ranks.csv")
    assert (m["tag"], m["seed"], m["kind"]) == ("recency_popularity", None, "cold_ranks")
    assert match("rgcn_full_T1960_s0_cold_per_exh.csv")["kind"] == "cold_per_exh"
    assert match("summary.csv") is None and match("rgcn_full_T1960_s0_info.json") is None
