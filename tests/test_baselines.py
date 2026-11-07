"""Heuristic baselines score what they claim to, using only pre-cutoff information."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def test_1960(base):
    return base["eval_snapshots"](base["data"], 1960, base["CFG"]["split"])[1]


def assert_orders_by(scores, key):
    """Sorting candidates by score must sort `key` in non-increasing order."""
    order = np.argsort(-scores, kind="stable")
    assert (np.diff(key[order]) <= 0).all()


def test_random_recall_matches_chance(base, test_1960):
    per_exh, _ = base["evaluate"](base["Random"](), test_1960, [50])
    scored = per_exh[per_exh["n_pos_in_pool"] > 0]
    chance = 50 / len(test_1960.candidates)
    assert abs(scored["recall@50"].mean() - chance) < 0.01


def test_popularity_counts_history(base, test_1960):
    scores = base["Popularity"](base["data"]).score(test_1960)
    counts = np.bincount(test_1960.hist_artist[1], minlength=base["data"].num_artists)
    assert np.array_equal(scores[0], counts[test_1960.candidates])


def test_curator_collab_counts_shared_exhibitions(base, test_1960):
    data = base["data"]
    scores = base["CuratorCollab"](data).score(test_1960)
    ae, ce = data.artist_edges, data.curator_edges
    hist = set(test_1960.hist_artist[0])
    curated = [i for i, e in enumerate(test_1960.targets) if (ce["exh_id"] == e).any()][:20]
    with_signal = 0
    for i in curated:
        curators = ce.loc[ce["exh_id"] == test_1960.targets[i], "curator"]
        their_exh = ce[ce["curator"].isin(curators) & ce["exh_id"].isin(hist)]
        # Sum over the target's curators of shows each shared with the artist.
        shared = ae.merge(their_exh, on="exh_id").groupby("artist").size()
        key = shared.reindex(test_1960.candidates, fill_value=0).to_numpy()
        assert_orders_by(scores[i], key)
        with_signal += key.max() > 0   # zero when the curators have no earlier shows
    assert with_signal >= 5


def test_collection_popularity_is_time_aware(base, test_1960):
    data = base["data"]
    scores = base["CollectionPopularity"](data).score(test_1960)
    acq = data.acquisitions
    before = np.bincount(acq.loc[acq["year"] < 1960, "artist"], minlength=data.num_artists)
    assert_orders_by(scores[0], before[test_1960.candidates])
    # An artist whose works were all acquired later must not get credit for them.
    later_only = np.setdiff1d(acq.loc[acq["year"] >= 1960, "artist"], acq.loc[acq["year"] < 1960, "artist"])
    pos = np.flatnonzero(np.isin(test_1960.candidates, later_only))
    assert len(pos) and (before[test_1960.candidates[pos]] == 0).all()


def test_heuristics_beat_random(base, test_1960):
    data = base["data"]
    for scorer in [base["Popularity"](data), base["CuratorCollab"](data),
                   base["AdamicAdar"](data)]:
        per_exh, _ = base["evaluate"](scorer, test_1960, [20])
        assert base["summarize"](per_exh)["curated"]["map@20"] > 0.03, scorer.name


def test_cold_start_baselines(base, test_1960):
    data = base["data"]
    cold = base["cold_start_snapshot"](data, test_1960)
    top = base["CFG"]["features"]["top_nationalities"]
    for scorer in [base["DemographicPopularity"](data, top), base["CuratorProfile"](data, top),
                   base["CollectionPopularity"](data)]:
        scores = scorer.score(cold)
        assert scores.shape == (len(cold.targets), len(cold.candidates))
        assert np.isfinite(scores).all()
