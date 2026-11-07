"""Features, graphs and models of notebook 03."""
import copy
import json

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def snaps(gnn):
    val, test = gnn["eval_snapshots"](gnn["data"], 1960, gnn["CFG"]["split"])
    train = gnn["train_snapshots"](gnn["data"], 1960, gnn["CFG"]["split"])
    return {"train": train[-1], "test": test}


def test_artist_features_are_time_aware(gnn, snaps):
    data, feats = gnn["data"], gnn["feats"]
    test = snaps["test"]
    x = feats.artist(test)
    exh_count = np.expm1(x[:, -4]).round()
    assert np.array_equal(exh_count, np.bincount(test.hist_artist[1], minlength=data.num_artists))
    assert np.array_equal(x[:, -1] > 0, exh_count > 0)
    works = np.expm1(x[:, -7]).round()
    acq = data.acquisitions
    assert np.array_equal(works, np.bincount(acq.loc[acq["year"] < test.ref_year, "artist"],
                                             minlength=data.num_artists))
    # Earlier snapshot, fewer works: nothing acquired later leaks in.
    earlier = np.expm1(feats.artist(snaps["train"])[:, -7]).round()
    assert (earlier <= works).all() and earlier.sum() < works.sum()


def test_graph_hides_target_artists(gnn, snaps):
    test = snaps["test"]
    g = gnn["build_hetero"](test, gnn["feats"])
    has_artist = g[gnn["HAS_ARTIST"]].edge_index
    assert set(has_artist[0].tolist()).isdisjoint(test.targets.tolist())
    assert set(test.target_curator[0]) <= set(g[gnn["CURATED_BY"]].edge_index[0].tolist())
    leaky = gnn["build_hetero"](test, gnn["feats"], leak=True)
    assert leaky[gnn["HAS_ARTIST"]].edge_index.shape[1] > has_artist.shape[1]
    none = gnn["build_hetero"](test, gnn["feats"], include_curators=False)
    assert "curator" not in none.node_types


def test_coexhibition_edges(gnn):
    hist = np.array([[0, 0, 0, 1, 1], [5, 6, 7, 5, 8]])
    edges = gnn["coexhibition_edges"](hist)
    pairs = set(map(tuple, edges.T.tolist()))
    assert pairs == {(5, 6), (6, 5), (5, 7), (7, 5), (6, 7), (7, 6), (5, 8), (8, 5)}


def small_cfg(gnn, **model):
    cfg = copy.deepcopy(gnn["CFG"])
    cfg["model"].update(hidden=16, layers=1, **model)
    return cfg


@pytest.mark.parametrize("name,mode", [("rgcn", "full"), ("rgcn", "none"), ("heteroconv", "full"),
                                       ("hgt", "full"), ("sage", "none"), ("gat", "none"),
                                       ("mlp", "none")])
def test_model_forward_backward(gnn, snaps, name, mode):
    torch.manual_seed(0)
    cfg = small_cfg(gnn, name=name, curator_mode=mode)
    builder = gnn["GraphBuilder"](gnn["feats"], cfg["model"])
    p = builder.prepare(snaps["train"])
    model = gnn["make_model"](cfg["model"], p.graph, gnn["data"].num_curators)
    scores = model(p.graph, p.targets, p.candidates)
    assert scores.shape == (len(p.targets), len(p.candidates))
    loss = gnn["ranking_loss"](scores, p.labels, "bpr", 5, torch.Generator().manual_seed(0))
    loss.backward()
    grads = [q.grad for q in model.parameters() if q.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)


def test_curator_ids_get_gradient(gnn, snaps):
    cfg = small_cfg(gnn, name="rgcn", curator_mode="full", curator_ids=True)
    builder = gnn["GraphBuilder"](gnn["feats"], cfg["model"])
    p = builder.prepare(snaps["train"])
    model = gnn["make_model"](cfg["model"], p.graph, gnn["data"].num_curators)
    emb = model.encoder.proj.ids["curator"].weight
    assert emb.shape == (gnn["data"].num_curators, 16) and (emb == 0).all()
    gnn["ranking_loss"](model(p.graph, p.targets, p.candidates), p.labels, "bpr", 5,
                        torch.Generator().manual_seed(0)).backward()
    target_curators = np.unique(snaps["train"].target_curator[1])
    assert (emb.grad[target_curators].abs().sum(1) > 0).any()


def test_ranking_loss_prefers_correct_order(gnn):
    labels = torch.zeros(2, 50, dtype=torch.bool)
    labels[0, :3] = labels[1, 10:12] = True
    good = labels.float() * 5
    gen = lambda: torch.Generator().manual_seed(0)
    for kind in ["bpr", "bce"]:
        assert gnn["ranking_loss"](good, labels, kind, 10, gen()) < \
            gnn["ranking_loss"](-good, labels, kind, 10, gen())
    assert gnn["ranking_loss"](good, torch.zeros_like(labels), "bpr", 10, gen()) is None


def test_run_experiment_saves_and_resumes(gnn, tmp_path, monkeypatch):
    monkeypatch.setitem(gnn, "RESULTS_DIR", tmp_path)
    cfg = small_cfg(gnn, epochs=2, eval_every=1)
    rows = gnn["run_experiment"]("rgcn", "full", 1950, 0, cfg=cfg)
    out = tmp_path / "gnn"
    for suffix in ["per_exh", "ranks", "cold_per_exh", "cold_ranks"]:
        assert (out / f"rgcn_full_T1950_s0_{suffix}.csv").exists()
    info = json.loads((out / "rgcn_full_T1950_s0_info.json").read_text())
    assert {"curated", "all", "cold_curated", "cold_all"} <= set(info["summary"])
    assert {r["subset"] for r in rows} == set(info["summary"])
    # Same config: loaded, not retrained.
    (out / "rgcn_full_T1950_s0_model.pt").unlink()
    again = gnn["run_experiment"]("rgcn", "full", 1950, 0, cfg=cfg)
    assert again == rows and not (out / "rgcn_full_T1950_s0_model.pt").exists()
    # Changed config: retrained.
    cfg["model"]["hidden"] = 8
    gnn["run_experiment"]("rgcn", "full", 1950, 0, cfg=cfg)
    assert (out / "rgcn_full_T1950_s0_model.pt").exists()
