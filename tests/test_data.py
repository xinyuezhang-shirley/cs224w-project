"""Cleaning, the integer-indexed dataset and the notebooks' shared code."""
import nbformat
import numpy as np
import pandas as pd

from conftest import NB_DIR

ALL = ["01_data_eda.ipynb", "02_baselines.ipynb", "03_gnn.ipynb", "04_analysis.ipynb"]
WITH_EVAL = ALL[1:]
# Cells that must be identical wherever they appear, keyed by a line they contain.
SHARED = {"REQUIRED = {": ALL, "CFG = {": ALL, "def load_tables": ALL, "class MomaData": ALL,
          "class Snapshot:": WITH_EVAL, "def ranking_metrics": WITH_EVAL}


def test_shared_cells_identical():
    cells = {n: [c.source for c in nbformat.read(NB_DIR / n, as_version=4).cells] for n in ALL}
    for marker, notebooks in SHARED.items():
        versions = {}
        for n in notebooks:
            found = [s for s in cells[n] if marker in s]
            assert len(found) == 1, f"{marker!r} in {n}: {len(found)} cells"
            versions[n] = found[0]
        assert len(set(versions.values())) == 1, f"{marker!r} differs between notebooks"


TOY_COLUMNS = ["ExhibitionID", "ExhibitionNumber", "ExhibitionTitle", "ExhibitionBeginDate",
               "ExhibitionEndDate", "ExhibitionSortOrder", "ExhibitionURL", "ExhibitionRole",
               "ExhibitionRoleinPressRelease", "ConstituentID", "ConstituentType",
               "DisplayName", "Nationality", "Gender", "ConstituentBeginDate",
               "ConstituentEndDate", "WikidataID", "ULANID"]
TOY_ROWS = [
    # Two "No#" exhibitions that share a number but are distinct.
    (None, "No#", "B", "3/1/1950", "4/1/1950", "1", None, "Curator", "Director", "1",
     "Individual", "Curator One", "American", "Male", "1900", "1980", None, None),
    (None, "No#", "B", "3/1/1950", "4/1/1950", "1", None, "Artist", "Artist", "2",
     "Individual", "Artist Two", "Nationality unknown", "male", "1890", None, None, None),
    ("7", "No#", "A", "1/5/1940", None, "1", None, "Artist", "Artist", None,
     "Individual", "Nameless Artist", None, None, None, None, None, None),
    ("7", "No#", "A", "1/5/1940", None, "1", None, "Curator", "Organizer", "3",
     "Institution", "Some Org", None, None, None, None, None, None),
    # Exhibition-only row (no participant) with an end date before its begin date.
    ("8", "12", "C", "6/1/1960", "5/1/1960", "1", None, None, None, None,
     None, None, None, None, None, None, None, None),
]


def test_clean_toy(base):
    toy = pd.DataFrame(TOY_ROWS, columns=TOY_COLUMNS, dtype=object)
    t = base["clean"](toy, base["CFG"]["roles"])
    ex, con, part = t["exhibitions"], t["constituents"], t["participation"]
    assert list(ex["title"]) == ["A", "B", "C"]                      # chronological ids
    assert ex["duration_days"].isna().tolist() == [True, False, True]  # negative -> missing
    assert list(ex["has_curator"]) == [False, True, False]           # institutions don't count
    assert part.set_index("cid")["role"].to_dict() == {
        "moma:1": "curator", "moma:2": "artist", "name:Nameless Artist": "artist",
        "moma:3": "other"}
    con = con.set_index("cid")
    assert con.loc["moma:2", "nationality"] == "unknown"
    assert con.loc["moma:2", "gender"] == "male"
    assert con.loc["moma:1", "birth_year"] == 1900


def test_real_tables(base):
    tables, data = base["tables"], base["data"]
    ex = tables["exhibitions"]
    assert len(ex) == 1787 and ex["begin"].notna().all()
    assert (ex["exh_id"] == np.arange(len(ex))).all() and ex["begin"].is_monotonic_increasing
    assert not tables["participation"].duplicated(["exh_id", "cid", "role"]).any()
    assert data.num_artists == 11273 and data.num_curators == 316
    assert "Nationality unknown" not in set(data.artists["nationality"])


def test_acquisitions_linked(base):
    data = base["data"]
    acq = data.acquisitions
    assert acq["artist"].between(0, data.num_artists - 1).all()
    assert acq["year"].is_monotonic_increasing and acq["year"].min() >= 1929
    # Thousands of exhibited artists have collection works; spot-check one link by cid.
    assert acq["artist"].nunique() > 3000
    artist = acq["artist"].iloc[0]
    cid = data.artists.loc[artist, "cid"]
    raw = base["tables"]["acquisitions"]
    assert (raw["cid"] == cid).sum() >= (acq["artist"] == artist).sum() > 0


def test_shuffled_curators(base):
    data = base["data"]
    shuffled = data.with_shuffled_curators(seed=0)
    before = data.curator_edges.groupby("curator").size()
    after = shuffled.curator_edges.groupby("curator").size().reindex(before.index, fill_value=0)
    assert (before - after).abs().sum() <= 0.02 * len(data.curator_edges)
    assert len(data.curator_edges.merge(shuffled.curator_edges)) < 0.5 * len(data.curator_edges)
    decade = lambda d, e: (d.exhibitions.loc[e["exh_id"], "begin"].dt.year // 10).value_counts()
    assert decade(data, data.curator_edges).sub(
        decade(shuffled, shuffled.curator_edges), fill_value=0).abs().sum() <= 0.02 * len(
        data.curator_edges)
    assert shuffled.artist_edges is data.artist_edges
