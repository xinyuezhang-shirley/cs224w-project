"""Load the notebooks' own code, so the tests check exactly what runs in Colab.

Each fixture executes a notebook's code cells, in order, up to a stop marker (before any long
training loops). The real data is downloaded on first use, like in the notebooks.
"""
import os
from pathlib import Path

import nbformat
import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def code_cells(name):
    return [c.source for c in nbformat.read(NB_DIR / name, as_version=4).cells
            if c.cell_type == "code"]


def run_cells(name, stop, skip=()):
    """Execute code cells of `name` until one starting with `stop`; return the namespace."""
    os.chdir(NB_DIR)
    ns = {"display": lambda *a, **k: None, "__name__": "__notebook__"}
    for src in code_cells(name):
        if src.startswith(stop):
            return ns
        if not src.startswith(skip):
            exec(compile(src, name, "exec"), ns)
    raise AssertionError(f"stop marker {stop!r} not found in {name}")


@pytest.fixture(scope="session")
def base():
    """Notebook 02 up to the baseline runs: data, snapshots, metrics, baselines, Node2Vec."""
    return run_cells("02_baselines.ipynb", stop="RUN_NODE2VEC")


@pytest.fixture(scope="session")
def gnn():
    """Notebook 03 up to the experiments: features, graphs, models, training."""
    return run_cells("03_gnn.ipynb", stop="RUN_TUNING", skip=("RUN_LEAK_CHECK",))
