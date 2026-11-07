# Six Degrees of MoMA

Temporal link prediction on MoMA's Exhibition History (1929–1989): given an exhibition's
recorded curator(s), its metadata and the exhibition network before a cutoff, rank artists by
how likely they are to participate. CS 224W project by Shirley Zhang, Alexis Tuchinda and
Sheryl Chen.

## Notebooks

Each notebook is self-contained. You can drag a single notebook into Colab and run it: it
installs missing packages, downloads the data if needed, and writes outputs next to itself.

| notebook | what it does | runtime |
|---|---|---|
| `01_data_eda.ipynb` | download, clean, curator coverage, sparsity, debut rates, collection coverage | ~1 min |
| `02_baselines.ipynb` | popularity, recency, collection, curator-collab, common neighbors, Adamic–Adar, Node2Vec; cold-start baselines | ~12 min CPU |
| `03_gnn.ipynb` | R-GCN / HeteroConv / HGT / GraphSAGE / GAT / feature-only MLP, leak check, design check, curator ablation, test-time curator removal | hours on CPU; GPU recommended |
| `04_analysis.ipynb` | report tables and figures: main results, ablation CIs, subgroups, cold start, limitations, case studies | ~1 min, needs 02 and 03 |

Run them in order: 01, 02, 03, 04. Notebook 04 trains nothing. It reads the files that 02 and 03
save in `results/`. On Colab, set `USE_DRIVE = True` in the second cell to keep data and results
in `MyDrive/moma-project` across sessions. Otherwise, everything is lost when the runtime resets.

`03_gnn` saves every run as it finishes and skips runs that already exist with the same config.
If Colab disconnects, run it again to continue. Teammates sharing the Drive folder can split
`EXPERIMENTS` between them.

The shared cells are identical in every notebook: install, config, cleaning, `MomaData`,
snapshots, metrics and evaluation. **If you change one of them, change it in all four.**
`tests/test_data.py::test_shared_cells_identical` fails if they drift apart.

## Running locally and testing

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab notebooks/
python -m pytest          # 37 tests, about 1 minute
```

The tests execute the notebooks' own code cells (see `tests/conftest.py`), so they test exactly
what runs in Colab. They cover:

- cleaning on a hand-made example, and the real table sizes;
- that the shared cells are identical in all four notebooks;
- no temporal leakage in any split, and that time-aware features use only the past;
- metrics on hand-computed examples, and the bootstrap;
- that each baseline scores what it claims to;
- that every model trains (forward and backward pass);
- saving and resuming experiment runs.

## Data

- **Exhibitions**: [MoMA Exhibition History](https://github.com/MuseumofModernArt/exhibitions),
  1,787 exhibitions with 11,273 artists and 316 curators.
- **Collection**: [MoMA Collection](https://github.com/MuseumofModernArt/collection)
  `Artworks.csv`, used only for acquisition dates. 4,423 of the exhibited artists have works in
  the collection. Artists are matched by MoMA ConstituentID, which both datasets share.

## Design decisions

- **Exhibition identity.** ExhibitionID is missing for about 430 rows, and the number "No#"
  covers about 50 different exhibitions. Exhibitions are therefore keyed on (number, title, begin
  date), which gives 1,787 exhibitions. Constituents without a ConstituentID are keyed by
  display name.
- **Roles.** `ExhibitionRole == "Curator"` held by an individual makes a curator node: 316
  curators, recorded on 1,204 exhibitions. Institutions listed as curators, installers, designers,
  judges, arrangers and preparers count as "other" and are not modeled. The mapping is
  `CFG["roles"]`.
- **Snapshots.** A snapshot contains all edges of exhibitions that began before `history_end`,
  plus the curator edges and metadata of its target exhibitions. Target artist edges are labels
  only. Candidates are the artists with at least one exhibition in the history. Artist and
  curator history features, and collection features, are recomputed from each snapshot's visible
  past.
- **Splits for cutoff T** (T ∈ 1950, 1960, 1970, 1980).
  - Validation targets are [T−5, T), with history before T−5.
  - Test targets are [T, T+10), with history before T.
  - Training uses rolling pseudo-cutoffs t < T−5, with targets in [t, t+5). Every training example
    therefore has the same form as the test task, and the model never sees edges from after its
    own targets.
- **Primary evaluation subset.** Test exhibitions with a recorded curator and at least one
  returning artist.
- **Cold start.** The same test exhibitions rank only the artists who debut at MoMA in the test
  decade. Knowing who debuts somewhere in the decade is an oracle assumption; predicting where is
  the task.
- **Curator ablations.**
  - `none` drops the curator node type.
  - `shuffled` permutes curator→exhibition assignments within each decade, which keeps every
    curator's and every exhibition's degree.
  - Exhibition features exclude curator counts, so `none` cannot recover curator information
    through them.
- **Seeds.** The R-GCN ablation uses 5 seeds and the other models 3. Comparisons use a paired
  bootstrap over test exhibitions, after averaging seeds per exhibition.
