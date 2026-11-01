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
| `01_data_eda.ipynb` | download, clean, curator coverage, sparsity, debut rates | < 1 min |
| `02_baselines.ipynb` | popularity, recency, curator-collab, common neighbors, Adamic–Adar, Node2Vec | ~5 min CPU |
| `03_gnn.ipynb` | R-GCN / HeteroConv / HGT / GraphSAGE / GAT, leak check, curator ablation | GPU recommended |

Run 02 before 03 if you want baselines in 03's comparison tables. On Colab, set
`USE_DRIVE = True` in the second cell to keep data and results in
`MyDrive/moma-project` across sessions. Otherwise, everything is lost when the runtime resets.

The shared cells are identical in every notebook: config, cleaning, `MomaData`, snapshots,
metrics and evaluation. **If you change one of them, change it in all three.**

Running locally:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab notebooks/
```

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
  curator history features are recomputed from each snapshot's visible history.
- **Splits for cutoff T** (T ∈ 1950, 1960, 1970, 1980).
  - Validation targets are [T−5, T), with history before T−5.
  - Test targets are [T, T+10), with history before T.
  - Training uses rolling pseudo-cutoffs t < T−5, with targets in [t, t+5). Every training example
    therefore has the same form as the test task, and the model never sees edges from after its
    own targets.
- **Primary evaluation subset.** Test exhibitions with a recorded curator and at least one
  returning artist. Exhibitions whose artists are all MoMA debuts are kept for the cold-start
  analysis.
- **Curator ablations.**
  - `none` drops the curator node type.
  - `shuffled` permutes curator→exhibition assignments within each decade, which keeps every
    curator's and every exhibition's degree.
  - Exhibition features exclude curator counts, so `none` cannot recover curator information
    through them.
