# Assignment 2 — Question 1

Reproducible implementation of the decision-tree baseline and bootstrap
aggregation experiment on Olivetti Faces. All features are raw, flattened
pixels. The split seed is `242009`, obtained from `M24MA2009`.

## Run

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python q1.py --out results
```

The first run downloads Olivetti Faces through scikit-learn.It trains on clean images only and evaluates the
final frozen models on clean and center-masked test data.

Question 2 reuses that exact split and the selected Question 1 settings:

```sh
python q2.py --results results
```

Question 2 outputs are written under `results/q2/`. In particular,
`boosting_grid.csv` contains validation-based model selection,
`selected_boosting_diagnostics.png` contains the four requested round-wise
plots, and the comparison and per-example explanations are saved as CSV/JSON.

Question 3 reuses the saved split and selected settings from Questions 1 and 2:

```sh
python q3.py --results results
```

The full assignment run uses 30 resamples and writes its tables and figure to
`results/q3/`.Selected tree has `max_depth=None`, which is already
unlimited; the script therefore uses depth 32 as a deep finite comparator and
records this limitation in `results/q3/config.json`.

Tree selection and Bagging selection uses validation accuracy only. Ties are resolved by the
listed grid order (`max_depth` 2, 4, 8, 16, None; then `min_samples_leaf`
1, 2, 4).