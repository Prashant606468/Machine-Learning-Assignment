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

The first run downloads Olivetti Faces through scikit-learn. The script writes
the identity-stratified split (`split.npz`), configuration, CSV/JSON logs, and
figures under `results/`. It trains on clean images only and evaluates the
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
`results/q3/`. Your selected tree has `max_depth=None`, which is already
unlimited; the script therefore uses depth 32 as a deep finite comparator and
records this limitation in `results/q3/config.json`.

Tree selection uses validation accuracy only. Ties are resolved by the
listed grid order (`max_depth` 2, 4, 8, 16, None; then `min_samples_leaf`
1, 2, 4). The bagging selection uses validation accuracy only, with the
smallest B winning ties. OOB predictions are formed only from trees for which
that training image was out of bag; images with no OOB votes are excluded.
Pairwise disagreement averages the fraction of the 240 training examples on
which each pair of trees disagree. Vote margin is averaged over validation
examples.

The assignment requires the student’s name, repository URL, and final commit
hash on the paper’s first page. Those were not supplied, so add them before
submission. Do not report generated results until the script has completed
successfully on your environment.
