#!/usr/bin/env python3
"""Question 1: decision tree baseline and bagging from scratch."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROLL_NUMBER = "M24MA2009"
SEED = int("".join(c for c in ROLL_NUMBER if c.isdigit()))
DEPTHS = (2, 4, 8, 16, None)
LEAF_SIZES = (1, 2, 4)
B_VALUES = (1, 5, 10, 25, 50, 100, 200)


def make_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train, valid, test = [], [], []
    for identity in np.unique(y):
        idx = np.flatnonzero(y == identity)
        perm = rng.permutation(idx)
        train.extend(perm[:6]); valid.extend(perm[6:8]); test.extend(perm[8:])
    return tuple(np.asarray(a, dtype=np.int64) for a in (train, valid, test))


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y == pred))


def tree_stats(model: DecisionTreeClassifier, X: np.ndarray) -> dict:
    leaves = model.apply(X)
    _, counts = np.unique(leaves, return_counts=True)
    return {"leaves": int(len(counts)), "mean_samples_per_leaf": float(counts.mean()),
            "fraction_singleton_leaves": float(np.mean(counts == 1))}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def plot_bagging(rows: list[dict], out: Path) -> None:
    b = [r["B"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, title in zip(axes, ("oob_accuracy", "pairwise_disagreement", "mean_vote_margin"),
                              ("OOB accuracy", "Pairwise disagreement", "Mean validation vote margin")):
        ax.plot(b, [r[key] for r in rows], marker="o")
        ax.set_xscale("log"); ax.set_xticks(b, labels=[str(v) for v in b]); ax.set_xlabel("Number of trees (B)")
        ax.set_ylabel(title); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def masked(X: np.ndarray) -> np.ndarray:
    images = X.reshape(-1, 64, 64).copy()
    images[:, 24:40, 24:40] = 0
    return images.reshape(len(images), -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)

    data = fetch_olivetti_faces(shuffle=False)
    X = data.images.reshape(len(data.images), -1).astype(np.float32)
    y = data.target.astype(np.int64)
    tr, va, te = make_split(y)
    np.savez_compressed(out / "split.npz", train=tr, validation=va, test=te)
    config = {"roll_number": ROLL_NUMBER, "seed": SEED, "dataset": "fetch_olivetti_faces",
              "features": "raw flattened 64x64 pixels", "train_per_identity": 6,
              "validation_per_identity": 2, "test_per_identity": 2,
              "depths": list(DEPTHS), "min_samples_leaf": list(LEAF_SIZES),
              "bagging_B": list(B_VALUES), "bootstrap_size": int(len(tr)),
              "tie_breaking": "grid order; smallest B", "mask_rows_cols": [24, 39]}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    grid = []
    fitted = {}
    for depth in DEPTHS:
        for leaf in LEAF_SIZES:
            model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
            model.fit(Xtr, ytr); fitted[(depth, leaf)] = model
            row = {"max_depth": depth if depth is not None else "None", "min_samples_leaf": leaf,
                   "train_accuracy": accuracy(ytr, model.predict(Xtr)),
                   "validation_accuracy": accuracy(yva, model.predict(Xva)),
                   **tree_stats(model, Xtr)}
            grid.append(row)
    # Copy the selected record so adding its held-out test metric does not
    # mutate the grid row that is written to tree_grid.csv.
    chosen_row = dict(max(grid, key=lambda r: r["validation_accuracy"]))
    chosen_depth = None if chosen_row["max_depth"] == "None" else int(chosen_row["max_depth"])
    chosen_leaf = int(chosen_row["min_samples_leaf"])
    tree = fitted[(chosen_depth, chosen_leaf)]
    chosen_row.update(test_accuracy=accuracy(yte, tree.predict(Xte)))
    write_csv(out / "tree_grid.csv", grid)
    (out / "selected_tree.json").write_text(json.dumps(chosen_row, indent=2) + "\n")

    # One seeded sequence of bootstrap samples; smaller B results are prefixes.
    rng = np.random.default_rng(SEED)
    n = len(tr); max_b = max(B_VALUES)
    boot_indices = [rng.integers(0, n, size=n) for _ in range(max_b)]
    models, oob_masks, distinct_counts = [], [], []
    fit_start = time.perf_counter()
    for sample in boot_indices:
        model = DecisionTreeClassifier(max_depth=chosen_depth, min_samples_leaf=chosen_leaf,
                                       random_state=SEED)
        model.fit(Xtr[sample], ytr[sample]); models.append(model)
        present = np.zeros(n, dtype=bool); present[np.unique(sample)] = True
        oob_masks.append(~present); distinct_counts.append(int(present.sum()))
    bagging_fit_seconds = time.perf_counter() - fit_start

    rows, predictions_by_b = [], {}
    classes = np.unique(y)
    oob_pred_by_b = {}
    for B in B_VALUES:
        ensemble = models[:B]
        p_tr = np.stack([m.predict(Xtr) for m in ensemble])
        p_va = np.stack([m.predict(Xva) for m in ensemble])
        p_te = np.stack([m.predict(Xte) for m in ensemble])
        pte_masked = np.stack([m.predict(masked(Xte)) for m in ensemble])
        def vote(pred):
            return np.stack([np.sum(pred == c, axis=0) for c in classes], axis=1)
        vtr, vva, vte, vtm = map(vote, (p_tr, p_va, p_te, pte_masked))
        # Deterministic ties: lowest class id (classes are sorted).
        labels = lambda v: classes[np.argmax(v, axis=1)]
        margins = lambda v: np.sort(v / B, axis=1)[:, -1] - np.sort(v / B, axis=1)[:, -2]
        # OOB label vote per original training item.
        sums = np.zeros((n, len(classes)), dtype=np.int32); voters = np.zeros(n, dtype=np.int32)
        for j, m in enumerate(ensemble):
            ids = np.flatnonzero(oob_masks[j])
            if len(ids):
                pred = m.predict(Xtr[ids]); sums[ids, pred] += 1; voters[ids] += 1
        eligible = voters > 0
        oob_pred = np.argmax(sums[eligible], axis=1)
        oob_accuracy = accuracy(ytr[eligible], oob_pred)
        oob_pred_by_b[B] = (eligible, sums, voters)
        disagreement = []
        for i, j in combinations(range(B), 2):
            disagreement.append(float(np.mean(p_tr[i] != p_tr[j])))
        row = {"B": B, "train_accuracy": accuracy(ytr, labels(vtr)),
               "validation_accuracy": accuracy(yva, labels(vva)), "test_accuracy": accuracy(yte, labels(vte)),
               "mean_distinct_bootstrap_examples": float(np.mean(distinct_counts[:B])),
               "oob_accuracy": oob_accuracy,
               "oob_eligible_examples": int(eligible.sum()),
               "pairwise_disagreement": float(np.mean(disagreement)) if disagreement else 0.0,
               "mean_vote_margin": float(np.mean(margins(vva)))}
        rows.append(row)
        predictions_by_b[B] = {"train_votes": vtr, "valid_votes": vva, "test_votes": vte,
                               "masked_test_votes": vtm, "test_pred": labels(vte),
                               "masked_pred": labels(vtm), "test_confidence": np.max(vte, axis=1) / B,
                               "masked_confidence": np.max(vtm, axis=1) / B}
    write_csv(out / "bagging_results.csv", rows)
    plot_bagging(rows, out / "bagging_diagnostics.png")
    # Expected unique observations in a size-n bootstrap sample: n[1-(1-1/n)^n].
    expected_unique = n * (1 - (1 - 1 / n) ** n)
    selected = max(rows, key=lambda r: r["validation_accuracy"])
    Bfinal = int(selected["B"])
    (out / "summary.json").write_text(json.dumps({
        "selected_tree": chosen_row, "selected_bagging_B": Bfinal,
        "bootstrap_size": n, "expected_distinct_per_bootstrap": expected_unique,
        "observed_distinct_at_B200": rows[-1]["mean_distinct_bootstrap_examples"],
        "bagging_fit_seconds_for_200_trees": bagging_fit_seconds,
        "oob_note": "OOB accuracy excludes training examples with zero OOB votes.",
    }, indent=2) + "\n")

    # Clean and masked evaluation with mean winning vote fraction as confidence.
    bag = predictions_by_b[Bfinal]
    base_te, base_mask = tree.predict(Xte), tree.predict(masked(Xte))
    base_proba = tree.predict_proba(Xte).max(axis=1)
    base_mask_proba = tree.predict_proba(masked(Xte)).max(axis=1)
    comparison = [{"model": "single_tree", "clean_accuracy": accuracy(yte, base_te),
                  "masked_accuracy": accuracy(yte, base_mask), "clean_mean_confidence": float(base_proba.mean()),
                  "masked_mean_confidence": float(base_mask_proba.mean())},
                 {"model": f"bagging_B{Bfinal}", "clean_accuracy": accuracy(yte, bag["test_pred"]),
                  "masked_accuracy": accuracy(yte, bag["masked_pred"]),
                  "clean_mean_confidence": float(bag["test_confidence"].mean()),
                  "masked_mean_confidence": float(bag["masked_confidence"].mean())}]
    write_csv(out / "clean_masked_comparison.csv", comparison)

    corrected = np.flatnonzero((base_te != yte) & (bag["test_pred"] == yte))[:2]
    both_wrong = np.flatnonzero((base_te != yte) & (bag["test_pred"] != yte))[:2]
    cases = []
    for kind, indices in (("bagging_corrects_tree", corrected), ("both_wrong", both_wrong)):
        for i in indices:
            votes = bag["test_votes"][i]
            cases.append({"case": kind, "global_image_index": int(te[i]), "true_label": int(yte[i]),
                          "tree_prediction": int(base_te[i]),
                          "bagging_vote_counts": {str(int(c)): int(v) for c, v in zip(classes, votes) if v},
                          "bagging_prediction": int(bag["test_pred"][i]),
                          "tree_masked_prediction": int(base_mask[i]),
                          "bagging_masked_vote_counts": {str(int(c)): int(v) for c, v in zip(classes, bag["masked_test_votes"][i]) if v}})
    (out / "illustrative_test_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(json.dumps({"results_dir": str(out.resolve()), "selected_tree": chosen_row,
                      "selected_bagging_B": Bfinal, "expected_distinct": expected_unique,
                      "comparison": comparison, "case_counts": {"corrected": len(corrected),
                      "both_wrong": len(both_wrong)}}, indent=2))


if __name__ == "__main__":
    main()
