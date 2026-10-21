#!/usr/bin/env python3
"""Question 2: multiclass SAMME boosting and comparison with bagging."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

from q1 import SEED, accuracy, masked

DEPTHS = (1, 2, 3, 4)
ROUND_COUNTS = (10, 25, 50, 100)
COMPARE_COUNTS = (25, 50, 100)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class SAMME:
    """SAMME with explicit rejection of learners at or below random baseline."""
    def __init__(self, max_depth: int, n_classes: int, random_state: int):
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.random_state = random_state
        self.estimators: list[DecisionTreeClassifier] = []
        self.alphas: list[float] = []
        self.history: list[dict] = []
        self.stop_reason: str | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, rounds: int) -> "SAMME":
        self.classes_ = np.unique(y)
        n = len(y); k = len(self.classes_)
        weights = np.full(n, 1 / n, dtype=np.float64)
        aggregate = np.zeros((n, k), dtype=np.float64)
        for t in range(rounds):
            learner = DecisionTreeClassifier(max_depth=self.max_depth,
                                             random_state=self.random_state + t)
            fit_start = time.perf_counter()
            learner.fit(X, y, sample_weight=weights)
            fit_seconds = time.perf_counter() - fit_start
            pred = learner.predict(X)
            err = float(weights[pred != y].sum())
            random_error = 1 - 1 / k
            if err >= random_error - 1e-14:
                self.stop_reason = (f"round {t + 1}: weighted error {err:.12g} is not below "
                                    f"multiclass random error {random_error:.12g}")
                break
            if err <= 0:
                # SAMME's coefficient tends to infinity for a perfect learner;
                # use a finite numerical approximation and terminate after it.
                alpha = math.log((1 - 1e-15) / 1e-15) + math.log(k - 1)
            else:
                alpha = math.log((1 - err) / err) + math.log(k - 1)
            self.estimators.append(learner); self.alphas.append(float(alpha))
            aggregate[np.arange(n), pred] += alpha
            weights *= np.exp(alpha * (pred != y))
            weights /= weights.sum()
            weighted_ensemble_error = float(weights[np.argmax(aggregate, axis=1) != y].sum())
            self.history.append({
                "round": t + 1, "weighted_learner_error": err, "alpha": float(alpha),
                "max_sample_weight": float(weights.max()),
                "top10_weight_mass": float(np.sort(weights)[-min(10, n):].sum()),
                "ensemble_weighted_training_error": weighted_ensemble_error,
                "fit_seconds": float(fit_seconds),
                "weights_after_update": weights.copy(), "predictions": pred.copy(),
            })
            if err <= 0:
                self.stop_reason = f"round {t + 1}: perfect weak learner; stopped"
                break
        if not self.estimators:
            raise RuntimeError(f"SAMME produced no usable learners (depth={self.max_depth}; {self.stop_reason})")
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for alpha, learner in zip(self.alphas, self.estimators):
            scores[np.arange(len(X)), learner.predict(X)] += alpha
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.scores(X) / sum(self.alphas)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.scores(X), axis=1)]


def toy_round() -> dict:
    """A fully logged 3-class, 9-sample SAMME round with a specified weak rule."""
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2])
    weights = np.full(len(y), 1 / len(y), dtype=float)
    err = float(weights[pred != y].sum())
    alpha = math.log((1 - err) / err) + math.log(2)
    weights *= np.exp(alpha * (pred != y)); weights /= weights.sum()
    return {"description": "9 labeled samples, 3 classes; specified weak learner makes one error",
            "true_labels": y.tolist(), "initial_weights": [1 / 9] * 9,
            "weak_learner_predictions": pred.tolist(), "weighted_error": err,
            "alpha": alpha, "updated_normalized_weights": weights.tolist(),
            "update_rule": "w_i <- w_i exp(alpha * I[h(x_i) != y_i]), then normalize"}


def plot_history(model: SAMME, path: Path) -> None:
    rounds = [r["round"] for r in model.history]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    series = (("weighted_learner_error", "Weighted weak learner error"),
              ("alpha", "Learner weight α"),
              ("max_sample_weight", "Maximum sample weight"),
              ("top10_weight_mass", "Weight on 10 heaviest images"))
    for ax, (key, title) in zip(axes.flat, series):
        ax.plot(rounds, [r[key] for r in model.history]); ax.set_title(title)
        ax.set_xlabel("Round"); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="Question 1 results directory containing split.npz and model selections")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    q1dir = args.results; out = args.out or (q1dir / "q2"); out.mkdir(parents=True, exist_ok=True)
    split_file = q1dir / "split.npz"
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} missing; run q1.py first to create the shared split")
    split = np.load(split_file)
    tr, va, te = split["train"], split["validation"], split["test"]
    tree_cfg = json.loads((q1dir / "selected_tree.json").read_text())
    q1_summary = json.loads((q1dir / "summary.json").read_text())
    tree_depth = tree_cfg["max_depth"]
    tree_depth = None if tree_depth == "None" else int(tree_depth)
    tree_leaf = int(tree_cfg["min_samples_leaf"])
    final_b = int(q1_summary["selected_bagging_B"])

    data = fetch_olivetti_faces(shuffle=False)
    X = data.images.reshape(len(data.images), -1).astype(np.float32)
    y = data.target.astype(np.int64); classes = np.unique(y); k = len(classes)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    (out / "samme_toy_round.json").write_text(json.dumps(toy_round(), indent=2) + "\n")

    # Fit each depth once to the maximum requested rounds; snapshots supply the grid.
    fitted: dict[int, SAMME] = {}; grid = []
    for depth in DEPTHS:
        model = SAMME(depth, k, SEED).fit(Xtr, ytr, max(ROUND_COUNTS))
        fitted[depth] = model
        if not model.estimators:
            continue
        for rounds in ROUND_COUNTS:
            used = min(rounds, len(model.estimators))
            if not used:
                continue
            snap = SAMME(depth, k, SEED)
            snap.classes_ = classes; snap.estimators = model.estimators[:used]; snap.alphas = model.alphas[:used]
            grid.append({"weak_depth": depth, "requested_rounds": rounds, "actual_rounds": used,
                         "validation_accuracy": accuracy(yva, snap.predict(Xva)),
                         "train_accuracy": accuracy(ytr, snap.predict(Xtr)),
                         "test_accuracy": accuracy(yte, snap.predict(Xte)),
                         "stop_reason": model.stop_reason or ""})
    if not grid:
        raise RuntimeError("No SAMME model produced a usable weak learner")
    write_csv(out / "boosting_grid.csv", grid)
    best = max(grid, key=lambda r: r["validation_accuracy"])
    depth_sel = int(best["weak_depth"]); rounds_sel = int(best["actual_rounds"])
    full = fitted[depth_sel]
    chosen = SAMME(depth_sel, k, SEED)
    chosen.classes_ = classes; chosen.estimators = full.estimators[:rounds_sel]
    chosen.alphas = full.alphas[:rounds_sel]; chosen.history = full.history[:rounds_sel]
    chosen.stop_reason = full.stop_reason
    plot_history(chosen, out / "selected_boosting_diagnostics.png")
    (out / "selected_boosting.json").write_text(json.dumps({
        "weak_depth": depth_sel, "rounds": rounds_sel, "validation_accuracy": best["validation_accuracy"],
        "train_accuracy": accuracy(ytr, chosen.predict(Xtr)), "test_accuracy": accuracy(yte, chosen.predict(Xte)),
        "stop_reason": chosen.stop_reason, "alphas": chosen.alphas,
        "history": [{k: v for k, v in row.items() if k not in ("weights_after_update", "predictions")}
                    for row in chosen.history]}, indent=2) + "\n")

    # Locate the training image that attained the largest normalized weight.
    max_round, max_row = max(enumerate(chosen.history), key=lambda pair: pair[1]["max_sample_weight"])
    weight_rows = []
    for h in chosen.history:
        w = h["weights_after_update"]
        idx = int(np.argmax(w))
        weight_rows.append((float(w[idx]), idx, h["round"]))
    peak_weight, local_idx, peak_round = max(weight_rows)
    global_idx = int(tr[local_idx])
    face_history = []
    for h in chosen.history:
        pred = int(h["predictions"][local_idx]); weights = h["weights_after_update"]
        face_history.append({"round": h["round"], "prediction": pred, "true_label": int(ytr[local_idx]),
                             "misclassified": pred != int(ytr[local_idx]),
                             "weight_after_update": float(weights[local_idx]), "alpha": h["alpha"]})
    (out / "high_weight_training_face.json").write_text(json.dumps({
        "global_image_index": global_idx, "identity": int(ytr[local_idx]),
        "peak_weight": peak_weight, "peak_round": peak_round,
        "prediction_history": face_history}, indent=2) + "\n")

    # Timed bagging/boosting comparison at 25, 50, and 100 learners.
    bag_rng = np.random.default_rng(SEED)
    max_compare = max(COMPARE_COUNTS)
    boot = [bag_rng.integers(0, len(tr), size=len(tr)) for _ in range(max_compare)]
    bag_models: list[DecisionTreeClassifier] = []
    bag_fit_seconds: list[float] = []
    for sample in boot:
        learner = DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=tree_leaf, random_state=SEED)
        fit_start = time.perf_counter()
        learner.fit(Xtr[sample], ytr[sample])
        bag_fit_seconds.append(time.perf_counter() - fit_start)
        bag_models.append(learner)
    comparison = []
    boost_by_count = {}
    for count in COMPARE_COUNTS:
        # Reuse the already fitted 100-round selected-depth model and take prefixes.
        used = min(count, len(full.estimators))
        boost = SAMME(depth_sel, k, SEED); boost.classes_ = classes
        boost.estimators = full.estimators[:used]; boost.alphas = full.alphas[:used]
        boost_by_count[count] = boost
        for model_name, pred_fn, seconds in (
            ("bagging", lambda xx, count=count: np.apply_along_axis(
                lambda col: np.bincount(col.astype(int), minlength=k).argmax(), 0,
                np.stack([m.predict(xx) for m in bag_models[:count]])), sum(bag_fit_seconds[:count])),
            ("SAMME", boost.predict, 0.0)):
            if model_name == "SAMME":
                seconds = sum(h["fit_seconds"] for h in full.history[:used])
            p_train, p_valid, p_test = pred_fn(Xtr), pred_fn(Xva), pred_fn(Xte)
            comparison.append({"model": model_name, "learners": count, "actual_learners": count if model_name == "bagging" else used,
                               "train_accuracy": accuracy(ytr, p_train), "validation_accuracy": accuracy(yva, p_valid),
                               "clean_test_accuracy": accuracy(yte, p_test), "training_seconds": seconds})
    write_csv(out / "bagging_boosting_comparison.csv", comparison)

    # Frozen Q1 tree and final Q1 bagger, alongside the selected boosting model.
    tree = DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=tree_leaf, random_state=SEED).fit(Xtr, ytr)
    # Recreate the Q1 bootstrap sequence and fit its chosen prefix.
    bag_rng = np.random.default_rng(SEED)
    frozen_bag = []
    for _ in range(final_b):
        sample = bag_rng.integers(0, len(tr), size=len(tr))
        m = DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=tree_leaf, random_state=SEED)
        m.fit(Xtr[sample], ytr[sample]); frozen_bag.append(m)

    Xmask = masked(Xte)
    bag_pred = lambda xx: np.stack([m.predict(xx) for m in frozen_bag])
    boost_clean_scores, boost_mask_scores = chosen.scores(Xte), chosen.scores(Xmask)
    tree_clean, tree_mask = tree.predict(Xte), tree.predict(Xmask)
    bag_clean_votes, bag_mask_votes = bag_pred(Xte), bag_pred(Xmask)
    bag_clean_labels = np.array([np.bincount(col, minlength=k).argmax() for col in bag_clean_votes.T])
    bag_mask_labels = np.array([np.bincount(col, minlength=k).argmax() for col in bag_mask_votes.T])
    boost_clean, boost_mask = chosen.predict(Xte), chosen.predict(Xmask)
    bag_clean_vote_fractions = np.stack([(bag_clean_votes == c).sum(axis=0) for c in classes], axis=0) / final_b
    bag_mask_vote_fractions = np.stack([(bag_mask_votes == c).sum(axis=0) for c in classes], axis=0) / final_b
    evaluations = [
        {"model": "single_tree", "clean_accuracy": accuracy(yte, tree_clean),
         "masked_accuracy": accuracy(yte, tree_mask),
         "clean_mean_winning_probability": float(tree.predict_proba(Xte).max(axis=1).mean()),
         "masked_mean_winning_probability": float(tree.predict_proba(Xmask).max(axis=1).mean())},
        {"model": f"bagging_B{final_b}", "clean_accuracy": accuracy(yte, bag_clean_labels),
         "masked_accuracy": accuracy(yte, bag_mask_labels),
         "clean_mean_winning_probability": float(bag_clean_vote_fractions.max(axis=0).mean()),
         "masked_mean_winning_probability": float(bag_mask_vote_fractions.max(axis=0).mean())},
        {"model": f"SAMME_depth{depth_sel}_rounds{rounds_sel}", "clean_accuracy": accuracy(yte, boost_clean),
         "masked_accuracy": accuracy(yte, boost_mask),
         "clean_mean_winning_probability": float((boost_clean_scores / sum(chosen.alphas)).max(axis=1).mean()),
         "masked_mean_winning_probability": float((boost_mask_scores / sum(chosen.alphas)).max(axis=1).mean())},
    ]
    write_csv(out / "clean_masked_model_comparison.csv", evaluations)

    different = np.flatnonzero(bag_mask_labels != boost_mask)
    if len(different):
        # Prefer a case where exactly one of the two predictions is correct.
        informative = [i for i in different if (bag_mask_labels[i] == yte[i]) != (boost_mask[i] == yte[i])]
        case_i = int(informative[0] if informative else different[0])
        bvotes = np.bincount(bag_mask_votes[:, case_i], minlength=k)
        tree_preds = np.array([m.predict(Xmask[case_i:case_i+1])[0] for m in frozen_bag])
        case = {"global_image_index": int(te[case_i]), "true_label": int(yte[case_i]),
                "masked_tree_prediction": int(tree_mask[case_i]),
                "masked_bagging_prediction": int(bag_mask_labels[case_i]),
                "masked_bagging_vote_counts": {str(c): int(bvotes[c]) for c in range(k) if bvotes[c]},
                "masked_boosting_prediction": int(boost_mask[case_i]),
                "masked_boosting_normalized_scores": chosen.predict_proba(Xmask[case_i:case_i+1])[0].tolist(),
                "bagging_tree_disagreement_fraction": float(np.mean([
                    tree_preds[i] != tree_preds[j] for i, j in combinations(range(len(tree_preds)), 2)])),
                "bagging_tree_predictions": tree_preds.tolist(),
                "boosting_weighted_learner_predictions": [int(m.predict(Xmask[case_i:case_i+1])[0]) for m in chosen.estimators],
                "boosting_alphas": chosen.alphas,
                "boosting_high_weight_training_examples": [
                    {"global_image_index": int(tr[i]), "label": int(ytr[i]),
                     "weight": float(chosen.history[-1]["weights_after_update"][i])}
                    for i in np.argsort(chosen.history[-1]["weights_after_update"])[-10:][::-1]]}
    else:
        case = {"note": "No test face produced different masked bagging and boosting predictions."}
    (out / "different_mask_response_case.json").write_text(json.dumps(case, indent=2) + "\n")

    config = {"seed": SEED, "depths": list(DEPTHS), "round_counts": list(ROUND_COUNTS),
              "comparison_counts": list(COMPARE_COUNTS), "bagging_B_for_Q1_d": final_b,
              "selected_tree": {"max_depth": tree_depth, "min_samples_leaf": tree_leaf},
              "selected_boosting": {"depth": depth_sel, "rounds": rounds_sel}}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"results_dir": str(out.resolve()), "selected_boosting": config["selected_boosting"],
                      "validation_accuracy": best["validation_accuracy"], "grid_rows": len(grid),
                      "comparison": comparison, "clean_masked": evaluations,
                      "toy_example": toy_round(), "different_mask_case": case}, indent=2))


if __name__ == "__main__":
    main()
