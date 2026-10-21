#!/usr/bin/env python3
"""Question 3: empirical bias/variance from identity-wise bootstrap refits."""
from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

from q1 import SEED, accuracy, masked
from q2 import SAMME

REPETITIONS = 30
SHALLOW_DEPTH = 2
DEEP_FINITE_DEPTH = 32
DEPTH_BAG_SIZE = 100


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def one_hot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), len(classes)), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def bias_variance(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """probabilities has shape (repetitions, test examples, classes)."""
    mean_p = probabilities.mean(axis=0)
    bias2 = float(np.mean(np.sum((mean_p - targets) ** 2, axis=1)))
    variance = float(np.mean(np.sum((probabilities - mean_p[None, :, :]) ** 2, axis=2)))
    return bias2, variance


def leaf_occupancy(model: DecisionTreeClassifier, X_fit: np.ndarray) -> dict:
    leaf_ids = model.apply(X_fit)
    _, counts = np.unique(leaf_ids, return_counts=True)
    return {"leaves": int(len(counts)), "mean_samples_per_leaf": float(counts.mean()),
            "fraction_singleton_leaves": float(np.mean(counts == 1))}


def ensemble_disagreement(predictions: np.ndarray, n_classes: int) -> float:
    """Mean pairwise disagreement; predictions shape is (trees, examples)."""
    n_trees = predictions.shape[0]
    if n_trees < 2:
        return 0.0
    counts = np.stack([(predictions == c).sum(axis=0) for c in range(n_classes)], axis=0)
    per_example = (n_trees**2 - np.sum(counts**2, axis=0)) / (n_trees * (n_trees - 1))
    return float(np.mean(per_example))


def fit_tree(depth, leaf, X, y, random_state):
    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=random_state)
    model.fit(X, y)
    return model


def fit_bag(X, y, depth, leaf, bootstrap_indices):
    models = []
    occupancies = []
    for j, idx in enumerate(bootstrap_indices):
        model = fit_tree(depth, leaf, X[idx], y[idx], SEED)
        models.append(model)
        occupancies.append(leaf_occupancy(model, X[idx]))
    return models, occupancies


def bag_probabilities(models, X, classes):
    tree_predictions = np.stack([m.predict(X) for m in models])
    votes = np.stack([(tree_predictions == c).sum(axis=0) for c in classes], axis=1)
    return votes / len(models), tree_predictions


def plot_depth_diagnostics(rows: list[dict], path: Path) -> None:
    depths = [r["depth_label"] for r in rows if r["model"] == "single_tree"]
    lookup = {(r["model"], r["depth_label"]): r for r in rows}
    x = np.arange(len(depths)); width = .35
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, title in zip(axes,
                                ("bias_squared", "variance", "mean_test_accuracy"),
                                ("Bias squared", "Variance", "Mean test accuracy")):
        ax.bar(x - width/2, [lookup[("single_tree", d)][field] for d in depths], width, label="Single tree")
        ax.bar(x + width/2, [lookup[("bagging_100", d)][field] for d in depths], width, label="100-tree bagging")
        ax.set_xticks(x, depths); ax.set_xlabel("max_depth setting"); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="Question 1 results directory; expects q2/selected_boosting.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS,
                        help="Use 30 for the assignment; smaller values are useful only for a quick smoke run")
    args = parser.parse_args()
    q1dir = args.results; out = args.out or (q1dir / "q3"); out.mkdir(parents=True, exist_ok=True)
    split_path = q1dir / "split.npz"
    q2_selected_path = q1dir / "q2" / "selected_boosting.json"
    for path in (split_path, q1dir / "selected_tree.json", q1dir / "summary.json", q2_selected_path):
        if not path.exists():
            raise FileNotFoundError(f"Required selection/split file missing: {path}. Run q1.py and q2.py first.")

    split = np.load(split_path)
    train_ids, test_ids = split["train"], split["test"]
    tree_cfg = json.loads((q1dir / "selected_tree.json").read_text())
    q1_summary = json.loads((q1dir / "summary.json").read_text())
    boost_cfg = json.loads(q2_selected_path.read_text())
    selected_depth = tree_cfg["max_depth"]
    selected_depth = None if selected_depth == "None" else int(selected_depth)
    selected_leaf = int(tree_cfg["min_samples_leaf"])
    final_b = int(q1_summary["selected_bagging_B"])
    boost_depth = int(boost_cfg["weak_depth"]); boost_rounds = int(boost_cfg["rounds"])

    data = fetch_olivetti_faces(shuffle=False)
    X = data.images.reshape(len(data.images), -1).astype(np.float32)
    y = data.target.astype(np.int64); classes = np.unique(y); k = len(classes)
    Xtrain, ytrain = X[train_ids], y[train_ids]
    Xtest, ytest = X[test_ids], y[test_ids]
    Xtest_masked = masked(Xtest)
    target_vectors = one_hot(ytest, classes)
    identities = np.unique(ytrain)
    train_local_by_identity = {c: np.flatnonzero(ytrain == c) for c in identities}
    rng = np.random.default_rng(SEED)
    depth_settings = [(SHALLOW_DEPTH, str(SHALLOW_DEPTH)), (selected_depth, "None" if selected_depth is None else str(selected_depth)),
                      (DEEP_FINITE_DEPTH, str(DEEP_FINITE_DEPTH))]
    # Keep the settings distinct if the selected finite depth equals a probe depth.
    unique_depth_settings = []
    seen = set()
    for depth, label in depth_settings:
        key = (depth, label)
        if key not in seen:
            unique_depth_settings.append((depth, label)); seen.add(key)
    unlimited_selected = selected_depth is None

    # Store per-refit probability vectors for the three selected models and depth probes.
    selected_probs = {name: [] for name in ("single_tree", "bagging", "SAMME")}
    selected_mask_acc = {name: [] for name in selected_probs}
    selected_clean_acc = {name: [] for name in selected_probs}
    depth_probs = {(model_name, label): [] for model_name in ("single_tree", "bagging_100")
                   for _, label in unique_depth_settings}
    depth_acc = {(model_name, label): [] for model_name in ("single_tree", "bagging_100")
                 for _, label in unique_depth_settings}
    depth_leaf_rows = {(model_name, label): [] for model_name in ("single_tree", "bagging_100")
                       for _, label in unique_depth_settings}
    depth_disagreements = {label: [] for _, label in unique_depth_settings}
    q3a_rows = []
    max_b_for_probes = max(final_b, DEPTH_BAG_SIZE)
    run_started = time.perf_counter()

    for rep in range(args.repetitions):
        # Six draws with replacement within each identity's six original train faces.
        resampled_local = []
        for identity in identities:
            original_local = train_local_by_identity[identity]
            resampled_local.extend(rng.choice(original_local, size=len(original_local), replace=True).tolist())
        resampled_local = np.asarray(resampled_local, dtype=np.int64)
        Xr, yr = Xtrain[resampled_local], ytrain[resampled_local]

        # Selected single tree, frozen Q1 parameters.
        tree = fit_tree(selected_depth, selected_leaf, Xr, yr, SEED)
        p_tree = tree.predict_proba(Xtest)
        selected_probs["single_tree"].append(p_tree)
        selected_clean_acc["single_tree"].append(accuracy(ytest, tree.predict(Xtest)))
        selected_mask_acc["single_tree"].append(accuracy(ytest, tree.predict(Xtest_masked)))

        # Generate one shared set of bootstrap index vectors for comparable depths.
        bootstrap_indices = [rng.integers(0, len(yr), size=len(yr)) for _ in range(max_b_for_probes)]
        selected_bag, selected_bag_occupancy = fit_bag(Xr, yr, selected_depth, selected_leaf,
                                                        bootstrap_indices[:final_b])
        p_bag, pred_bag = bag_probabilities(selected_bag, Xtest, classes)
        selected_probs["bagging"].append(p_bag)
        selected_clean_acc["bagging"].append(accuracy(ytest, classes[np.argmax(p_bag, axis=1)]))
        p_bag_mask, pred_bag_mask = bag_probabilities(selected_bag, Xtest_masked, classes)
        selected_mask_acc["bagging"].append(accuracy(ytest, classes[np.argmax(p_bag_mask, axis=1)]))

        # Selected SAMME model, same update rule as Q2.
        boost = SAMME(boost_depth, k, SEED).fit(Xr, yr, boost_rounds)
        p_boost = boost.predict_proba(Xtest)
        selected_probs["SAMME"].append(p_boost)
        selected_clean_acc["SAMME"].append(accuracy(ytest, boost.predict(Xtest)))
        selected_mask_acc["SAMME"].append(accuracy(ytest, boost.predict(Xtest_masked)))

        q3a_rows.extend([
            {"repetition": rep + 1, "model": "single_tree", "test_accuracy": selected_clean_acc["single_tree"][-1],
             "masked_accuracy": selected_mask_acc["single_tree"][-1]},
            {"repetition": rep + 1, "model": "bagging", "test_accuracy": selected_clean_acc["bagging"][-1],
             "masked_accuracy": selected_mask_acc["bagging"][-1]},
            {"repetition": rep + 1, "model": "SAMME", "test_accuracy": selected_clean_acc["SAMME"][-1],
             "masked_accuracy": selected_mask_acc["SAMME"][-1]},
        ])

        # Q3(b): single trees and 100-tree bagging at shallow, selected, and deep finite settings.
        for depth, label in unique_depth_settings:
            if depth == selected_depth:
                probe_tree = tree
                probe_bag_models = selected_bag[:DEPTH_BAG_SIZE]
                probe_bag_leafs = selected_bag_occupancy[:DEPTH_BAG_SIZE]
                if len(probe_bag_models) < DEPTH_BAG_SIZE:
                    extra_models, extra_leafs = fit_bag(Xr, yr, depth, selected_leaf,
                                                        bootstrap_indices[len(probe_bag_models):DEPTH_BAG_SIZE])
                    probe_bag_models += extra_models; probe_bag_leafs += extra_leafs
            else:
                probe_tree = fit_tree(depth, selected_leaf, Xr, yr, SEED)
                probe_bag_models, probe_bag_leafs = fit_bag(
                    Xr, yr, depth, selected_leaf, bootstrap_indices[:DEPTH_BAG_SIZE])

            ps = probe_tree.predict_proba(Xtest)
            depth_probs[("single_tree", label)].append(ps)
            depth_acc[("single_tree", label)].append(accuracy(ytest, probe_tree.predict(Xtest)))
            depth_leaf_rows[("single_tree", label)].append(leaf_occupancy(probe_tree, Xr))

            pb, pred_depth_bag = bag_probabilities(probe_bag_models, Xtest, classes)
            depth_probs[("bagging_100", label)].append(pb)
            depth_acc[("bagging_100", label)].append(accuracy(ytest, classes[np.argmax(pb, axis=1)]))
            depth_leaf_rows[("bagging_100", label)].extend(probe_bag_leafs)
            depth_disagreements[label].append(ensemble_disagreement(pred_depth_bag, k))

        print(f"Completed repetition {rep + 1}/{args.repetitions}", flush=True)

    # Q3(a) and (d): measured bias/variance and accuracy distributions.
    q3a_summary = []
    final_table = []
    for name, probs in selected_probs.items():
        p = np.stack(probs)
        bias2, variance = bias_variance(p, target_vectors)
        mean_acc = float(np.mean(selected_clean_acc[name]))
        std_acc = float(np.std(selected_clean_acc[name], ddof=1)) if args.repetitions > 1 else 0.0
        q3a_summary.append({"model": name, "bias_squared": bias2, "variance": variance,
                            "mean_test_accuracy": mean_acc, "std_test_accuracy": std_acc})
        final_table.append({"model": name, "clean_test_accuracy_mean": mean_acc,
                            "masked_test_accuracy_mean": float(np.mean(selected_mask_acc[name])),
                            "bias_squared": bias2, "variance": variance})
    write_csv(out / "q3a_bias_variance_accuracy.csv", q3a_summary)
    write_csv(out / "q3a_per_repetition_accuracy.csv", q3a_rows)
    write_csv(out / "q3d_final_model_table.csv", final_table)

    # Q3(b,c): depth probes, occupancy, and ensemble disagreement.
    depth_rows = []
    for depth, label in unique_depth_settings:
        for model_name in ("single_tree", "bagging_100"):
            p = np.stack(depth_probs[(model_name, label)])
            bias2, variance = bias_variance(p, target_vectors)
            accs = np.asarray(depth_acc[(model_name, label)])
            occ = depth_leaf_rows[(model_name, label)]
            depth_rows.append({"model": model_name, "depth_label": label,
                               "bias_squared": bias2, "variance": variance,
                               "mean_test_accuracy": float(accs.mean()),
                               "std_test_accuracy": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
                               "mean_leaves": float(np.mean([v["leaves"] for v in occ])),
                               "mean_samples_per_leaf": float(np.mean([v["mean_samples_per_leaf"] for v in occ])),
                               "fraction_singleton_leaves": float(np.mean([v["fraction_singleton_leaves"] for v in occ])),
                               "mean_pairwise_test_disagreement": (float(np.mean(depth_disagreements[label]))
                                                                   if model_name == "bagging_100" else "")})
    # Add paired bagging gains/reductions to make Q3(c) directly auditable.
    depth_lookup = {(r["model"], r["depth_label"]): r for r in depth_rows}
    for row in depth_rows:
        if row["model"] == "bagging_100":
            single = depth_lookup[("single_tree", row["depth_label"])]
            row["accuracy_gain_vs_single"] = row["mean_test_accuracy"] - single["mean_test_accuracy"]
            row["variance_removed_vs_single"] = single["variance"] - row["variance"]
        else:
            row["accuracy_gain_vs_single"] = ""
            row["variance_removed_vs_single"] = ""
    write_csv(out / "q3bc_depth_diagnostics.csv", depth_rows)
    plot_depth_diagnostics(depth_rows, out / "q3b_depth_bias_variance_accuracy.png")

    probability_arrays = {f"selected_{name}": np.stack(values) for name, values in selected_probs.items()}
    for (model_name, label), values in depth_probs.items():
        probability_arrays[f"{model_name}_depth_{label}"] = np.stack(values)
    np.savez_compressed(out / "probability_vectors.npz", **probability_arrays)

    configuration = {
        "seed": SEED, "repetitions": args.repetitions, "resampling": "within identity, six draws with replacement from original six train faces",
        "fixed_test_split": str(split_path), "selected_tree": {"max_depth": selected_depth, "min_samples_leaf": selected_leaf},
        "final_bagging_B": final_b, "selected_boosting": {"weak_depth": boost_depth, "rounds": boost_rounds},
        "depth_probes": [{"max_depth": depth, "label": label} for depth, label in unique_depth_settings],
        "deep_probe_note": ("Selected Q1 max_depth=None is already unlimited; no finite setting can be deeper. "
                            "The depth-32 probe is the deep finite cap comparator, not deeper than the selected unlimited setting."
                            if unlimited_selected else "Depth 32 is the requested deeper cap relative to the selected finite setting."),
        "accuracy_std": "sample standard deviation (ddof=1)",
        "masked_test_used_for": "Q3(d) robustness comparison only; bias and variance use the fixed clean test set",
    }
    (out / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    run_seconds = time.perf_counter() - run_started
    summary = {"repetitions": args.repetitions, "runtime_seconds": run_seconds,
               "q3a": q3a_summary, "q3b_c": depth_rows, "q3d": final_table,
               "deep_probe_note": configuration["deep_probe_note"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"results_dir": str(out.resolve()), "runtime_seconds": run_seconds,
                      "q3a": q3a_summary, "q3d": final_table,
                      "deep_probe_note": configuration["deep_probe_note"]}, indent=2))


if __name__ == "__main__":
    main()
