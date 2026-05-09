"""
=============================================================
Rare Disease Detection Pipeline
=============================================================
A complete machine learning pipeline for detecting rare diseases
in a REAL WORLD healthcare dataset with 2:1 class imbalance.

Key design decisions:
  - Stratified train/test split to preserve class ratios
  - StandardScaler fitted ONLY on training data (prevents leakage)
  - SMOTE applied ONLY to training data (prevents leakage)
  - Reproducible via fixed RANDOM_STATE seeds throughout
  - Modular functions for easy maintenance and testing

This is the ITERATIVE PROMPT VERSION - refined with guardrails.
=============================================================
"""

import os
import json
import warnings
from typing import Any, Dict, List, Tuple, cast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

ModelType = LogisticRegression | RandomForestClassifier

# ─────────────────────────────────────────────
#  Global configuration
# ─────────────────────────────────────────────
RANDOM_STATE  = 42          # Fixed seed for full reproducibility
TEST_SIZE     = 0.20        # 20 % held-out test set
N_SAMPLES     = 10_000      # Total synthetic samples
IMBALANCE_RATIO = 0.01      # ~1 % positive (disease) class

PLOTS_DIR   = "plots"
METRICS_DIR = "metrics"

# ─────────────────────────────────────────────
#  1. Load real healthcare DATA
# ─────────────────────────────────────────────

def load_real_healthcare_data() -> pd.DataFrame:
    """
    Load real patient data from CSV: 5,000 patient records.
    Features: 10 clinical/lab measurements.
    Target: disease_status (1=disease, 0=healthy).
    Imbalance: 99% healthy, 1% disease (realistic rare disease scenario).

    Returns
    -------
    pd.DataFrame with feature columns + disease label.
    """
    csv_path = Path(__file__).parent.parent / "data" / "rare_disease_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\n{'='*55}")
    print(f" Loaded Real Healthcare Dataset")
    print(f"{'='*55}")
    print(f"  Shape: {df.shape[0]:,} samples x {df.shape[1]} features")
    print(f"  Features: {list(df.columns[:-1])}")
    print(f"  Target column: {df.columns[-1]}")

    # Class distribution report
    label_col = df.columns[-1]
    n_healthy = (df[label_col] == 0).sum()
    n_disease = (df[label_col] == 1).sum()
    print(f"\n  Healthy (0): {n_healthy:,} ({n_healthy/len(df)*100:.1f}%)")
    print(f"  Disease (1): {n_disease:,} ({n_disease/len(df)*100:.1f}%)")
    print(f"  Imbalance:  1 : {n_healthy // max(n_disease, 1)}")
    print(f"{'='*55}\n")

    return df


# ─────────────────────────────────────────────
#  2. Train / test split  (STRATIFIED)
# ─────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train and test sets with stratification so that
    the rare positive class is proportionally represented in both.

    No SMOTE or scaling is applied here — those steps happen AFTER
    the split and are fitted only on training data.

    Returns
    -------
    X_train, X_test, y_train, y_test (all as NumPy arrays)
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        stratify     = np.asarray(y),           # ← preserves the 99/1 ratio in both sets
        random_state = random_state,
    )

    print(f"  Train size : {len(y_train):,}  (disease: {y_train.sum():,})")
    print(f"  Test  size : {len(y_test):,}   (disease: {y_test.sum():,})\n")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
#  3. SMOTE oversampling  (TRAIN ONLY)
# ─────────────────────────────────────────────

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the TRAINING set only.

    Why only training?  Applying SMOTE to the test set would create
    synthetic samples that "leak" into evaluation — giving artificially
    optimistic metrics that don't reflect real-world performance.

    SMOTE generates synthetic minority-class samples by interpolating
    between existing minority samples and their k-nearest neighbours.

    Returns
    -------
    X_resampled, y_resampled : balanced training arrays.
    """
    smote = SMOTE(random_state=random_state)
    resampled = smote.fit_resample(X_train, y_train)
    X_resampled = np.asarray(resampled[0])
    y_resampled = np.asarray(resampled[1])

    print(f"  After SMOTE → Disease: {y_resampled.sum():,}  |  "
          f"Healthy: {(y_resampled==0).sum():,}  |  "
          f"Total: {len(y_resampled):,}\n")
    return X_resampled, y_resampled


# ─────────────────────────────────────────────
#  4. Feature scaling  (fit on TRAIN, transform both)
# ─────────────────────────────────────────────

def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data ONLY, then transform both sets.

    Fitting on test data (or the full dataset) would be data leakage —
    the model would have seen test-set statistics during training.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_scaled  = scaler.transform(X_test)         # transform only on test

    print("  Scaling complete (fitted on training data only).\n")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
#  5. Model definitions
# ─────────────────────────────────────────────

def build_models(random_state: int = RANDOM_STATE) -> Dict[str, ModelType]:
    """
    Return a dict of {model_name: unfitted_model} for the two classifiers.

    class_weight='balanced' is an extra safety net for Logistic Regression
    (helps even before SMOTE, but here SMOTE has already balanced the data).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter      = 1000,
            class_weight  = "balanced",
            random_state  = random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators  = 200,
            class_weight  = "balanced",
            random_state  = random_state,
            n_jobs        = -1,         # use all CPU cores
        ),
    }


# ─────────────────────────────────────────────
#  6. Training
# ─────────────────────────────────────────────

def train_models(
    models: Dict[str, ModelType],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, ModelType]:
    """
    Fit every model on the (SMOTE-balanced, scaled) training set.

    Returns
    -------
    fitted_models : same dict with fitted estimators.
    """
    fitted = {}
    for name, model in models.items():
        print(f"  Training {name} …")
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  ✓ {name} trained.\n")
    return fitted


# ─────────────────────────────────────────────
#  7. Evaluation metrics
# ─────────────────────────────────────────────

def evaluate_model(
    model: ModelType,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    """
    Compute a comprehensive set of evaluation metrics on the HELD-OUT test set.

    Metrics used
    ------------
    accuracy   – overall correct predictions (misleading for imbalanced data)
    precision  – of all predicted positives, how many are truly positive?
    recall     – of all actual positives, how many did we catch? (sensitivity)
    f1_score   – harmonic mean of precision and recall
    roc_auc    – area under the ROC curve (threshold-independent)
    confusion_matrix – TP, FP, FN, TN breakdown

    Returns
    -------
    metrics dict
    """
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]   # positive-class probability

    metrics = {
        "model"         : model_name,
        "accuracy"      : round(float(accuracy_score(y_test, y_pred)),          4),
        "precision"     : round(float(precision_score(y_test, y_pred,
                                                zero_division=0)),        4),
        "recall"        : round(float(recall_score(y_test, y_pred,
                                             zero_division=0)),           4),
        "f1_score"      : round(float(f1_score(y_test, y_pred,
                                         zero_division=0)),               4),
        "roc_auc"       : round(float(roc_auc_score(y_test, y_prob)),           4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n  ── {model_name} ──")
    for k, v in metrics.items():
        if k not in ("model", "confusion_matrix"):
            print(f"     {k:<15}: {v}")
    print(f"\n  Classification report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=["Healthy", "Disease"],
                                zero_division=0))
    return metrics


# ─────────────────────────────────────────────
#  8. Plotting helpers
# ─────────────────────────────────────────────

def plot_class_distribution(y_train_raw: np.ndarray, y_train_smote: np.ndarray) -> None:
    """Bar chart comparing class counts before/after SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution: Before vs After SMOTE", fontsize=14, fontweight="bold")

    for ax, y, title in zip(
        axes,
        [y_train_raw, y_train_smote],
        ["Before SMOTE (Training Set)", "After SMOTE (Training Set)"],
    ):
        counts = pd.Series(y).value_counts().sort_index()
        bars = ax.bar(
            ["Healthy (0)", "Disease (1)"],
            counts.values,
            color=["steelblue", "tomato"],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Sample Count")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(counts.values) * 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(all_metrics: List[Dict[str, object]]) -> None:
    """Side-by-side heatmaps for each model's confusion matrix."""
    n = len(all_metrics)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, all_metrics):
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred Healthy", "Pred Disease"],
            yticklabels=["True Healthy", "True Disease"],
            linewidths=0.5, linecolor="grey",
        )
        ax.set_title(f"{m['model']}\nConfusion Matrix", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(fitted_models: Dict[str, ModelType], X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Overlay ROC curves for all models on a single axes."""
    palette = ["royalblue", "tomato", "seagreen", "darkorange"]
    fig, ax = plt.subplots(figsize=(8, 7))

    for (name, model), color in zip(fitted_models.items(), palette):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val     = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {auc_val:.4f})")

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")

    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((-0.02, 1.05))
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves – Disease Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metrics_comparison(all_metrics: List[Dict[str, object]]) -> None:
    """Grouped bar chart comparing key scalar metrics across models."""
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x    = np.arange(len(metric_keys))
    width = 0.35
    palette = ["royalblue", "tomato"]

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (m, color) in enumerate(zip(all_metrics, palette)):
        vals = [float(cast(Any, m[k])) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=m["model"],
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([k.replace("_", " ").title() for k in metric_keys], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(fitted_models: Dict[str, ModelType], feature_names: List[str]) -> None:
    """
    For models that expose feature_importances_ (Random Forest),
    plot the top-20 features.  Logistic Regression shows absolute
    coefficient magnitudes as a proxy for importance.
    """
    for name, model in fitted_models.items():
        fig, ax = plt.subplots(figsize=(10, 7))

        if isinstance(model, RandomForestClassifier):
            importances = model.feature_importances_
            title_suffix = "Feature Importances (Gini)"
        else:
            importances = np.abs(model.coef_[0])
            title_suffix = "Feature Importances (|Coefficient|)"

        indices = np.argsort(importances)[::-1][:20]   # top-20 only
        top_names  = [feature_names[i] for i in indices]
        top_values = importances[indices]

        ax.barh(range(len(top_names)), top_values[::-1],
                color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=10)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(f"{name} – {title_suffix}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        sns.despine(ax=ax)

        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(PLOTS_DIR, f"feature_importance_{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  9. Persist results
# ─────────────────────────────────────────────

def save_metrics(all_metrics: List[Dict[str, object]]) -> None:
    """Write all scalar metrics to a JSON file for downstream use."""
    # Flatten confusion matrix to plain list (already is, just make sure)
    path = os.path.join(METRICS_DIR, "evaluation_metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved: {path}")

    # Also write a human-readable CSV summary
    rows = []
    for m in all_metrics:
        row = {k: v for k, v in m.items() if k != "confusion_matrix"}
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(METRICS_DIR, "metrics_summary.csv"), index=False
    )
    print(f"  Saved: {os.path.join(METRICS_DIR, 'metrics_summary.csv')}")


# ─────────────────────────────────────────────
#  10. Main orchestration
# ─────────────────────────────────────────────

def run_pipeline():
    """
    End-to-end pipeline execution.

    Order is critical for correctness:
        1. Load real data
        2. STRATIFIED split  →  train set | test set
        3. Scaler fitted on TRAIN only, applied to both
        4. SMOTE on TRAIN only
        5. Train models
        6. Evaluate on ORIGINAL (non-SMOTE) test set
        7. Plot & save
    """
    print("\n" + "="*55)
    print(" RARE DISEASE DETECTION PIPELINE")
    print("="*55)

    # ── Step 1 : Load real dataset ──────────────────────────
    print("\n[1/7] Loading real healthcare dataset …")
    df = load_real_healthcare_data()
    target_col = df.columns[-1]  # Last column is the target
    feature_names = [c for c in df.columns if c != target_col]

    # ── Step 2 : Stratified train/test split ───────────────
    print("[2/7] Splitting data (stratified) …")
    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)

    # ── Step 3 : Scaling fitted on training only ───────────
    print("[3/7] Scaling features …")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ── Step 4 : SMOTE on training set ONLY ───────────────
    print("[4/7] Applying SMOTE to training set …")
    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

    # ── Step 5 : Train models ─────────────────────────────
    print("[5/7] Training models …")
    models        = build_models()
    fitted_models = train_models(models, X_train_smote, y_train_smote)

    # ── Step 6 : Evaluate on untouched test set ───────────
    print("[6/7] Evaluating models on test set …")
    all_metrics = []
    for name, model in fitted_models.items():
        metrics = evaluate_model(model, name, X_test_scaled, y_test)
        all_metrics.append(metrics)

    # ── Step 7 : Plots & metrics persistence ──────────────
    print("[7/7] Saving plots and metrics …\n")
    plot_class_distribution(y_train, y_train_smote)
    plot_confusion_matrices(all_metrics)
    plot_roc_curves(fitted_models, X_test_scaled, y_test)
    plot_metrics_comparison(all_metrics)
    plot_feature_importance(fitted_models, feature_names)
    save_metrics(all_metrics)

    print("\n" + "="*55)
    print(" Pipeline complete!")
    print(f"  Plots  → ./{PLOTS_DIR}/")
    print(f"  Metrics→ ./{METRICS_DIR}/")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Change working directory to the project root so relative paths work consistently
    os.chdir(Path(__file__).parent.parent)
    os.makedirs(PLOTS_DIR,   exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    run_pipeline()