"""
master_prompt.py
================
Complete Healthcare ML Pipeline for Rare Disease Detection
using REAL WORLD DATA with SMOTE balancing & comprehensive evaluation.

This master script orchestrates the end-to-end pipeline:
  1. Load real healthcare dataset (99% healthy, 1% disease)
  2. Stratified train-test split (80-20) preserving class balance
  3. Scale features (StandardScaler on train only)
  4. Apply SMOTE to training set only (prevent leakage)
  5. Train Logistic Regression & Random Forest
  6. Evaluate on original imbalanced test set (real deployment conditions)
  7. Generate comprehensive visualizations & clincal analysis

Healthcare Metric Rationale
----------------------------
In rare disease detection, the cost asymmetry between error types is extreme:

    False Negative (FN) — predicting "healthy" when the patient has a disease.
        → Patient goes undiagnosed, disease progresses, potentially fatal.

    False Positive (FP) — predicting "disease" when the patient is healthy.
        → Unnecessary follow-up tests; anxiety; extra cost.

Because FNs carry far greater clinical risk than FPs, RECALL (sensitivity) is
the primary metric.  Accuracy is misleading: a naive model that predicts
"healthy" for every sample achieves 99 % accuracy while missing every patient.

Author : Healthcare ML Pipeline
Version: 1.0.0
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier
from sklearn.preprocessing       import StandardScaler
from sklearn.model_selection     import train_test_split
from sklearn.metrics             import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42


# ── Configuration ──────────────────────────────────────────────────────────────
os.chdir(Path(__file__).parent.parent)  # Change to repo root
OUTPUT_FIGURES_DIR = "Outputs/figures"
OUTPUT_METRICS_DIR = "Outputs/metrics"



# ── Model Definitions ─────────────────────────────────────────────────────────

def build_models(random_state: int = RANDOM_STATE) -> dict:
    """
    Instantiate untrained classifier objects with fixed hyperparameters.

    Parameters
    ----------
    random_state : Seed shared across both models for reproducibility.

    Returns
    -------
    dict mapping model names → sklearn estimator instances.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter     = 1_000,         # ensure convergence on scaled data
            random_state = random_state,
            solver       = "lbfgs",       # memory-efficient for dense data
            C            = 1.0,           # default regularisation strength
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators = 200,           # sufficient trees for stable OOB
            max_depth    = None,          # fully-grown trees (forest regularises)
            random_state = random_state,
            n_jobs       = -1,            # parallelise across all CPU cores
            class_weight = None,          # weight balancing done via SMOTE
        ),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(
    models:        dict,
    X_train_smote: np.ndarray,
    y_train_smote: np.ndarray,
) -> dict:
    """
    Fit each model on the SMOTE-balanced training data.

    Training on SMOTE data means the model learns from a balanced class
    distribution.  Evaluation is always performed on the original, imbalanced
    test set to simulate real deployment conditions.

    Parameters
    ----------
    models        : Dict of untrained estimators from `build_models`.
    X_train_smote : SMOTE-resampled feature matrix.
    y_train_smote : SMOTE-resampled labels.

    Returns
    -------
    dict mapping model names → fitted estimators.
    """
    fitted = {}
    print("\n" + "=" * 55)
    print("  MODEL TRAINING  (on SMOTE-balanced train set)")
    print("=" * 55)
    for name, clf in models.items():
        clf.fit(X_train_smote, y_train_smote)
        fitted[name] = clf
        print(f"  [✓] {name} trained.")
    return fitted


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    clf:    LogisticRegression | RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name:   str,
) -> dict:
    """
    Compute the full healthcare evaluation suite for one fitted model.

    Metrics computed:
        accuracy   — overall correct predictions (can be misleading here)
        precision  — of all disease predictions, how many are correct
        recall     — of all true disease cases, how many are caught (PRIMARY)
        f1         — harmonic mean of precision and recall
        roc_auc    — area under the ROC curve
        confusion_matrix — raw TP / TN / FP / FN counts

    Parameters
    ----------
    clf    : Fitted sklearn estimator.
    X_test : Scaled test features (never touched by SMOTE or scaler fitting).
    y_test : True test labels (imbalanced, as in real deployment).
    name   : Human-readable model name for reporting.

    Returns
    -------
    dict with all metric values plus raw arrays for plotting.
    """
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]   # probability of disease class

    metrics = {
        "model"            : name,
        "accuracy"         : round(float(accuracy_score(y_test, y_pred)), 4),
        "precision"        : round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall"           : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score"         : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc"          : round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix" : confusion_matrix(y_test, y_pred).tolist(),
        # Raw arrays for ROC curve plotting
        "_y_proba"         : y_proba,
        "_y_test"          : y_test,
    }

    _print_model_metrics(metrics)
    return metrics


def _print_model_metrics(m: dict) -> None:
    """Print a formatted per-model metric block to stdout."""
    print(f"\n  ── {m['model']} ──")
    print(f"    Accuracy  : {m['accuracy']:.4f}  ← potentially misleading (99 % naive baseline)")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}  ← PRIMARY metric (miss rate)")
    print(f"    F1-Score  : {m['f1_score']:.4f}")
    print(f"    ROC-AUC   : {m['roc_auc']:.4f}")
    cm = np.array(m["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()
    print(f"    Confusion  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"    False Negatives (missed diagnoses) : {fn}  ← clinical risk")


def evaluate_all_models(
    fitted_models: dict,
    X_test:        np.ndarray,
    y_test:        np.ndarray,
) -> dict:
    """
    Evaluate all fitted models and collect results into a single dict.

    Parameters
    ----------
    fitted_models : Dict of name → fitted estimator.
    X_test        : Scaled test features.
    y_test        : True test labels.

    Returns
    -------
    dict mapping model names → metric dicts.
    """
    print("\n" + "=" * 55)
    print("  MODEL EVALUATION  (on original imbalanced test set)")
    print("=" * 55)
    results = {}
    for name, clf in fitted_models.items():
        results[name] = evaluate_model(clf, X_test, y_test, name)
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    results:  dict,
    save_dir: str = OUTPUT_FIGURES_DIR,
) -> None:
    """
    Plot annotated confusion matrices for all models side by side.

    Each matrix shows absolute counts for TP, TN, FP, FN cells.
    Colour intensity reflects count magnitude; annotations add percentages
    relative to true class totals (sensitivity / specificity intuition).

    Parameters
    ----------
    results  : Output of `evaluate_all_models`.
    save_dir : Directory to write the PNG.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        "Confusion Matrices — Test Set (Imbalanced)",
        fontsize=13, fontweight="bold",
    )

    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred: Healthy", "Pred: Disease"],
            yticklabels=["True: Healthy", "True: Disease"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"},
        )
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(
            f"{name}\n"
            f"Recall={m['recall']:.3f}  |  Precision={m['precision']:.3f}\n"
            f"FN (missed diagnoses) = {fn}",
            fontsize=10, fontweight="bold",
        )
        ax.set_ylabel("Actual Label", fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Confusion matrices saved → {path}")


def plot_roc_curves(
    results:  dict,
    save_dir: str = OUTPUT_FIGURES_DIR,
) -> None:
    """
    Plot ROC curves for all models on a single axis with AUC annotations.

    The ROC curve plots True Positive Rate (recall) vs False Positive Rate at
    every probability threshold.  A higher AUC indicates better discrimination
    between healthy and diseased populations across all operating points.

    Parameters
    ----------
    results  : Output of `evaluate_all_models`.
    save_dir : Output directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, m), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(m["_y_test"], m["_y_proba"])
        ax.plot(
            fpr, tpr,
            label=f"{name}  (AUC = {m['roc_auc']:.3f})",
            color=color, linewidth=2.5,
        )

    # Random classifier reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random Classifier (AUC = 0.500)")
    # Shade the area under the first model curve for emphasis
    first_m = list(results.values())[0]
    fpr0, tpr0, _ = roc_curve(first_m["_y_test"], first_m["_y_proba"])
    ax.fill_between(fpr0, tpr0, alpha=0.07, color=palette[0])

    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate  (Recall / Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — Rare Disease Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] ROC curves saved → {path}")


def plot_metric_comparison(
    results:  dict,
    save_dir: str = OUTPUT_FIGURES_DIR,
) -> None:
    """
    Grouped bar chart comparing all evaluation metrics across models.

    Accuracy is deliberately included alongside recall and AUC to illustrate
    how it can give a misleadingly high score even when disease cases are
    being missed.

    Parameters
    ----------
    results  : Output of `evaluate_all_models`.
    save_dir : Output directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    metric_keys    = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels  = ["Accuracy", "Precision", "Recall\n(Primary)", "F1-Score", "ROC-AUC"]
    model_names    = list(results.keys())
    palette        = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    data = {name: [results[name][k] for k in metric_keys] for name in model_names}

    x    = np.arange(len(metric_keys))
    width = 0.35
    offsets = np.linspace(-(len(model_names) - 1) * width / 2,
                           (len(model_names) - 1) * width / 2,
                           len(model_names))

    fig, ax = plt.subplots(figsize=(11, 6))
    for (name, vals), offset, color in zip(data.items(), offsets, palette):
        bars = ax.bar(x + offset, vals, width, label=name, color=color,
                      alpha=0.85, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Model Performance Comparison — Healthcare Metrics",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(0.99, color="grey", linestyle="--", linewidth=1,
               label="Naive accuracy (always predict healthy)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # Highlight the Recall bar group (primary metric)
    ax.axvspan(x[2] - 0.45, x[2] + 0.45, alpha=0.08, color="#e74c3c",
               label="_nolegend_")
    ax.text(x[2], 1.10, "★ Primary\nMetric", ha="center", fontsize=8,
            color="#c0392b", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "metric_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Metric comparison chart saved → {path}")


def plot_precision_recall_curves(
    results: dict,
    save_dir: str = OUTPUT_FIGURES_DIR,
) -> None:
    """
    Plot precision-recall curves for all models.

    PR curves are especially informative for rare disease screening because
    they show how precision changes as recall increases under heavy imbalance.
    """
    os.makedirs(save_dir, exist_ok=True)
    palette = ["#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, m), color in zip(results.items(), palette):
        precision, recall, _ = precision_recall_curve(m["_y_test"], m["_y_proba"])
        ax.plot(
            recall,
            precision,
            label=f"{name}  (AP curve)",
            color=color,
            linewidth=2.5,
        )

    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title("Precision-Recall Curves — Rare Disease Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "precision_recall_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Precision-recall curves saved → {path}")


def plot_feature_importance(
    fitted_models: dict,
    feature_names: list[str],
    save_dir: str = OUTPUT_FIGURES_DIR,
) -> None:
    """
    Plot feature importance for Logistic Regression and Random Forest.

    Feature names come directly from the real CSV columns, so the chart uses
    the actual clinical feature labels instead of generic placeholders.
    """
    os.makedirs(save_dir, exist_ok=True)

    lr = fitted_models["Logistic Regression"]
    rf = fitted_models["Random Forest"]

    lr_importance = np.abs(lr.coef_.ravel())
    rf_importance = rf.feature_importances_

    lr_order = np.argsort(lr_importance)[::-1]
    rf_order = np.argsort(rf_importance)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].barh(
        np.array(feature_names)[lr_order][::-1],
        lr_importance[lr_order][::-1],
        color="#4C78A8",
        edgecolor="white",
    )
    axes[0].set_title("Logistic Regression - Feature Importances (|Coefficient|)", fontweight="bold")
    axes[0].set_xlabel("Importance Score")
    axes[0].grid(axis="x", alpha=0.25, linestyle="--")

    axes[1].barh(
        np.array(feature_names)[rf_order][::-1],
        rf_importance[rf_order][::-1],
        color="#F58518",
        edgecolor="white",
    )
    axes[1].set_title("Random Forest - Feature Importances", fontweight="bold")
    axes[1].set_xlabel("Importance Score")
    axes[1].grid(axis="x", alpha=0.25, linestyle="--")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Feature importance chart saved → {path}")


def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_smote: np.ndarray,
    save_dir: str = OUTPUT_FIGURES_DIR,
) -> None:
    """
    Show class balance before and after SMOTE.

    This is useful evidence for the assignment because it shows the data
    imbalance and how SMOTE only affects the training set.
    """
    os.makedirs(save_dir, exist_ok=True)

    healthy_counts = [int((y_train == 0).sum()), int((y_test == 0).sum()), int((y_train_smote == 0).sum())]
    disease_counts = [int((y_train == 1).sum()), int((y_test == 1).sum()), int((y_train_smote == 1).sum())]
    labels = ["Train", "Test", "Train + SMOTE"]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_healthy = ax.bar(x - width / 2, healthy_counts, width, label="Healthy (0)", color="#4C78A8")
    bars_disease = ax.bar(x + width / 2, disease_counts, width, label="Disease (1)", color="#E45756")

    for bars in (bars_healthy, bars_disease):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(healthy_counts + disease_counts) * 0.01,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution Before and After SMOTE", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Class distribution chart saved → {path}")


# ── Healthcare Analysis ───────────────────────────────────────────────────────

def print_healthcare_analysis(results: dict) -> None:
    """
    Print a structured clinical interpretation of the evaluation results.

    Addresses three key questions posed in the project requirements:
        1. Why recall is the critical metric in disease detection.
        2. Why accuracy is misleading in imbalanced healthcare settings.
        3. Clinical impact of False Positives vs False Negatives.
    """
    divider = "=" * 65
    print(f"\n{divider}")
    print("  HEALTHCARE ANALYSIS — METRIC INTERPRETATION")
    print(divider)

    print("""
  1. WHY RECALL IS CRITICAL IN RARE DISEASE DETECTION
  ─────────────────────────────────────────────────────
  Recall (sensitivity) = TP / (TP + FN)

  A False Negative means the model predicts "healthy" for a patient
  who actually has the disease.  In clinical practice this translates
  directly to a missed diagnosis:
    • The patient receives no treatment.
    • The disease may progress to an irreversible or fatal stage.
    • Early-intervention windows are lost.

  Maximising recall — even at the cost of some false alarms — is the
  medically correct operating point for rare disease screening.

  2. WHY ACCURACY IS MISLEADING WITH 99:1 IMBALANCE
  ─────────────────────────────────────────────────────
  A trivial classifier that predicts "Healthy" for every sample:
    → Accuracy = 99.0 %   (correct 4 950 out of 5 000 times)
    → Recall   =  0.0 %   (misses every single disease case)

  High accuracy here is entirely explained by the dominant negative
  class, not by any learned discriminative ability.  Using accuracy
  alone would endorse a model that is clinically useless.

  3. FALSE POSITIVES vs FALSE NEGATIVES — CLINICAL IMPACT
  ─────────────────────────────────────────────────────────
  False Positive (FP) — "Disease" predicted for a healthy patient:
    Consequence  : Follow-up tests, patient anxiety, additional cost.
    Severity     : Moderate — correctable with secondary screening.

  False Negative (FN) — "Healthy" predicted for a disease patient:
    Consequence  : Missed diagnosis, disease progression, risk of death.
    Severity     : HIGH — potentially irreversible clinical harm.

  Conclusion: In screening pipelines, we deliberately accept more FPs
  (lower precision) to drive FNs toward zero (higher recall).
""")

    print("  RESULTS SUMMARY")
    print("  ─────────────────────────────────────────────────────")
    for name, m in results.items():
        cm   = np.array(m["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  [{name}]")
        print(f"    Recall (sensitivity) : {m['recall']:.4f}")
        print(f"    Precision            : {m['precision']:.4f}")
        print(f"    ROC-AUC              : {m['roc_auc']:.4f}")
        print(f"    False Negatives      : {fn}  (missed disease cases)")
        print(f"    False Positives      : {fp}  (unnecessary referrals)")
    print(f"\n{divider}\n")


# ── Metrics Persistence ───────────────────────────────────────────────────────

def save_metrics(results: dict, save_dir: str = OUTPUT_METRICS_DIR) -> None:
    """
    Persist evaluation metrics to JSON and a human-readable CSV.

    NumPy arrays (_y_proba, _y_test) are stripped before serialisation as
    they are not JSON-compatible and are only needed for plotting.

    Parameters
    ----------
    results  : Output of `evaluate_all_models`.
    save_dir : Output directory for metric files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────
    clean = {}
    for name, m in results.items():
        clean[name] = {k: v for k, v in m.items() if not k.startswith("_")}

    json_path = os.path.join(save_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  [✓] Metrics JSON saved → {json_path}")

    # ── CSV ───────────────────────────────────────────────────────────────
    scalar_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    rows = []
    for name, m in results.items():
        row = {"model": name}
        row.update({k: m[k] for k in scalar_keys})
        cm = np.array(m["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        row.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})
        rows.append(row)

    csv_path = os.path.join(save_dir, "evaluation_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [✓] Metrics CSV  saved → {csv_path}")


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_training_pipeline(preprocessed: dict) -> dict:
    """
    Execute model training, evaluation, visualisation, and reporting.

    Parameters
    ----------
    preprocessed : Dict output from `preprocessing.run_preprocessing_pipeline`.

    Returns
    -------
    dict mapping model names → full evaluation metric dicts.
    """
    # ── Train ──────────────────────────────────────────────────────────────
    models        = build_models()
    fitted_models = train_models(
        models,
        preprocessed["X_train_smote"],
        preprocessed["y_train_smote"],
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    results = evaluate_all_models(
        fitted_models,
        preprocessed["X_test"],
        preprocessed["y_test"],
    )

    # ── Visualise ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  GENERATING VISUALISATIONS")
    print("=" * 55)
    plot_class_distribution(
        preprocessed["y_train"],
        preprocessed["y_test"],
        preprocessed["y_train_smote"],
    )
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_precision_recall_curves(results)
    plot_metric_comparison(results)
    plot_feature_importance(fitted_models, preprocessed["feature_names"])

    # ── Clinical Analysis ──────────────────────────────────────────────────
    print_healthcare_analysis(results)

    # ── Save Metrics ───────────────────────────────────────────────────────
    print("=" * 55)
    print("  SAVING METRICS")
    print("=" * 55)
    save_metrics(results)

    return results


# ── Data Loading & Preprocessing ───────────────────────────────────────────

def load_real_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the real healthcare dataset from CSV.
    
    Returns
    -------
    X : Feature matrix (n_samples, n_features)
    y : Target labels (n_samples,)
    feature_names : Real CSV feature names
    """
    csv_path = Path(__file__).parent.parent / "data" / "rare_disease_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  [✓] Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"      Columns: {list(df.columns)}")
    
    feature_names = df.columns[:-1].tolist()

    # Assume last column is target, others are features
    X = np.asarray(df.iloc[:, :-1].values, dtype=np.float64)
    y = np.asarray(df.iloc[:, -1].values, dtype=np.int64)

    return X, y, feature_names


def preprocess_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    End-to-end preprocessing: split → scale → SMOTE.
    
    Parameters
    ----------
    X : Feature matrix.
    y : Target labels.
    
    Returns
    -------
    dict with keys: X_train_smote, y_train_smote, X_test, y_test, feature_names
    """
    print("\n" + "=" * 55)
    print("  DATA PREPROCESSING")
    print("=" * 55)
    
    # Step 1: Stratified train-test split
    print("\n  [Step 1] Stratified 80-20 split (preserves class ratio)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"    Train: {len(y_train)} samples  |  Test: {len(y_test)} samples")
    print(f"    Train disease ratio: {y_train.mean():.4f}")
    print(f"    Test  disease ratio: {y_test.mean():.4f}")
    
    # Step 2: Scale features (fit on train only)
    print("\n  [Step 2] StandardScaler (fitted on train only)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"    Train scaled to μ≈0, σ≈1 | Test transformed (no fitting)")
    
    # Step 3: SMOTE on training set only
    print("\n  [Step 3] SMOTE resampling (train only)")
    smote = SMOTE(random_state=RANDOM_STATE)
    smote_result = smote.fit_resample(X_train_scaled, y_train)
    # Handle both 2-tuple and 3-tuple returns
    X_train_smote = np.asarray(smote_result[0], dtype=np.float64)
    y_train_smote = np.asarray(smote_result[1], dtype=np.int64)
    print(f"    Before SMOTE: {len(y_train)} train samples  ({y_train.mean():.4f} disease ratio)")
    print(f"    After SMOTE : {len(y_train_smote)} train samples  ({y_train_smote.mean():.4f} disease ratio)")
    
    preprocessed = {
        "y_train": y_train,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_names": feature_names,
    }
    
    return preprocessed


# ── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🏥  HEALTHCARE ML PIPELINE (MASTER)")
    print("=" * 55)
    
    # ── Load real dataset ──────────────────────────────────────────────────
    print("\n  [Step 0] Loading real healthcare dataset …")
    X, y, feature_names = load_real_dataset()
    print(f"    Disease samples: {(y == 1).sum()} ({y.mean():.4f} ratio)")
    print(f"    Healthy samples: {(y == 0).sum()}")
    
    # ── Preprocessing ──────────────────────────────────────────────────────
    preprocessed = preprocess_dataset(X, y, feature_names)
    
    # ── Training & evaluation ──────────────────────────────────────────────
    results = run_training_pipeline(preprocessed)
    
    print("\n" + "=" * 55)
    print("  🏥  Healthcare ML Pipeline COMPLETE")
    print("=" * 55)
    print(f"      Figures → {os.path.abspath(OUTPUT_FIGURES_DIR)}/")
    print(f"      Metrics → {os.path.abspath(OUTPUT_METRICS_DIR)}/\n")