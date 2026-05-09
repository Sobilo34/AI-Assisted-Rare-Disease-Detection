"""
Rare Disease Detection using Machine Learning + SMOTE
=====================================================
Handles extreme class imbalance in REAL WORLD healthcare datasets
using Synthetic Minority Over-sampling Technique (SMOTE).

Pipeline:
    1. Real dataset loading from CSV (5,000 patient records, 99:1 imbalance)
    2. Preprocessing (scaling, encoding)
    3. Stratified train-test split (preserves class ratio)
    4. SMOTE oversampling on training set only (prevents leakage)
    5. Multiple classifiers with cross-validation
    6. Comprehensive evaluation: ROC-AUC, PR-AUC, F1, Confusion Matrix
    7. Feature importance analysis

This is the NAIVE PROMPT VERSION - basic instruction.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score


# ─────────────────────────────────────────────
# 1. LOAD REAL HEALTHCARE DATASET
# ─────────────────────────────────────────────

def load_real_healthcare_dataset() -> pd.DataFrame:
    """
    Load real patient data from CSV file.
    Dataset: 5,000 patient records with 10 clinical features + 1 disease label.
    Class imbalance: 99% healthy, 1% disease (extremely imbalanced rare disease scenario).

    Returns
    -------
    pd.DataFrame with healthcare features and disease_status label.
    """
    csv_path = Path(__file__).parent.parent / "data" / "rare_disease_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\nDataset loaded from: {csv_path}")
    print(f"  Shape: {df.shape[0]} samples × {df.shape[1]} features")
    print(f"  Columns: {list(df.columns)}")

    # Check class distribution
    label_col = df.columns[-1]  # Assume last column is label
    n_positive = (df[label_col] == 1).sum()
    n_negative = (df[label_col] == 0).sum()
    print(f"\nDataset composition:")
    print(f"  Total samples  : {len(df)}")
    print(f"  Positive (rare): {n_positive}  ({n_positive/len(df)*100:.2f}%)")
    print(f"  Negative       : {n_negative}  ({n_negative/len(df)*100:.2f}%)")
    print(f"  Imbalance ratio: 1 : {n_negative // max(n_positive, 1)}\n")

    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """Separate features and target, normalize feature names."""
    df = df.copy()

    # Assume last column is the target label
    label_col = df.columns[-1]
    y = df[label_col].values
    X = df.drop(columns=[label_col]).values
    feature_names = df.drop(columns=[label_col]).columns.tolist()

    return X, y, feature_names


# ─────────────────────────────────────────────
# 3. SMOTE VARIANTS
# ─────────────────────────────────────────────

SMOTE_VARIANTS = {
    "SMOTE (standard)":    SMOTE(random_state=42),
    "Borderline-SMOTE":    BorderlineSMOTE(random_state=42),
    "SVM-SMOTE":           SVMSMOTE(random_state=42),
}


# ─────────────────────────────────────────────
# 4. CLASSIFIERS
# ─────────────────────────────────────────────

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", class_weight="balanced",
        probability=True, random_state=42
    ),
}


# ─────────────────────────────────────────────
# 5. EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate_model(name, clf, X_test, y_test, results_store):
    """Compute and print key metrics; store for later comparison."""
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    roc_auc  = roc_auc_score(y_test, y_proba)
    pr_auc   = average_precision_score(y_test, y_proba)
    f1       = f1_score(y_test, y_pred)
    g_mean   = geometric_mean_score(y_test, y_pred)

    results_store[name] = {
        "ROC-AUC":  roc_auc,
        "PR-AUC":   pr_auc,
        "F1":       f1,
        "G-Mean":   g_mean,
        "y_proba":  y_proba,
        "y_pred":   y_pred,
    }

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"  G-Mean  : {g_mean:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Healthy','Rare Disease'])}")
    return roc_auc, pr_auc


# ─────────────────────────────────────────────
# 6. PLOTTING
# ─────────────────────────────────────────────

def plot_all(results, X_test, y_test, feature_names, best_clf_name, best_clf):
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#0d1117")
    title_font = {"color": "#e6edf3", "fontsize": 13, "fontweight": "bold"}
    title_pad = 12.0
    label_font = {"color": "#8b949e", "fontsize": 10}

    # ── Panel 1: ROC curves ──────────────────
    ax1 = fig.add_subplot(3, 3, 1)
    _style_ax(ax1)
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax1.plot(fpr, tpr, label=f"{name} ({res['ROC-AUC']:.3f})", color=color, lw=2)
    ax1.plot([0,1],[0,1], ":", color="#484f58", lw=1)
    ax1.set_title("ROC Curves", fontdict=title_font, pad=title_pad)
    ax1.set_xlabel("False Positive Rate", fontdict=label_font)
    ax1.set_ylabel("True Positive Rate", fontdict=label_font)
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # ── Panel 2: PR curves ───────────────────
    ax2 = fig.add_subplot(3, 3, 2)
    _style_ax(ax2)
    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax2.plot(rec, prec, label=f"{name} ({res['PR-AUC']:.3f})", color=color, lw=2)
    ax2.set_title("Precision-Recall Curves", fontdict=title_font, pad=title_pad)
    ax2.set_xlabel("Recall", fontdict=label_font)
    ax2.set_ylabel("Precision", fontdict=label_font)
    ax2.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # ── Panel 3: Metric comparison bar chart ─
    ax3 = fig.add_subplot(3, 3, 3)
    _style_ax(ax3)
    metrics = ["ROC-AUC", "PR-AUC", "F1", "G-Mean"]
    x = np.arange(len(metrics))
    width = 0.18
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        ax3.bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
    ax3.set_xticks(x + width*1.5)
    ax3.set_xticklabels(metrics, color="#8b949e", fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Metric Comparison", fontdict=title_font, pad=title_pad)
    ax3.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # ── Panel 4: Confusion matrix (best model) ──
    ax4 = fig.add_subplot(3, 3, 4)
    _style_ax(ax4)
    cm = confusion_matrix(y_test, results[best_clf_name]["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
        xticklabels=["Healthy", "Rare Disease"],
        yticklabels=["Healthy", "Rare Disease"],
        cbar_kws={"shrink": 0.8},
    )
    ax4.set_title(f"Confusion Matrix\n({best_clf_name})", fontdict=title_font, pad=title_pad)
    ax4.set_xlabel("Predicted", fontdict=label_font)
    ax4.set_ylabel("Actual", fontdict=label_font)
    ax4.tick_params(colors="#8b949e")

    # ── Panel 5: Feature importance (RF / GB only) ──
    ax5 = fig.add_subplot(3, 3, 5)
    _style_ax(ax5)
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        indices = np.argsort(importances)[-15:]   # top 15
        ax5.barh(
            range(len(indices)),
            importances[indices],
            color="#3fb950", alpha=0.85
        )
        ax5.set_yticks(range(len(indices)))
        ax5.set_yticklabels([feature_names[i] for i in indices], color="#8b949e", fontsize=8)
        ax5.set_title(f"Feature Importances\n({best_clf_name})", fontdict=title_font, pad=title_pad)
        ax5.set_xlabel("Importance", fontdict=label_font)
    else:
        ax5.text(0.5, 0.5, "Not available\nfor this model",
                 ha="center", va="center", color="#8b949e", fontsize=12)
        ax5.set_title("Feature Importances", fontdict=title_font, pad=title_pad)

    # ── Panel 6: Class distribution before SMOTE ──
    ax6 = fig.add_subplot(3, 3, 6)
    _style_ax(ax6)
    unique, counts = np.unique(y_test, return_counts=True)
    # approximate full dataset from test proportion
    labels = ["Healthy", "Rare Disease"]
    bar_colors = ["#58a6ff", "#f78166"]
    ax6.bar(labels, counts, color=bar_colors, alpha=0.85, width=0.4)
    for i, cnt in enumerate(counts):
        ax6.text(i, cnt + 0.5, str(cnt), ha="center", va="bottom", color="#e6edf3", fontsize=11)
    ax6.set_title("Test Set Class Distribution", fontdict=title_font, pad=title_pad)
    ax6.set_ylabel("Count", fontdict=label_font)

    # ── Panel 7: Probability distribution ────
    ax7 = fig.add_subplot(3, 3, 7)
    _style_ax(ax7)
    proba_best = results[best_clf_name]["y_proba"]
    ax7.hist(proba_best[y_test == 0], bins=40, alpha=0.7, color="#58a6ff", label="Healthy")
    ax7.hist(proba_best[y_test == 1], bins=40, alpha=0.7, color="#f78166", label="Rare Disease")
    ax7.axvline(0.5, color="#e3b341", ls="--", lw=1.5, label="Default threshold (0.5)")
    ax7.set_title(f"Predicted Probability Distribution\n({best_clf_name})", fontdict=title_font, pad=title_pad)
    ax7.set_xlabel("P(Rare Disease)", fontdict=label_font)
    ax7.set_ylabel("Count", fontdict=label_font)
    ax7.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # ── Panel 8: Threshold analysis ───────────
    ax8 = fig.add_subplot(3, 3, 8)
    _style_ax(ax8)
    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores, precisions, recalls = [], [], []
    for thr in thresholds:
        y_thr = (proba_best >= thr).astype(int)
        f1_scores.append(f1_score(y_test, y_thr, zero_division=0))
        precisions.append(
            np.sum((y_thr == 1) & (y_test == 1)) / (np.sum(y_thr == 1) + 1e-9)
        )
        recalls.append(
            np.sum((y_thr == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-9)
        )
    ax8.plot(thresholds, f1_scores,   color="#3fb950", lw=2, label="F1")
    ax8.plot(thresholds, precisions,  color="#58a6ff", lw=2, label="Precision")
    ax8.plot(thresholds, recalls,     color="#f78166", lw=2, label="Recall")
    best_thr = thresholds[np.argmax(f1_scores)]
    ax8.axvline(best_thr, color="#e3b341", ls="--", lw=1.5,
                label=f"Best F1 @ {best_thr:.2f}")
    ax8.set_title("Threshold Analysis", fontdict=title_font, pad=title_pad)
    ax8.set_xlabel("Decision Threshold", fontdict=label_font)
    ax8.set_ylabel("Score", fontdict=label_font)
    ax8.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # ── Panel 9: Summary table ────────────────
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_facecolor("#0d1117")
    ax9.axis("off")
    table_data = [[name,
                   f"{res['ROC-AUC']:.4f}",
                   f"{res['PR-AUC']:.4f}",
                   f"{res['F1']:.4f}",
                   f"{res['G-Mean']:.4f}"]
                  for name, res in results.items()]
    col_labels = ["Model", "ROC-AUC", "PR-AUC", "F1", "G-Mean"]
    tbl = ax9.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if row % 2 == 0 else "#0d1117")
        cell.set_text_props(color="#e6edf3")
        cell.set_edgecolor("#30363d")
    ax9.set_title("Results Summary", fontdict=title_font, pad=title_pad)

    plt.suptitle(
        "Rare Disease Detection  ·  ML with SMOTE Oversampling",
        fontsize=16, fontweight="bold", color="#e6edf3", y=1.01
    )
    plt.tight_layout()
    out_path = Path(__file__).parent.parent / 'outputs' / 'figures' / 'naive_results.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    print(f"\n[✓] Plot saved → {out_path}")
    plt.close()


def _style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(smote_variant_name: str = "SMOTE (standard)"):
    # Change to repo root for consistent data paths
    os.chdir(Path(__file__).parent.parent)
    
    print("=" * 60)
    print("  RARE DISEASE DETECTION  |  ML + SMOTE PIPELINE")
    print("=" * 60)

    # ── Data ──────────────────────────────────
    df = load_real_healthcare_dataset()
    X, y, feature_names = preprocess(df)
    y = np.asarray(y, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # ── Scaling ───────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── SMOTE ─────────────────────────────────
    smote = SMOTE_VARIANTS[smote_variant_name]
    print(f"\nApplying {smote_variant_name} …")
    print(f"  Before — positives: {y_train.sum()}, negatives: {(y_train==0).sum()}")
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"  After  — positives: {y_resampled.sum()}, negatives: {(y_resampled==0).sum()}")

    # ── Train & evaluate ──────────────────────
    results = {}
    trained_clfs = {}
    print("\nTraining classifiers …")

    for clf_name, clf in CLASSIFIERS.items():
        clf.fit(X_resampled, y_resampled)
        trained_clfs[clf_name] = clf
        evaluate_model(clf_name, clf, X_test_scaled, y_test, results)

    # ── Cross-validation (best model by ROC-AUC) ──
    best_clf_name = max(results, key=lambda n: results[n]["ROC-AUC"])
    best_clf = trained_clfs[best_clf_name]

    print(f"\n{'='*60}")
    print(f"  Best model: {best_clf_name}  (ROC-AUC={results[best_clf_name]['ROC-AUC']:.4f})")
    print(f"{'='*60}")

    # Stratified 5-fold CV on the full (pre-SMOTE) training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pipeline = ImbPipeline([
        ("smote",      SMOTE(random_state=42)),
        ("classifier", CLASSIFIERS[best_clf_name].__class__(
            **CLASSIFIERS[best_clf_name].get_params()
        )),
    ])
    cv_scores = cross_val_score(
        cv_pipeline, X_train_scaled, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1
    )
    print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Plots ─────────────────────────────────
    plot_all(results, X_test_scaled, y_test, feature_names, best_clf_name, best_clf)

    return results, best_clf, scaler, feature_names


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results, best_clf, scaler, feature_names = run_pipeline(
        smote_variant_name="SMOTE (standard)"   # or "Borderline-SMOTE" / "SVM-SMOTE"
    )