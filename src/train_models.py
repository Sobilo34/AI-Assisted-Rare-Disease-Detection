"""
Model Training and Evaluation Pipeline for Rare Disease Detection

This module implements a professional ML training pipeline for healthcare classification
with emphasis on healthcare-oriented evaluation metrics and systematic model comparison.

Key Features:
    - Logistic Regression and Random Forest model training
    - Comprehensive evaluation using healthcare-relevant metrics
    - ROC-AUC analysis with curve comparison
    - Confusion matrix visualization
    - Systematic model comparison and reporting
    - Production-ready evaluation practices

HEALTHCARE CONTEXT:
    - In rare disease detection, RECALL is critical: Missing a disease (False Negative)
      is more costly than a false alarm (False Positive). We want to catch all true cases.
    - ACCURACY is misleading in imbalanced datasets: A 99% accuracy model that always
      predicts "healthy" would pass naive accuracy checks but fail clinically.
    - PRECISION matters too: Too many false positives require unnecessary follow-ups.
    - ROC-AUC considers the trade-off between TPR and FPR across all thresholds.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


def load_processed_data(data_dir: str) -> tuple:
    """
    Load preprocessed datasets that have been through SMOTE resampling.

    Args:
        data_dir (str): Path to directory containing processed CSV files.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) as DataFrames/Series.

    Raises:
        FileNotFoundError: If any required processed data file is missing.
    """
    data_path = Path(data_dir)
    
    required_files = [
        'X_train_resampled.csv',
        'y_train_resampled.csv',
        'X_test_scaled.csv',
        'y_test.csv'
    ]
    
    for file in required_files:
        if not (data_path / file).exists():
            raise FileNotFoundError(f"Missing {file} in {data_dir}")
    
    X_train = pd.read_csv(data_path / 'X_train_resampled.csv')
    y_train = pd.read_csv(data_path / 'y_train_resampled.csv').iloc[:, 0]
    X_test = pd.read_csv(data_path / 'X_test_scaled.csv')
    y_test = pd.read_csv(data_path / 'y_test.csv').iloc[:, 0]
    
    print(f"\n{'='*70}")
    print(f"PROCESSED DATA LOADED")
    print(f"{'='*70}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                              random_state: int = 42) -> LogisticRegression:
    """
    Train Logistic Regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int): Seed for reproducibility. Default: 42

    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print(f"\n{'='*70}")
    print(f"LOGISTIC REGRESSION - TRAINED")
    print(f"{'='*70}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       random_state: int = 42) -> RandomForestClassifier:
    """
    Train Random Forest Classifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int): Seed for reproducibility. Default: 42

    Returns:
        RandomForestClassifier: Trained model.
    """
    n_estimators = 100
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    print(f"\n{'='*70}")
    print(f"RANDOM FOREST - TRAINED")
    print(f"{'='*70}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of trees: {n_estimators}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return model


def generate_predictions(model, X_test: pd.DataFrame, model_name: str) -> tuple:
    """
    Generate class and probability predictions from a trained model.

    Args:
        model: Trained ML model.
        X_test (pd.DataFrame): Test features.
        model_name (str): Name of the model for logging.

    Returns:
        tuple: (y_pred, y_proba) class predictions and probability predictions.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    print(f"\n{model_name} - Predictions Generated")
    print(f"  Predicted samples: {X_test.shape[0]}")
    print(f"  Positive predictions: {(y_pred == 1).sum()}")
    print(f"  Negative predictions: {(y_pred == 0).sum()}")
    
    return y_pred, y_proba


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray,
                  model_name: str) -> dict:
    """
    Compute comprehensive evaluation metrics for a model.
    
    HEALTHCARE CONTEXT:
    - Recall (Sensitivity): What proportion of actual diseases did we catch?
      In healthcare, a False Negative (missing a disease) is more costly than
      a False Positive. High recall is often prioritized.
    - Precision: Of all disease predictions, how many were correct?
      Too many false positives lead to unnecessary follow-up tests and anxiety.
    - F1-Score: Harmonic mean of Precision and Recall. Balances both metrics.
    - ROC-AUC: Considers performance across all classification thresholds.

    Args:
        y_true (pd.Series): Ground truth labels.
        y_pred (np.ndarray): Predicted class labels.
        y_proba (np.ndarray): Predicted probabilities for positive class.
        model_name (str): Name of the model.

    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    
    print(f"\n{'='*70}")
    print(f"{model_name} - EVALUATION METRICS")
    print(f"{'='*70}")
    for key, value in metrics.items():
        if key != 'Model':
            print(f"{key:.<20} {value:.4f}")
    
    return metrics


def print_detailed_report(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    """
    Print detailed classification report and confusion matrix.

    Args:
        y_true (pd.Series): Ground truth labels.
        y_pred (np.ndarray): Predicted class labels.
        model_name (str): Name of the model.
    """
    print(f"\n{'='*70}")
    print(f"{model_name} - DETAILED CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Disease']))
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*70}")
    print(f"{model_name} - CONFUSION MATRIX")
    print(f"{'='*70}")
    print(f"                 Predicted Healthy | Predicted Disease")
    print(f"Actual Healthy   {cm[0, 0]:>16} | {cm[0, 1]:>16}")
    print(f"Actual Disease   {cm[1, 0]:>16} | {cm[1, 1]:>16}")


def plot_roc_curves(y_test: pd.Series, y_proba_lr: np.ndarray, 
                   y_proba_rf: np.ndarray, output_path: str) -> None:
    """
    Create and save ROC curve comparison plot for both models.

    Args:
        y_test (pd.Series): Ground truth test labels.
        y_proba_lr (np.ndarray): Probability predictions from Logistic Regression.
        y_proba_rf (np.ndarray): Probability predictions from Random Forest.
        output_path (str): Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Logistic Regression ROC Curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    ax.plot(fpr_lr, tpr_lr, color='#FF6B6B', lw=2.5,
            label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
    
    # Random Forest ROC Curve
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    ax.plot(fpr_rf, tpr_rf, color='#4ECDC4', lw=2.5,
            label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
    
    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((-0.02, 1.02))
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison - Rare Disease Detection', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ ROC curve comparison saved to: {output_path}")
    plt.close()


def plot_confusion_matrices(y_test: pd.Series, y_pred_lr: np.ndarray,
                           y_pred_rf: np.ndarray, output_path: str) -> None:
    """
    Create and save confusion matrix plots for both models.

    Args:
        y_test (pd.Series): Ground truth test labels.
        y_pred_lr (np.ndarray): Class predictions from Logistic Regression.
        y_pred_rf (np.ndarray): Class predictions from Random Forest.
        output_path (str): Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confusion Matrices - Model Comparison', fontsize=14, fontweight='bold')
    
    # Logistic Regression Confusion Matrix
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Logistic Regression', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_xticklabels(['Healthy', 'Disease'])
    axes[0].set_yticklabels(['Healthy', 'Disease'])
    
    # Random Forest Confusion Matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Random Forest', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_xticklabels(['Healthy', 'Disease'])
    axes[1].set_yticklabels(['Healthy', 'Disease'])
    
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to: {output_path}")
    plt.close()


def save_metrics_comparison(metrics_list: list, output_path: str) -> None:
    """
    Save model comparison metrics to CSV file.

    Args:
        metrics_list (list): List of dictionaries containing metrics for each model.
        output_path (str): Path to save the CSV file.
    """
    df_metrics = pd.DataFrame(metrics_list)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(df_metrics.to_string(index=False))
    print(f"\n✓ Metrics comparison saved to: {output_path}")


def main():
    """
    Execute the complete model training and evaluation pipeline.
    
    Pipeline Flow:
        1. Load preprocessed datasets
        2. Train Logistic Regression
        3. Train Random Forest
        4. Generate predictions for both models
        5. Evaluate metrics for both models
        6. Print detailed reports
        7. Create visualizations (ROC curves, confusion matrices)
        8. Compare models systematically
        9. Save results
    """
    # Define paths
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    figures_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    metrics_dir = Path(__file__).parent.parent / 'outputs' / 'metrics'
    
    print("\n" + "="*70)
    print("RARE DISEASE DETECTION - MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    # Step 1: Load processed data
    X_train, y_train, X_test, y_test = load_processed_data(str(data_dir))
    
    # Step 2: Train Logistic Regression
    model_lr = train_logistic_regression(X_train, y_train)
    
    # Step 3: Train Random Forest
    model_rf = train_random_forest(X_train, y_train)
    
    # Step 4: Generate predictions
    y_pred_lr, y_proba_lr = generate_predictions(model_lr, X_test, "Logistic Regression")
    y_pred_rf, y_proba_rf = generate_predictions(model_rf, X_test, "Random Forest")
    
    # Step 5: Evaluate models
    metrics_lr = evaluate_model(y_test, y_pred_lr, y_proba_lr, "Logistic Regression")
    metrics_rf = evaluate_model(y_test, y_pred_rf, y_proba_rf, "Random Forest")
    
    # Step 6: Print detailed reports
    print_detailed_report(y_test, y_pred_lr, "Logistic Regression")
    print_detailed_report(y_test, y_pred_rf, "Random Forest")
    
    # Step 7: Create visualizations
    plot_roc_curves(y_test, y_proba_lr, y_proba_rf,
                   str(figures_dir / 'roc_curve_comparison.png'))
    plot_confusion_matrices(y_test, y_pred_lr, y_pred_rf,
                           str(figures_dir / 'confusion_matrices.png'))
    
    # Step 8: Save metrics comparison
    metrics_all = [metrics_lr, metrics_rf]
    save_metrics_comparison(metrics_all, str(metrics_dir / 'model_comparison.csv'))
    
    # Step 9: Summary
    print(f"\n{'='*70}")
    print(f"TRAINING AND EVALUATION COMPLETE")
    print(f"{'='*70}")
    print("Key Insights:")
    print(f"  • Best Accuracy: {max(metrics_lr['Accuracy'], metrics_rf['Accuracy']):.4f}")
    print(f"  • Best Recall (Critical for Healthcare): "
          f"{max(metrics_lr['Recall'], metrics_rf['Recall']):.4f}")
    print(f"  • Best ROC-AUC: {max(metrics_lr['ROC-AUC'], metrics_rf['ROC-AUC']):.4f}")
    print("\nNote: In healthcare, RECALL is prioritized to minimize False Negatives")
    print("      (missed disease cases), even if it increases False Positives.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()