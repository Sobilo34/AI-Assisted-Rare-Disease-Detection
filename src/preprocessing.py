"""
Preprocessing Pipeline for Imbalanced Healthcare Classification

This module implements a professional ML preprocessing pipeline for rare disease detection
with emphasis on preventing data leakage and handling class imbalance through SMOTE.

Key Features:
    - Stratified train-test split to preserve class distribution
    - StandardScaler fitted ONLY on training data to prevent leakage
    - SMOTE applied ONLY to training data
    - Comprehensive logging and visualization
    - Reproducible pipeline with fixed random states
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Any, Tuple, cast

def load_dataset(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw dataset and separate features from target.

    Args:
        data_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset has unexpected structure.
    """
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(path)
    print(f"\n{'='*70}")
    print(f"DATASET LOADED")
    print(f"{'='*70}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Assume last column is target, others are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def perform_stratified_split(X: pd.DataFrame, y: pd.Series, 
                             test_size: float = 0.2, 
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split to preserve class distribution.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of data for test set. Default: 0.2
        random_state (int): Seed for reproducibility. Default: 42

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\n{'='*70}")
    print(f"STRATIFIED TRAIN-TEST SPLIT")
    print(f"{'='*70}")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Split ratio: {test_size*100:.1f}% test, {(1-test_size)*100:.1f}% train")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler fitted ONLY on training data.
    
    This prevents data leakage by ensuring the scaler learns statistics
    exclusively from the training set before transforming test data.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]: (X_train_scaled, X_test_scaled, scaler)
            - X_train_scaled: Scaled training features
            - X_test_scaled: Scaled test features
            - scaler: Fitted StandardScaler object for reference
    """
    scaler = StandardScaler()
    
    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n{'='*70}")
    print(f"FEATURE SCALING (StandardScaler)")
    print(f"{'='*70}")
    print(f"Scaler fitted on training data only (prevents leakage)")
    print(f"Training data - Mean: {X_train_scaled.mean():.4f}, "
          f"Std: {X_train_scaled.std():.4f}")
    print(f"Test data - Mean: {X_test_scaled.mean():.4f}, "
          f"Std: {X_test_scaled.std():.4f}")
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train_scaled: pd.DataFrame, y_train: pd.Series, 
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to training data ONLY to handle class imbalance.
    
    SMOTE is applied AFTER scaling to avoid fitting on raw data statistics.
    This ensures the synthetic samples are generated in the scaled space.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target.
        random_state (int): Seed for reproducibility. Default: 42

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X_train_resampled, y_train_resampled)
    """
    smote = SMOTE(random_state=random_state)
    resampled = smote.fit_resample(X_train_scaled, y_train)
    if len(resampled) == 3:
        X_resampled_raw, y_resampled_raw, _ = resampled
    else:
        X_resampled_raw, y_resampled_raw = resampled
    
    # Convert back to DataFrame
    X_train_resampled = pd.DataFrame(cast(Any, X_resampled_raw), columns=X_train_scaled.columns)
    y_train_resampled = pd.Series(cast(Any, y_resampled_raw), name=y_train.name)
    
    print(f"\n{'='*70}")
    print(f"SMOTE RESAMPLING (Applied to Training Data Only)")
    print(f"{'='*70}")
    print(f"Original training set size: {X_train_scaled.shape[0]}")
    print(f"Resampled training set size: {X_train_resampled.shape[0]}")
    print(f"Increase: {X_train_resampled.shape[0] - X_train_scaled.shape[0]} samples")
    
    return X_train_resampled, y_train_resampled


def print_class_distributions(y_original: pd.Series, y_train_resampled: pd.Series,
                              y_test: pd.Series) -> None:
    """
    Print detailed class distribution analysis.

    Args:
        y_original (pd.Series): Original target distribution.
        y_train_resampled (pd.Series): Resampled training target.
        y_test (pd.Series): Test target.
    """
    print(f"\n{'='*70}")
    print(f"CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nOriginal Dataset:")
    print(y_original.value_counts(normalize=True).sort_index())
    
    print(f"\nTraining Set (After SMOTE):")
    print(y_train_resampled.value_counts(normalize=True).sort_index())
    
    print(f"\nTest Set (No SMOTE applied):")
    print(y_test.value_counts(normalize=True).sort_index())


def print_dataset_shapes(X_train: pd.DataFrame, X_train_resampled: pd.DataFrame,
                        X_test: pd.DataFrame, y_train: pd.Series,
                        y_train_resampled: pd.Series, y_test: pd.Series) -> None:
    """
    Print comprehensive dataset shape information.

    Args:
        X_train (pd.DataFrame): Original training features.
        X_train_resampled (pd.DataFrame): Resampled training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Original training target.
        y_train_resampled (pd.Series): Resampled training target.
        y_test (pd.Series): Test target.
    """
    print(f"\n{'='*70}")
    print(f"DATASET SHAPES - BEFORE AND AFTER PREPROCESSING")
    print(f"{'='*70}")
    
    print(f"\nBEFORE SMOTE:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    
    print(f"\nAFTER SMOTE:")
    print(f"  X_train_resampled: {X_train_resampled.shape}")
    print(f"  y_train_resampled: {y_train_resampled.shape}")
    
    print(f"\nTEST SET (Unchanged):")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")


def visualize_class_distribution(y_train_original: pd.Series, 
                                 y_train_resampled: pd.Series,
                                 y_test: pd.Series,
                                 output_path: str) -> None:
    """
    Create and save visualization comparing class distributions.

    Args:
        y_train_original (pd.Series): Original training target.
        y_train_resampled (pd.Series): Resampled training target.
        y_test (pd.Series): Test target.
        output_path (str): Path to save the visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Class Distribution: Before and After SMOTE', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Original Training Distribution
    train_counts = y_train_original.value_counts().sort_index()
    axes[0].bar(train_counts.index, train_counts.values, color='#FF6B6B', alpha=0.7)
    axes[0].set_title('Training Set (Before SMOTE)', fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Plot 2: Resampled Training Distribution
    resampled_counts = y_train_resampled.value_counts().sort_index()
    axes[1].bar(resampled_counts.index, resampled_counts.values, color='#4ECDC4', alpha=0.7)
    axes[1].set_title('Training Set (After SMOTE)', fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(resampled_counts.values):
        axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Plot 3: Test Distribution
    test_counts = y_test.value_counts().sort_index()
    axes[2].bar(test_counts.index, test_counts.values, color='#95E1D3', alpha=0.7)
    axes[2].set_title('Test Set (Unchanged)', fontweight='bold')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Count')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_counts.values):
        axes[2].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()


def save_processed_data(X_train_resampled: pd.DataFrame, y_train_resampled: pd.Series,
                       X_test_scaled: pd.DataFrame, y_test: pd.Series,
                       output_dir: str) -> None:
    """
    Save processed datasets to CSV files.

    Args:
        X_train_resampled (pd.DataFrame): Resampled and scaled training features.
        y_train_resampled (pd.Series): Resampled training target.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target.
        output_dir (str): Directory to save the processed data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save files
    X_train_resampled.to_csv(output_path / 'X_train_resampled.csv', index=False)
    y_train_resampled.to_csv(output_path / 'y_train_resampled.csv', index=False)
    X_test_scaled.to_csv(output_path / 'X_test_scaled.csv', index=False)
    y_test.to_csv(output_path / 'y_test.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"PROCESSED DATA SAVED")
    print(f"{'='*70}")
    print(f"Output directory: {output_path}")
    print(f"✓ X_train_resampled.csv ({X_train_resampled.shape})")
    print(f"✓ y_train_resampled.csv ({y_train_resampled.shape})")
    print(f"✓ X_test_scaled.csv ({X_test_scaled.shape})")
    print(f"✓ y_test.csv ({y_test.shape})")


def main():
    """
    Execute the complete preprocessing pipeline.
    
    Pipeline Flow:
        1. Load raw dataset
        2. Perform stratified train-test split
        3. Scale features (fit on training data only)
        4. Apply SMOTE to training data
        5. Log class distributions and shapes
        6. Create visualization
        7. Save processed datasets
    """
    # Define paths
    data_path = Path(__file__).parent.parent / 'data' / 'rare_disease_dataset.csv'
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    figures_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    
    print("\n" + "="*70)
    print("RARE DISEASE DETECTION - PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Load dataset
    X, y = load_dataset(str(data_path))
    
    # Step 2: Stratified split
    X_train, X_test, y_train, y_test = perform_stratified_split(X, y)
    
    # Step 3: Scale features (fit on training data ONLY)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 4: Apply SMOTE to training data ONLY
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)
    
    # Step 5: Print class distributions
    print_class_distributions(y, y_train_resampled, y_test)
    
    # Step 6: Print dataset shapes
    print_dataset_shapes(X_train, X_train_resampled, X_test, 
                        y_train, y_train_resampled, y_test)
    
    # Step 7: Visualize class distributions
    visualize_class_distribution(y_train, y_train_resampled, y_test,
                                str(figures_dir / 'smote_class_distribution.png'))
    
    # Step 8: Save processed data
    save_processed_data(X_train_resampled, y_train_resampled, 
                       X_test_scaled, y_test, str(processed_dir))
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print("Pipeline Summary:")
    print("  ✓ Data leakage prevented (scaler fitted on train data only)")
    print("  ✓ Class imbalance addressed (SMOTE applied to training set)")
    print("  ✓ Test set preserved (no synthetic samples in test)")
    print("  ✓ Reproducible pipeline (fixed random states)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
