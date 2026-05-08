"""
data_generation.py

Generates a synthetic healthcare dataset for rare disease detection.

Project:
AI-Assisted Rare Disease Detection Using Structured Prompt Engineering
and SMOTE-Based Imbalanced Classification.

Purpose:
- Simulate a high-stakes healthcare ML scenario.
- Create a severely imbalanced dataset: 99% Healthy, 1% Rare Disease.
- Save the dataset for later preprocessing, SMOTE, model training, and evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification


RANDOM_STATE = 42
N_SAMPLES = 5000

FEATURE_NAMES = [
    "age",
    "blood_pressure",
    "cholesterol",
    "glucose",
    "bmi",
    "heart_rate",
    "inflammation_marker",
    "genetic_risk_score",
    "oxygen_saturation",
    "white_blood_cell_count",
]

TARGET_NAME = "disease_status"


def generate_rare_disease_dataset(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a synthetic rare disease detection dataset.

    Class labels:
    - 0 = Healthy
    - 1 = Rare Disease

    The dataset is intentionally imbalanced to simulate a realistic healthcare
    challenge where rare disease cases are much less common than healthy cases.
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.99, 0.01],
        class_sep=1.2,
        flip_y=0.0,
        random_state=random_state,
    )

    dataset = pd.DataFrame(X, columns=FEATURE_NAMES)
    dataset[TARGET_NAME] = y

    return dataset


def print_dataset_summary(dataset: pd.DataFrame) -> None:
    """
    Print basic dataset information and class imbalance summary.
    """

    print("\n========== DATASET SUMMARY ==========")
    print(f"Dataset shape: {dataset.shape}")

    print("\nFirst 5 rows:")
    print(dataset.head())

    print("\nClass distribution:")
    class_counts = dataset[TARGET_NAME].value_counts().sort_index()
    print(class_counts)

    print("\nClass distribution percentages:")
    class_percentages = dataset[TARGET_NAME].value_counts(normalize=True).sort_index() * 100
    print(class_percentages.round(2).astype(str) + "%")

    print("\nLabel meaning:")
    print("0 = Healthy")
    print("1 = Rare Disease")


def plot_class_distribution(dataset: pd.DataFrame, output_path: Path) -> None:
    """
    Create and save a bar chart showing class imbalance.
    """

    class_counts = dataset[TARGET_NAME].value_counts().sort_index()
    labels = ["Healthy", "Rare Disease"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, class_counts.values)
    plt.title("Class Distribution: Rare Disease Detection Dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")

    for index, value in enumerate(class_counts.values):
        plt.text(index, value, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nClass distribution chart saved to: {output_path}")


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """
    Save generated dataset as a CSV file.
    """

    dataset.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")


def main() -> None:
    """
    Main execution function.
    """

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    data_dir = project_root / "data"
    figures_dir = project_root / "outputs" / "figures"

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "rare_disease_dataset.csv"
    figure_path = figures_dir / "class_distribution_before_smote.png"

    dataset = generate_rare_disease_dataset()

    print_dataset_summary(dataset)
    save_dataset(dataset, dataset_path)
    plot_class_distribution(dataset, figure_path)


if __name__ == "__main__":
    main()