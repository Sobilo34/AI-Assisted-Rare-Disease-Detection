# Architects of the Algorithmic Future  
## AI-Assisted Rare Disease Detection Using SMOTE

> CCC1243 Artificial Intelligence Project  
> Bachelor in Computer Science (Hons)  
> Albukhary International University  
> By: Bilal Oyeleke Soliu  

---

# Project Overview

This project focuses on building a machine learning pipeline for **rare disease detection** under severe class imbalance in healthcare datasets.

The project follows **Hurdle A: Imbalanced Data (SDG 3 - Good Health and Well-being)** from the CCC1243 Artificial Intelligence assignment.

The system uses:

- A synthetic healthcare dataset
- 5,000 patient records
- 10 healthcare-related features
- Extreme `99:1` class imbalance
- SMOTE oversampling
- Logistic Regression
- Random Forest

The major goal of this project was not only to build a working ML pipeline, but also to investigate how **prompt engineering** and **In-Context Learning (ICL)** affect AI-assisted software engineering.

---

# Problem Statement

Rare disease datasets are highly imbalanced.

In real healthcare environments:

- Most patients are healthy
- Very few patients actually have the disease

This creates a major ML challenge because models become biased toward predicting the majority class.

A model can achieve:

- `99% accuracy`
- while still completely failing to detect disease patients.

This project focuses more on:

- Recall
- False negatives
- Precision-Recall tradeoff
- Healthcare interpretation

instead of relying only on accuracy.

---

# Objectives

- Build a reproducible ML pipeline for rare disease detection
- Prevent data leakage during preprocessing
- Apply SMOTE correctly on training data only
- Compare Logistic Regression and Random Forest
- Evaluate healthcare-aware metrics
- Investigate prompt evolution from naive prompts to structured technical prompts
- Study how AI-generated code improves through prompt refinement

---

# Technologies Used

## Programming Language

- Python

## Libraries

- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib
- seaborn

## Development Tools

- VS Code
- Claude AI
- GitHub

---

# Machine Learning Workflow

```text
Dataset → Train/Test Split → StandardScaler → SMOTE → Model Training → Evaluation
