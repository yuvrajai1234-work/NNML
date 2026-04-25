# Employee Attrition Risk Analysis (Kaggle Dataset)

## Overview
This project predicts employee attrition using the **IBM HR Analytics Employee Attrition & Performance** dataset from Kaggle. The models are implemented from scratch in C to demonstrate the underlying mathematical principles of each algorithm.

## File Structure
- `dataset.h`: Core header for loading, normalizing, and shuffling the CSV data.
- `knn.c`: K-Nearest Neighbors implementation.
- `Naive_Bayes.c`: Gaussian Naive Bayes implementation.
- `linear_regression.c`: Linear Regression (Gradient Descent) implementation.
- `IBM_HR_Attrition.csv`: The dataset sourced from Kaggle/IBM.
- `.gitignore`: Standard ignore rules for C binaries and datasets.

## Results
Results based on the IBM HR dataset (80/20 train/test split):

### K-Nearest Neighbors
- Accuracy: 0.8469
- Precision: 0.8571
- Recall: 0.1200
- F1-score: 0.2105

### Naive Bayes
- Accuracy: 0.8163
- Precision: 0.4643
- Recall: 0.5200
- F1-score: 0.4906

### Linear Regression
- Accuracy: 0.8367
- Precision: 1.0000
- Recall: 0.0400
- F1-score: 0.0769

## Summary
- **Best Model**: **KNN** and **Linear Regression** show high accuracy, but **Naive Bayes** provides a better balance of Precision and Recall (highest F1-score) for this specific imbalanced dataset.
- **Strengths**: Naive Bayes handles the probabilistic nature of the features well, while Linear Regression is very conservative with its predictions.
- **Weaknesses**: The dataset is highly imbalanced (fewer "Yes" cases), making it difficult for simple models to achieve high recall without more advanced techniques like oversampling or class weighting.

## Compilation
To compile and run any model:
```bash
gcc <filename>.c -o <output_name> -lm
./<output_name>
```
Example:
```bash
gcc Naive_Bayes.c -o nb -lm
./nb
```
