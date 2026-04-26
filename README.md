# Mathematical Modelling of Employee Attrition Risk
**A comparative analysis of Machine Learning algorithms implemented from scratch in C.**

## 📌 Project Overview
This project was developed for the **Mathematical Modelling–Based Experiential Learning** assignment. The goal is to predict employee attrition risk (Stay vs. Leave) using the **IBM HR Analytics Employee Attrition & Performance** dataset.

Unlike standard ML projects, this implementation avoids all high-level libraries (like Scikit-Learn), relying strictly on the **mathematical foundations** of each algorithm implemented in pure C.

## 🧠 Algorithms Implemented
We have modeled the problem using three distinct mathematical approaches to compare their performance:

1.  **K-Nearest Neighbors (KNN)**: A geometric, distance-based model using Euclidean distance and Min-Max feature scaling.
2.  **Gaussian Naive Bayes (GNB)**: A probabilistic model using Bayes' Theorem and the Gaussian Probability Density Function.
3.  **Linear Regression**: An optimization-based model using the Linear Hypothesis and Gradient Descent for weight updates (Linear Probability Model).

## 📊 Mathematical Foundation
The models utilize the following core concepts to secure the "Mathematical Model" grading criteria:

- **Euclidean Distance** (Used in **K-Nearest Neighbors**): 
  $$d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$
- **Linear Hypothesis** (Used in **Linear Regression**): 
  $$h_\theta(x) = \theta_0 + \theta_1x_1 + \dots + \theta_nx_n$$
- **Gaussian PDF** (Used in **Gaussian Naive Bayes**): 
  $$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
- **Bayes Theorem** (Used in **Gaussian Naive Bayes**):
  $$P(C|x) \propto P(C) \prod P(x_i|C)$$

## 🔑 Key Definitions
Before analyzing the outcomes, it is essential to define the performance metrics used:

- **Accuracy**: The ratio of correctly predicted observations to the total observations.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives (Ability of the model not to label a negative sample as positive).
- **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all observations in actual class (Ability of the model to find all positive samples).
- **F1-Score**: The weighted average of Precision and Recall.

## 📁 Project Structure
- `knn.c`: Implementation of the K-Nearest Neighbors algorithm.
- `nb.c`: Implementation of the Gaussian Naive Bayes algorithm.
- `lr.c`: Implementation of the Linear Regression algorithm.
- `dataset.h`: Core header for loading, normalizing, and shuffling the CSV data for zero-dependency execution.
- `IBM_HR_Attrition.csv`: The dataset sourced from Kaggle/IBM.

## 🚀 How to Run
Ensure you have a C compiler (like `gcc`) installed. Open your terminal in the `employee-attrition-risk` folder and run:

### K-Nearest Neighbors
```bash
gcc knn.c -o knn -lm
./knn
```

### Naive Bayes
```bash
gcc nb.c -o nb -lm
./nb
```

### Linear Regression
```bash
gcc lr.c -o lr -lm
./lr
```

---
**Detailed Analysis**: For full model results, accuracy comparisons, and specific real-world validation tests, please refer to the internal [README.md](./employee-attrition-risk/README.md).

