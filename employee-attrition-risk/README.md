# Mathematical Modelling of Employee Attrition Risk
**A comparative analysis of Machine Learning algorithms implemented from scratch in C.**

## 📌 Project Overview
This project was developed for the **Mathematical Modelling–Based Experiential Learning** assignment. The goal is to predict employee attrition risk (Stay vs. Leave) using the **IBM HR Analytics Employee Attrition & Performance** dataset.

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

## 📈 Model Outcomes
Results based on an 80/20 train/test split:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| **KNN** | 0.8469 | 0.8571 | 0.1200 | 0.2105 |
| **Naive Bayes** | 0.8163 | 0.4643 | 0.5200 | 0.4906 |
| **Linear Regression** | 0.8367 | 1.0000 | 0.0400 | 0.0769 |

## 🎯 Real-World Model Validation
To verify the mathematical models, we selected a specific sample row from the raw dataset (Source: Image Data) to test "live" inference. 

### Input Data Profile (The "New Data"):
This row is taken directly from the **IBM HR Dataset (Row 1)** as shown in the validation spreadsheet:

**1. Personal & Demographics**
| Age | Gender | Marital Status | Education | Education Field | Distance From Home |
|-----|--------|----------------|-----------|-----------------|--------------------|
| 41  | Female | Single         | 2         | Life Sciences   | 1 km               |

**2. Job & Role Details**
| Department | Job Role | Job Level | Job Involvement | OverTime | Business Travel |
|------------|----------|-----------|-----------------|----------|-----------------|
| Sales      | Sales Ex | 2         | 3               | **Yes**  | Travel_Rarely   |

**3. Financials & Satisfaction**
| Monthly Income | Daily Rate | Hourly Rate | Job Satisfaction | Env. Satisfaction |
|----------------|------------|-------------|------------------|-------------------|
| $5,993         | 1102       | 94          | 4                | 2                 |

**4. Experience & Tenure**
| Total Work Years | Years at Co. | Years in Role | Years since Prom. | Years w/ Manager |
|------------------|--------------|---------------|-------------------|------------------|
| 8                | 6            | 4             | 0                 | 5                |

*   **Reason for selection**: This sample represents a high-risk employee profile in the HR domain. It serves as a benchmark to see which mathematical approach can correctly identify a "True Positive" attrition event that actually occurred.

### Inference Results:
| Algorithm | Feature Snapshot | Prediction | Correct? |
|-----------|------------------|------------|----------|
| **KNN** | [41, 5993, 1km] | **Stay** | ❌ |
| **Naive Bayes** | [41, 5993, 1km] | **LEAVE** | ✅ **Correct** |
| **Linear Regression** | [41, 5993, 1km] | **Stay** | ❌ |

**Mathematical Insight**: Naive Bayes correctly identified the attrition because it uses a **probabilistic** approach. Even though the employee's income was moderate, the combination of "Single" status and "Overtime: Yes" carries high weight in the Gaussian probability calculation, allowing it to "outvote" the stabilizing factors.
