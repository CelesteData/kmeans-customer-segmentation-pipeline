# Unsupervised Customer & Behavioral Segmentation Pipeline

An end-to-end unsupervised machine learning workflow built in Python (`scikit-learn`) that transforms raw behavioral and transactional records into distinct, actionable segments using **K-Means clustering**.

## Overview

In operational and financial environments, raw customer or account data often hides distinct behavioral patterns—such as high-value steady users versus high-risk churn candidates. This project implements a reproducible clustering pipeline designed to:

1. **Validate & Clean Raw Inputs:** Handle missing values, audit data distributions, and detect anomalies prior to distance-based modeling.
2. **Engineer & Standardize Features:** Apply scaling (`StandardScaler`) and dimensionality reduction (`PCA`) so high-magnitude variables do not distort Euclidean distance calculations.
3. **Determine Optimal Cluster Geometry:** Evaluate candidate cluster counts ($k = 2 \dots 10$) systematically using the **Elbow Method (Inertia / WCSS)** diagnostic.
4. **Profile & Operationalize Segments:** Translate mathematical cluster centroids back into plain-language behavioral personas and targeted business recommendations.

---

## Tech Stack

- **Language:** Python 3.10+
- **Data Manipulation & Validation:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`)
- **Visualization:** `matplotlib`
- **Environment:** Jupyter Notebook

---

## Methodology & Pipeline Architecture

```text
Raw Dataset
   │
   ▼
1. Data Audit & Integrity Checks (missingness, duplicates, outlier diagnostics)
   │
   ▼
2. Feature Engineering & Scaling (StandardScaler normalization)
   │
   ▼
3. Optimal k Selection (Elbow Method analysis)
   │
   ▼
4. K-Means Model Training (k-means++ initialization, fixed random_state)
   │
   ▼
5. Dimensionality Reduction (2D PCA projection) & Cluster Profiling
