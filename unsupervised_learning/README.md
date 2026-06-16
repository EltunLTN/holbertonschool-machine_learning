# Unsupervised Learning

This directory contains introductory unsupervised learning tasks from the Holberton machine learning curriculum. It uses Scikit-learn for preprocessing, dimensionality reduction, and clustering.

## Learning Objectives

- Standardize tabular data with `StandardScaler`
- Reduce dimensionality with Principal Component Analysis (PCA)
- Cluster data with K-Means and evaluate cluster count with inertia and silhouette score
- Perform agglomerative hierarchical clustering with optional PCA preprocessing

## Files

- `0-standardize.py`: Defines `Standardize(X)` — z-score normalization using `StandardScaler`
- `1-pca.py`: Defines `Apply_PCA(X, n_components, random_state)` — PCA transform and fitted model
- `2-k_means.py`: Defines `K_Means(X, n_clusters, random_state)` — fits a K-Means model
- `3-optimal_k.py`: Defines `optimal_k(X, max_clusters, random_state)` — inertia and silhouette vs k
- `4-agglomerative.py`: Defines `Agglomerative_Clustering(...)` — hierarchical clustering with optional PCA

## Requirements

- Python 3.x
- NumPy
- Scikit-learn
- `pycodestyle` style compliance (where required by checker)

## Usage

Run Python task files with:

```bash
python3 0-standardize.py
```

Example:

```python
import numpy as np
from Apply_PCA import Apply_PCA  # or import from 1-pca module

X = np.random.randn(100, 4)
X_pca, pca = Apply_PCA(X, n_components=2, random_state=42)
```
