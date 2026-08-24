# Unsupervised Learning - Dimensionality Reduction

This directory contains implementations of dimensionality reduction techniques from the Holberton machine learning curriculum. Dimensionality reduction is used to reduce the number of features in high-dimensional datasets while preserving important information.

## Learning Objectives

- Understand Principal Component Analysis (PCA) and its mathematical foundations
- Implement PCA from scratch using eigendecomposition
- Apply dimensionality reduction to high-dimensional data
- Use PCA for data visualization and feature extraction
- Understand variance preservation and explained variance ratio
- Reduce computational complexity and storage requirements

## Files

- `0-pca.py`: Defines `pca(X, var=0.95)` — compute PCA transformation preserving specified variance

## Requirements

- Python 3.x
- NumPy
- scikit-learn (for comparison/validation)
- `pycodestyle` style compliance where required

## Key Concepts

**Principal Component Analysis (PCA)**: 
- Unsupervised linear transformation that finds principal components (directions of maximum variance)
- Components are orthogonal to each other
- Ranked by amount of variance explained

**Variance Preservation**:
- Parameter `var` specifies minimum fraction of variance to preserve (e.g., 0.95 = 95%)
- Fewer components needed for lower variance thresholds
- Trades reconstruction accuracy for dimensionality reduction

**Principal Components**:
- Eigenvectors of the covariance matrix
- Sorted by corresponding eigenvalues (variance explained)
- First component explains most variance, second explains next most, etc.

**Explained Variance Ratio**: 
- Fraction of total variance explained by each component
- Cumulative ratio helps determine number of components needed

**Applications**:
1. Data visualization (project to 2D/3D for plotting)
2. Feature extraction (use reduced features for downstream models)
3. Noise reduction (discard low-variance components)
4. Computational efficiency (reduce memory and training time)

## Usage

```python
import numpy as np
from dimensionality_reduction import pca

# Load high-dimensional data (e.g., 10000 samples, 784 features)
X = np.random.randn(10000, 784)

# Apply PCA preserving 95% of variance
X_reduced = pca(X, var=0.95)
print(X_reduced.shape)  # Will have fewer than 784 features

# For visualization
X_2d = pca(X, var=0.99)  # Or specify n_components
# Can then plot X_2d[:, 0] vs X_2d[:, 1]
```

## Comparison with Other Techniques

| Technique | Type | Interpretability | Speed | Non-linear |
|-----------|------|------------------|-------|-----------|
| **PCA** | Linear | High | Very Fast | No |
| **t-SNE** | Non-linear | Low | Slow | Yes |
| **UMAP** | Non-linear | Medium | Fast | Yes |
| **Factor Analysis** | Linear | High | Fast | No |

## Implementation Steps

1. **Standardize data**: Center and scale features to zero mean and unit variance
2. **Compute covariance matrix**: Cov = (X^T * X) / (n-1)
3. **Eigendecomposition**: Find eigenvalues and eigenvectors
4. **Sort by variance**: Order eigenvectors by descending eigenvalues
5. **Select components**: Choose components that preserve required variance
6. **Transform data**: Project data onto selected components: X_new = X * W

## Advantages and Limitations

**Advantages**:
- Linear, simple, and interpretable
- Very fast computation
- Effective for many applications
- Well-understood mathematically

**Limitations**:
- Linear only (can't capture non-linear structures)
- Assumes variance = information (not always true)
- Sensitive to feature scaling
- Original features harder to interpret after transformation

## References

- Turk, M., & Pentland, A. (1991). Eigenfaces for recognition
- Jolliffe, I. T. (2002). Principal Component Analysis (comprehensive reference)
