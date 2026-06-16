# Math - Multivariate Probability

This directory contains multivariate statistics tasks from the Holberton machine learning curriculum. It covers mean and covariance computation, correlation matrices, and the multivariate normal probability density function.

## Learning Objectives

- Compute the mean vector and covariance matrix of a dataset
- Convert a covariance matrix to a correlation matrix
- Implement the multivariate normal PDF from first principles
- Validate array shapes and dimensions with clear error messages

## Files

- `0-mean_cov.py`: Defines `mean_cov(X)` — returns mean (1, d) and covariance (d, d) from data (n, d)
- `1-correlation.py`: Defines `correlation(C)` — converts a square covariance matrix to a correlation matrix
- `multinormal.py`: Defines `MultiNormal` class with `pdf(x)` for a multivariate normal distribution

## Requirements

- Python 3.x
- NumPy
- `pycodestyle` style compliance (where required by checker)

## Usage

Run Python task files with:

```bash
python3 0-mean_cov.py
```

Example:

```python
import numpy as np
from multinormal import MultiNormal

data = np.random.randn(5, 100)
mn = MultiNormal(data)
```
