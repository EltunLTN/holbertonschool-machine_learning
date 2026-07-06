# Holberton School - Machine Learning

This repository contains tasks from the **Machine Learning** curriculum at Holberton School. The project builds mathematical foundations (linear algebra, calculus, probability), data visualization skills, data pipeline tooling, and introductory machine learning algorithms.

## Directory Structure

- `math/linear_algebra/` — Matrix operations with Python lists and NumPy
- `math/advanced_linear_algebra/` — Determinants, inverses, and matrix definiteness
- `math/calculus/` — Calculus theory and polynomial derivative/integral functions
- `math/probability/` — Discrete and continuous probability distribution classes
- `math/multivariate_prob/` — Mean, covariance, correlation, and multivariate normal PDF
- `math/bayesian_prob/` — Likelihood, intersection, marginal, and posterior probability
- `math/plotting/` — Matplotlib line, scatter, histogram, and bar plots
- `pipeline/pandas/` — DataFrame creation, manipulation, and analysis with Pandas
- `unsupervised_learning/` — Standardization, PCA, K-Means, and hierarchical clustering

Each directory contains a `README.md` with task descriptions and a list of files.

## Requirements

- Python 3.x
- NumPy, Matplotlib, Pandas, Scikit-learn, SciPy (depending on the module)
- `pycodestyle` style compliance where required by the checker

## Usage

Navigate to a task directory and run a file:

```bash
cd math/probability
python3 binomial.py
```

Some plotting and pipeline scripts expect data files (for example `pca.npz` or CSV datasets) in the working directory.


