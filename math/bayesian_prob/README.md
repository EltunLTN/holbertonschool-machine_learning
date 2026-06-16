# Math - Bayesian Probability

This directory contains Bayesian inference tasks from the Holberton machine learning curriculum. Tasks model a clinical trial scenario: given observed side effects among n patients, compute likelihood, intersection, marginal, and posterior probabilities over candidate values of p.

## Learning Objectives

- Compute binomial likelihood for discrete probability hypotheses
- Apply Bayes' theorem with prior beliefs
- Normalize posterior distributions over a grid of probabilities
- Extend to continuous posteriors using the Beta distribution and SciPy

## Files

- `0-likelihood.py`: Defines `likelihood(x, n, P)` — binomial likelihood for each probability in P
- `1-intersection.py`: Defines `intersection(x, n, P, Pr)` — likelihood × prior for each hypothesis
- `2-marginal.py`: Defines `marginal(x, n, P, Pr)` — total probability of observing the data
- `3-posterior.py`: Defines `posterior(x, n, P, Pr)` — normalized posterior over P
- `100-continuous.py`: Defines `posterior(x, n, p1, p2)` — probability that p lies in [p1, p2] using a Beta posterior

## Requirements

- Python 3.x
- NumPy
- SciPy (for `100-continuous.py`)
- `pycodestyle` style compliance (where required by checker)

## Usage

Run Python task files with:

```bash
python3 0-likelihood.py
```

Example:

```python
import numpy as np
from likelihood import likelihood

P = np.linspace(0, 1, 11)
print(likelihood(x=26, n=130, P=P))
```
