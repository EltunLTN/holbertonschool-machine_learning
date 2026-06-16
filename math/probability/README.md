# Math - Probability

This directory contains probability distribution classes from the Holberton machine learning curriculum. Each class can be initialized with explicit parameters or estimated from a list of observed data.

## Learning Objectives

- Model common discrete and continuous distributions in Python
- Implement probability mass/density functions (PMF/PDF) and cumulative distribution functions (CDF)
- Validate constructor inputs and handle edge cases
- Estimate distribution parameters from sample data

## Files

- `binomial.py`: Defines `Binomial` with `pmf(k)` and `cdf(k)` for n trials and success probability p
- `poisson.py`: Defines `Poisson` with `pmf(k)` and `cdf(k)` for event rate λ
- `exponential.py`: Defines `Exponential` with `pdf(x)` and `cdf(x)` for rate λ
- `normal.py`: Defines `Normal` with `pdf(x)` and `cdf(x)` using mean and standard deviation

## Requirements

- Python 3.x
- `pycodestyle` style compliance (where required by checker)
- Executable scripts with shebang where required

## Usage

Import and instantiate a distribution class:

```python
from binomial import Binomial

b = Binomial(n=10, p=0.5)
print(b.pmf(5))
print(b.cdf(5))
```

Or estimate parameters from data:

```python
b = Binomial(data=[0, 1, 1, 0, 1, 1, 1])
```

Run a module directly:

```bash
python3 binomial.py
```
