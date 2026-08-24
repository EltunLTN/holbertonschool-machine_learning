# Math - Foundations for Machine Learning

This directory contains mathematical foundations and implementations for machine learning from the Holberton machine learning curriculum. It covers essential linear algebra, calculus, probability, and numerical computation concepts needed to understand and implement machine learning algorithms.

## Learning Objectives

- Build strong mathematical foundations in linear algebra and matrix operations
- Understand calculus concepts used in optimization (derivatives, gradients)
- Master probability theory including distributions and Bayesian inference
- Implement mathematical operations efficiently using NumPy
- Apply mathematical concepts to real machine learning tasks

## Subdirectories

### 1. **linear_algebra/** - Matrix Operations and Transformations
   - Matrix operations, reshaping, and slicing
   - Transposition, element-wise operations, multiplication
   - Matrix decomposition and transformations

### 2. **calculus/** - Differentiation and Integration
   - Derivative computation and symbolic differentiation
   - Polynomial operations
   - Series summation and numerical integration
   - Application to optimization problems

### 3. **probability/** - Discrete Probability Distributions
   - Basic probability concepts and axioms
   - Binomial and normal distributions
   - Probability mass functions (PMF) and cumulative distribution functions (CDF)
   - Confidence intervals and statistical inference

### 4. **bayesian_prob/** - Bayesian Inference and Statistics
   - Bayes' theorem and conditional probability
   - Prior and posterior distributions
   - Likelihood and marginal probability
   - Continuous Bayesian inference using Beta distributions

### 5. **multivariate_prob/** - Multivariate Probability
   - Multivariate normal distribution
   - Correlation matrices and covariance
   - Correlation coefficient computation
   - Ellipse plotting and correlation visualization

### 6. **advanced_linear_algebra/** - Advanced Matrix Operations
   - Eigenvalue decomposition
   - Matrix determinant and rank
   - Matrix inverse and pseudo-inverse
   - Definite matrices and applications

### 7. **convolutions_and_pooling/** - Signal Processing Operations
   - Convolution operations (valid, same, padded)
   - Multi-channel convolutions
   - Pooling operations (max, average)
   - Applications to image processing

### 8. **plotting/** - Data Visualization
   - Creating and formatting plots
   - Scatter plots, line plots, histograms
   - 3D plotting
   - Visualization best practices

## Requirements

- Python 3.x
- NumPy (primary library for numerical computation)
- SciPy (for advanced mathematical functions)
- Matplotlib (for visualization in plotting module)
- `pycodestyle` style compliance where required

## Mathematical Concepts by Module

### Linear Algebra
- Vectors and matrices as representations of data
- Matrix multiplication (dot products, outer products)
- Norms (L1, L2, Frobenius)
- Dimensionality and rank
- Determinant and invertibility

### Calculus
- Partial derivatives and gradients (basis for optimization)
- Chain rule for backpropagation
- Taylor series approximation
- Numerical differentiation and integration

### Probability
- Sample space, events, and probability axioms
- Conditional probability and independence
- Random variables and distributions
- Expected value and variance
- Central Limit Theorem

### Bayesian Inference
- Prior, likelihood, and posterior
- Bayes' theorem: P(θ|x) = P(x|θ)P(θ) / P(x)
- Conjugate priors (Beta-Binomial)
- Continuous posterior distributions

### Signal Processing
- Convolution as feature extraction
- Kernel/filter design
- Downsampling via pooling
- Stride and padding parameters

## Implementation Philosophy

All modules follow these principles:
1. **From scratch**: Implementations built with NumPy, avoiding library shortcuts
2. **Educational**: Code is clear and annotated, prioritizing understanding
3. **Efficient**: Uses vectorized NumPy operations for performance
4. **Well-tested**: Handles edge cases and validates inputs
5. **Practical**: Examples demonstrate real-world usage

## Common Workflow

Typical workflow for using math modules in machine learning:

```python
import numpy as np

# 1. Load and preprocess data (linear_algebra)
from linear_algebra import matrix_operations

# 2. Compute statistics and distributions (probability)
from probability import binomial_distribution
P_x = binomial_distribution(x=26, n=130, p=0.2)

# 3. Apply mathematical transformations (calculus, convolutions)
from calculus import poly_derivative
gradient = poly_derivative(coefficients)

# 4. Visualize results (plotting)
from plotting import create_scatter_plot
create_scatter_plot(data)

# 5. Use in ML models (foundations)
# These math operations power all ML algorithms
```

## Usage Pattern

Each subdirectory contains numbered files (0-file.py, 1-file.py, etc.) following curriculum progression:

```bash
# Run individual tasks
python3 0-linear_algebra_basics.py

# Or import into your own scripts
from linear_algebra import matrix_multiplication
result = matrix_multiplication(matrix_a, matrix_b)
```

## Mathematical Formula Reference

Key equations implemented in this directory:

**Matrix Multiplication**: C[i,j] = Σₖ A[i,k] × B[k,j]

**Norm**: ||x||₂ = √(Σ xᵢ²)

**Derivative**: f'(x) = lim(h→0) [f(x+h) - f(x)] / h

**Bayes' Theorem**: P(A|B) = P(B|A) × P(A) / P(B)

**Convolution**: (f * g)[n] = Σₘ f[m] × g[n - m]

**Normal Distribution**: f(x) = (1/(σ√(2π))) × exp(-(x-μ)²/(2σ²))

## Learning Path

Recommended order for learning:

1. **Start with**: linear_algebra/ (foundational)
2. **Then learn**: calculus/ (needed for optimization)
3. **Move to**: probability/ (understanding data)
4. **Explore**: bayesian_prob/ (inference)
5. **Apply to**: convolutions_and_pooling/ (image processing)
6. **Visualize**: plotting/ (communicate results)
7. **Advanced**: advanced_linear_algebra/ (deeper understanding)

## Practical Applications

- **Linear Algebra**: Feature transformation, dimensionality reduction, matrix decomposition
- **Calculus**: Gradient descent optimization, backpropagation
- **Probability**: Model evaluation, confidence intervals, hypothesis testing
- **Bayesian**: Posterior inference, probabilistic modeling
- **Convolutions**: Image feature extraction, CNNs, signal processing
- **Plotting**: Model analysis, result visualization, presentation

## Performance Considerations

- Use NumPy vectorization instead of Python loops (100x faster)
- Avoid matrix inverses when possible; use decomposition
- Use sparse matrices for large, sparse data
- Leverage broadcasting for memory efficiency
- Profile code to identify bottlenecks

## References

- Boyd, S., & Vandenberghe, L. (2018). Introduction to Applied Linear Algebra
- Strang, G. (2016). Introduction to Linear Algebra (5th Edition)
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective
