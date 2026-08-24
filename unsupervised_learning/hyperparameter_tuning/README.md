# Unsupervised Learning - Hyperparameter Tuning

This directory contains implementations of advanced hyperparameter optimization techniques from the Holberton machine learning curriculum. It focuses on Gaussian Processes (GP) and Bayesian Optimization for efficient hyperparameter search.

## Learning Objectives

- Understand Gaussian Processes as probability models for functions
- Implement Gaussian Process prediction with mean and variance
- Use acquisition functions to balance exploration vs. exploitation
- Implement Bayesian Optimization for efficient hyperparameter search
- Apply GP-based and Bayesian optimization to real optimization problems
- Understand the trade-off between sample efficiency and computational cost

## Files

- `0-gp.py`: Defines `GaussianProcess(X, Y, kernel=SquaredExponential())` — initialize GP model
- `1-gp.py`: Adds `predict(X_test, return_std=False)` — predict function values at test points
- `2-gp.py`: Adds `update_y(Y_new)` — update GP predictions with new observations
- `3-bayes_opt.py`: Defines `BayesianOptimization(f, X_init, Y_init, bounds, ac_type='ucb')` — initialize Bayesian optimizer
- `4-bayes_opt.py`: Adds `maximize(iterations)` — run Bayesian optimization loop
- `5-bayes_opt.py`: Adds `best_params()` and visualization — get optimal hyperparameters found
- `6-bayes_opt.py`: Adds `optimize_with_gp()` — complete optimization with early stopping

## Requirements

- Python 3.x
- NumPy
- SciPy (for kernel functions and optimization)
- scikit-learn (for comparison and metrics)
- Matplotlib (for visualization)
- `pycodestyle` style compliance where required

## Key Concepts

**Gaussian Process (GP)**:
- Probabilistic model that defines a distribution over functions
- Predictions include both mean (point estimate) and variance (uncertainty estimate)
- Kernel function defines correlation between points
- Non-parametric: grows in complexity with data

**Common Kernel Functions**:
- **Squared Exponential (RBF)**: Smooth, infinitely differentiable
- **Matérn**: Flexibility in smoothness control
- **Linear**: For linear relationships
- **Periodic**: For cyclical patterns

**Mean Function**: Prior belief about the function (usually constant or zero)

**Covariance Function**: Kernel that captures smoothness and correlation structure

**Prediction Uncertainty**: Variance estimate helps identify promising regions to sample

**Bayesian Optimization**:
- Uses Gaussian Process as surrogate model
- Iteratively:
  1. Train GP on observed evaluations
  2. Select next point using acquisition function
  3. Evaluate objective at that point
  4. Update GP with new observation

**Acquisition Functions** (balance exploration vs. exploitation):
1. **Expected Improvement (EI)**: Prioritizes points likely to improve best value found
2. **Upper Confidence Bound (UCB)**: Balance between exploitation (low GP mean) and exploration (high GP variance)
3. **Probability of Improvement (PI)**: Probability that point will improve over current best

**Advantages over Grid/Random Search**:
- Sample efficient: finds good solutions with fewer evaluations
- Handles expensive objective functions
- Naturally handles mixed variable types
- Provides uncertainty estimates

## Usage

```python
import numpy as np
from bayes_opt import BayesianOptimization
from gp import GaussianProcess

# Define objective function (e.g., model cross-validation error)
def objective(x):
    # x is a hyperparameter value
    model = train_model(x)
    return -cross_validation_error(model)  # Negate for maximization

# Define bounds for hyperparameter
bounds = [(0.0001, 0.1)]  # Learning rate bounds

# Initialize Bayesian Optimizer
optimizer = BayesianOptimization(
    f=objective,
    X_init=np.array([[0.01]]),
    Y_init=np.array([objective(0.01)]),
    bounds=bounds,
    ac_type='ucb'  # Acquisition function type
)

# Run optimization
optimizer.maximize(iterations=20)

# Get best hyperparameters
best_params = optimizer.best_params()
print(f"Best learning rate: {best_params}")
```

## Optimization Problem Example

```python
# 1D function optimization
def sphere_function(x):
    return -((x - 2)**2 + 3)  # Max at x=2, value=3

# 2D Rosenbrock function
def rosenbrock(x, y):
    return -((1 - x)**2 + 100 * (y - x**2)**2)

# Setup optimization
optimizer = BayesianOptimization(
    f=sphere_function,
    X_init=np.array([[-1.0]]),
    Y_init=np.array([sphere_function(-1.0)]),
    bounds=[(-5, 5)],
    ac_type='ei'
)

# Run for 30 iterations
for _ in range(30):
    optimizer.maximize(iterations=1)
```

## Comparison with Other Methods

| Method | Sample Efficient | Handles Noise | Interpretability | Speed |
|--------|-----------------|--------------|-----------------|-------|
| **Grid Search** | Poor | Yes | High | Slow |
| **Random Search** | Poor | Yes | High | Slow |
| **Bayesian Opt** | Excellent | Yes | Medium | Fast |
| **Hyperband** | Good | Yes | Medium | Very Fast |
| **Genetic Alg** | Poor | Yes | Low | Medium |

## Practical Tips

1. **Normalize inputs**: Scale hyperparameters to [0, 1] for better GP performance
2. **Use appropriate kernels**: RBF works for most cases, Matérn for non-smooth functions
3. **Acquisition function tuning**: UCB explores more (good for exploration), EI exploits more
4. **Parallel evaluation**: Can evaluate multiple points in parallel to speed up optimization
5. **Warm start**: Initialize with good guess to get better results faster

## References

- Rasmussen, C. E., & Williams, C. K. (2006). Gaussian Processes for Machine Learning
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms
