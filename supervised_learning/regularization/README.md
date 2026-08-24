# Supervised Learning - Regularization

This directory contains implementations of regularization techniques from the Holberton machine learning curriculum. Regularization is crucial for preventing overfitting and improving model generalization on unseen data.

## Learning Objectives

- Understand L2 regularization (weight decay) and its mathematical formulation
- Implement L2 regularization in forward and backward propagation
- Apply dropout for stochastic regularization during training
- Combine multiple regularization strategies effectively
- Implement early stopping to prevent overfitting
- Compare regularization techniques and their effects on model performance

## Files

- `0-l2_reg_cost.py`: Defines `l2_reg_cost(cost, lambtha, weights, L, m)` — compute L2 regularization cost
- `1-l2_reg_gradient_descent.py`: Defines `l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L)` — L2 regularized gradient descent
- `2-l2_reg_cost.py`: Defines `l2_reg_cost(y, y_hat, lambtha)` — L2 cost for Keras models
- `3-l2_reg_create_layer.py`: Defines `l2_reg_create_layer(layer_units, layer_activation, lambtha)` — create Keras layer with L2 regularization
- `4-dropout_forward_prop.py`: Defines `dropout_forward_prop(X, weights, L, keep_prob)` — forward propagation with dropout
- `5-dropout_gradient_descent.py`: Defines `dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L)` — backprop with dropout
- `6-dropout_create_layer.py`: Defines `dropout_create_layer(prev_units, layer_units, activation, keep_prob)` — create Keras layer with dropout
- `7-early_stopping.py`: Defines `early_stopping(y_val, y_pred, patience, threshold)` — implement early stopping criterion

## Requirements

- Python 3.x
- NumPy
- TensorFlow/Keras 2.x or higher
- `pycodestyle` style compliance where required

## Key Concepts

**L2 Regularization (Ridge)**: Adds penalty term λ/2m * Σ||W||² to loss function
- Prevents large weights that may cause overfitting
- Shrinks all weights proportionally towards zero
- Controlled by regularization parameter λ (lambda)

**Dropout**: Randomly deactivates neurons during training with probability (1 - keep_prob)
- Prevents co-adaptation of neurons
- Forces network to learn redundant representations
- Applied only during training, not during inference

**Early Stopping**: Stops training when validation performance stops improving
- Monitors validation metric (e.g., loss or accuracy)
- Stops if no improvement for `patience` consecutive epochs
- Prevents training beyond optimal generalization point

**Keep Probability**: Fraction of neurons retained during dropout (typically 0.5-0.8)

## Usage

```python
import numpy as np
from regularization_module import l2_reg_cost, dropout_forward_prop, early_stopping

# L2 Regularization
cost_with_l2 = l2_reg_cost(base_cost, lambtha=0.01, weights=W, L=3, m=m)

# Dropout
X_dropped, mask = dropout_forward_prop(X, weights, L=3, keep_prob=0.8)

# Early Stopping
stop = early_stopping(y_val, y_pred, patience=10, threshold=0.001)

# Keras with Regularization and Dropout
from regularization_module import l2_reg_create_layer, dropout_create_layer

model = Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

## Best Practices

1. **Tuning λ**: Start with small values (0.001-0.01) and increase if needed
2. **Dropout Placement**: Apply after activation functions, typically 10-50% drop rate
3. **Validation Monitoring**: Always monitor validation metrics to assess overfitting
4. **Patience Parameter**: Set patience based on dataset size (typical: 5-20 epochs)
5. **Combine Techniques**: Use multiple regularization methods together for best results

## Regularization Trade-offs

- **Under-regularization**: Model overfits, high training accuracy but poor test performance
- **Over-regularization**: Model underfits, poor performance on both training and test data
- **Balanced regularization**: Good generalization with acceptable bias-variance trade-off
