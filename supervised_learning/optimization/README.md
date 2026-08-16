# Optimization in Machine Learning

This directory contains implementations of various optimization techniques and methods used to improve machine learning model training and performance.

## Overview

Optimization is a fundamental aspect of machine learning that focuses on finding the best parameters for a model. This directory covers:

- **Data Preprocessing**: Normalization and shuffling techniques
- **Training Strategies**: Mini-batch training and mini-batch gradient descent
- **Optimization Algorithms**: Momentum, RMSProp, and Adam optimizers
- **Learning Rate Techniques**: Learning rate decay schedules
- **Normalization Methods**: Batch normalization for improved training stability

## Topics Covered

### 1. Data Normalization
- **File**: `0-norm_constants.py`, `1-normalize.py`
- **Description**: Techniques to normalize data by scaling features to have zero mean and unit variance, which helps accelerate convergence during training.

### 2. Data Shuffling
- **File**: `2-shuffle_data.py`
- **Description**: Randomizing the order of training samples to prevent the model from learning patterns based on data order and to improve generalization.

### 3. Mini-Batch Training
- **File**: `3-mini_batch.py`
- **Description**: Dividing the dataset into smaller batches for training, which balances the trade-off between computational efficiency and gradient accuracy.

### 4. Moving Average
- **File**: `4-moving_average.py`
- **Description**: Computing moving averages to smooth noisy signals and estimate parameters more robustly.

### 5. Momentum Optimization
- **Files**: `5-momentum.py`, `6-momentum.py`
- **Description**: The Momentum optimizer accelerates gradient descent by building up velocity in directions of consistent gradient, helping overcome local minima and accelerating training.

### 6. RMSProp Optimizer
- **Files**: `7-RMSProp.py`, `8-RMSProp.py`
- **Description**: Root Mean Square Propagation (RMSProp) is an adaptive learning rate method that divides the learning rate by an exponentially decaying average of squared gradients, preventing exploding or vanishing gradients.

### 7. Adam Optimizer
- **Files**: `9-Adam.py`, `10-Adam.py`
- **Description**: Adaptive Moment Estimation (Adam) combines the benefits of momentum and RMSProp by maintaining both first and second moment estimates of gradients, resulting in stable and efficient training.

### 8. Learning Rate Decay
- **Files**: `11-learning_rate_decay.py`, `12-learning_rate_decay.py`
- **Description**: Techniques to reduce the learning rate over time during training, allowing for more fine-tuned convergence as the optimization progresses.

### 9. Batch Normalization
- **Files**: `13-batch_norm.py`, `14-batch_norm.py`
- **Description**: Normalizing the activations of each layer in a neural network to accelerate training, reduce internal covariate shift, and allow for higher learning rates.

## Learning Progression

1. Start with **data normalization** to prepare your data
2. Learn about **shuffling** to improve model generalization
3. Implement **mini-batch** training strategies
4. Master basic **momentum** optimization
5. Explore advanced optimizers: **RMSProp** and **Adam**
6. Apply **learning rate decay** for better convergence
7. Implement **batch normalization** for deeper networks

## Key Concepts

### Gradient Descent Variants
- **Batch Gradient Descent**: Uses entire dataset
- **Stochastic Gradient Descent (SGD)**: Uses one sample at a time
- **Mini-Batch Gradient Descent**: Uses small batches of data

### Optimizer Characteristics

| Optimizer | Advantages | Use Case |
|-----------|-----------|----------|
| Momentum | Accelerates convergence, escapes local minima | General purpose |
| RMSProp | Adaptive learning rates, handles sparse data | Recurrent networks |
| Adam | Combines momentum and adaptive rates, robust | Deep learning, general purpose |
| Batch Norm | Faster training, allows higher learning rates | Deep neural networks |

## Mathematical Background

### Momentum
```
v = β*v + (1-β)*∇J(θ)
θ = θ - α*v
```

### RMSProp
```
S = β*S + (1-β)*∇J(θ)²
θ = θ - α*∇J(θ)/√(S + ε)
```

### Adam
```
m = β₁*m + (1-β₁)*∇J(θ)
v = β₂*v + (1-β₂)*∇J(θ)²
θ = θ - α*m/(√v + ε)
```

## Implementation Tips

1. **Always normalize your data** before training
2. **Shuffle training data** in each epoch
3. **Start with Adam** optimizer - it works well out of the box
4. **Use mini-batches** of size 32-256 for most problems
5. **Apply batch normalization** in deep networks
6. **Monitor learning rate decay** to ensure convergence
7. **Experiment with hyperparameters** like β, β₁, β₂, and learning rate

## Usage Example

```python
# Normalize data
X_norm = normalize(X)

# Shuffle data
X_shuffled, Y_shuffled = shuffle_data(X_norm, Y)

# Create mini-batches
mini_batches = create_mini_batches(X_shuffled, Y_shuffled, batch_size=32)

# Initialize optimizer (e.g., Adam)
optimizer = Adam(alpha=0.001)

# Training loop
for epoch in range(num_epochs):
    for mini_batch in mini_batches:
        # Forward pass
        predictions = model.forward(mini_batch)
        
        # Compute loss and gradients
        loss, gradients = model.backward(predictions, labels)
        
        # Update parameters using optimizer
        optimizer.update(model.parameters, gradients)
```

## References

- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization
- Tieleman, T., & Hinton, G. (2012). Lecture 6.5—RMSprop
- Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/k^2)
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training

## Notes

- Different optimizers perform differently on different problems
- Hyperparameter tuning is essential for optimal performance
- Always validate on a separate test set
- Consider using adaptive learning rate schedules for better convergence