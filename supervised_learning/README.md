# Supervised Learning

This directory contains supervised learning implementations and algorithms from the Holberton machine learning curriculum. It covers a wide range of machine learning techniques for classification and regression tasks, from fundamentals to advanced deep learning architectures.

## Learning Objectives

- Implement foundational machine learning algorithms from scratch
- Build and train neural networks with various architectures
- Master optimization techniques and regularization strategies
- Understand and apply convolutional neural networks for image processing
- Implement specialized architectures (ResNets, transfer learning, object detection)
- Apply deep learning to style transfer and other advanced tasks
- Develop and evaluate classification models
- Handle hyperparameter optimization and model selection

## Subdirectories

### 1. **classification/** - Binary and Multi-class Classification
   - Logistic regression from scratch
   - Multi-class classification techniques
   - Binary classification with various models
   - Model evaluation metrics and techniques

### 2. **neural_style_transfer/** - Artistic Image Transformation
   - Neural style transfer implementation
   - VGG19-based feature extraction
   - Gram matrix computation for style representation
   - Content and style loss optimization

### 3. **object_detection/** - Real-time Object Detection (YOLO)
   - YOLO architecture implementation
   - Anchor boxes and multi-scale predictions
   - Non-maximum suppression for duplicate removal
   - Intersection over Union (IoU) computation
   - Image preprocessing and prediction

### 4. **regularization/** - Preventing Overfitting
   - L2 regularization (weight decay)
   - Dropout for stochastic regularization
   - Early stopping convergence criteria
   - Combining multiple regularization techniques

### 5. **optimization/** - Training and Parameter Optimization
   - Gradient descent variants (SGD, Adam, RMSprop)
   - Learning rate scheduling
   - Momentum and adaptive methods
   - Batch normalization

### 6. **transfer_learning/** - Leveraging Pre-trained Models
   - Using pre-trained models (ResNet, VGG, Inception)
   - Fine-tuning for specific tasks
   - Feature extraction without training
   - Domain adaptation

### 7. **keras/** - High-level Deep Learning
   - Building models with Keras API
   - Custom layers and loss functions
   - Model training and evaluation
   - Callbacks and logging

### 8. **cnn/** - Convolutional Neural Networks
   - Building blocks of CNNs
   - Convolutional and pooling layers
   - Image processing with CNNs
   - LeNet, AlexNet, VGG architectures

### 9. **deep_cnns/** - Deep Residual Networks
   - Identity and projection blocks
   - ResNet architecture
   - Skip connections for very deep networks
   - He initialization

### 10. **error_analysis/** - Model Evaluation and Debugging
   - Confusion matrix computation
   - Sensitivity, precision, specificity, F1 score
   - Classification metrics analysis
   - Error distribution analysis

### 11. **decision_tree/** - Tree-based Models
   - Building decision trees from scratch
   - Information gain and splitting criteria
   - Random forests for ensemble learning
   - Isolation forests for anomaly detection

## Requirements

- Python 3.x
- NumPy (numerical computation)
- TensorFlow/Keras 2.x or higher (deep learning)
- scikit-learn (classical ML algorithms)
- Matplotlib (visualization)
- OpenCV (image processing)
- `pycodestyle` style compliance where required

## Typical Supervised Learning Workflow

```python
import numpy as np
from supervised_learning.classification import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Step 1: Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 2: Create and train model
model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

## Algorithm Categories

### Linear Models
- **Logistic Regression**: Binary and multi-class classification
- **Linear Regression**: Continuous value prediction
- Fast, interpretable, good baseline

### Neural Networks
- **Fully Connected Networks**: General-purpose learning
- **Convolutional Networks**: Image processing
- **Recurrent Networks**: Sequential data
- Flexible, powerful, requires careful tuning

### Tree-based Models
- **Decision Trees**: Hierarchical splitting
- **Random Forests**: Ensemble of trees
- **Gradient Boosting**: Sequential tree learning
- Interpretable, handles non-linearity, prone to overfitting

### Ensemble Methods
- **Bagging**: Bootstrap aggregating (Random Forests)
- **Boosting**: Sequential error correction (AdaBoost, Gradient Boosting)
- **Stacking**: Combining multiple models
- Often achieve best performance

## Model Complexity vs. Data Size

```
Small Dataset          Medium Dataset        Large Dataset
(< 1000 samples)       (1K - 100K)           (> 100K samples)
    ↓                        ↓                       ↓
- Linear models        - Medium NNs            - Deep NNs
- Regularization       - Some regularization   - Minimal regularization
- Simple features      - Feature engineering   - Automatic FE (deep)
- Cross-validation     - Validation set        - Large test set
```

## Training Best Practices

### Data Preparation
1. **Split Data**: 70% train, 15% validation, 15% test (or 80/20)
2. **Stratification**: Maintain class distribution in splits
3. **Feature Scaling**: Normalize/standardize features
4. **Handle Imbalance**: Use appropriate class weights

### Model Training
1. **Start Simple**: Linear model baseline before complex models
2. **Monitor**: Track training and validation loss
3. **Prevent Overfitting**: Use regularization and early stopping
4. **Tune Hyperparameters**: Use grid search or random search
5. **Ensemble**: Combine multiple models for better performance

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'dropout_rate': [0.2, 0.5, 0.7]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

## Common Pitfalls

1. **Data Leakage**: Fitting on test data or preprocessing before splitting
2. **Imbalanced Data**: Not accounting for class imbalance
3. **Poor Validation**: Not using proper train/test/validation splits
4. **Overfitting**: Complex model on small dataset
5. **Wrong Metrics**: Using accuracy for imbalanced data instead of F1/AUC
6. **Hyperparameter Tuning on Test**: Choosing hyperparameters based on test performance
7. **Ignoring Baseline**: Not comparing against simple baselines

## Model Selection Guide

| Problem | Dataset Size | Speed | Interpretability | Best Algorithm |
|---------|-------------|-------|------------------|----------------|
| Binary Classification | Small | Fast | High | Logistic Regression |
| Binary Classification | Large | Fast | High | SVM |
| Multi-class | Small | Medium | High | Decision Tree |
| Multi-class | Large | Slow | Low | Deep NN |
| Regression | Small | Fast | High | Linear Regression |
| Regression | Large | Slow | Low | Neural Network |
| Image Classification | Large | Slow | Low | CNN (ResNet) |
| Object Detection | Large | Medium | Low | YOLO/Faster R-CNN |

## Feature Engineering Tips

1. **Domain Knowledge**: Use subject matter expertise
2. **Statistical Features**: Mean, variance, quantiles
3. **Interaction Terms**: Products of features
4. **Polynomial Features**: x², x³ for non-linearity
5. **Binning**: Convert continuous to categorical
6. **Encoding**: Handle categorical variables properly
7. **Scaling**: Normalize for distance-based algorithms

## Debugging Model Performance

```python
# If training accuracy is low:
# - Increase model complexity
# - Better features
# - More training iterations
# - Lower regularization

# If validation accuracy is low:
# - Reduce model complexity
# - More regularization
# - More training data
# - Better feature engineering

# If test accuracy is much lower than validation:
# - Test set distribution different from training
# - Data leakage possible
# - Hyperparameters tuned on test set
```

## Model Evaluation Metrics

### Classification
- **Accuracy**: (TP + TN) / Total (overall correctness)
- **Precision**: TP / (TP + FP) (false positive rate)
- **Recall**: TP / (TP + FN) (false negative rate)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **AUC-ROC**: Area under ROC curve (probabilistic ranking)

### Regression
- **MAE**: Mean Absolute Error (average error magnitude)
- **MSE**: Mean Squared Error (penalizes large errors)
- **RMSE**: Root Mean Squared Error (interpretable scale)
- **R²**: Coefficient of determination (variance explained)

## Training Visualization

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

## Performance Optimization

- **GPU Acceleration**: Use CUDA/cuDNN for neural networks
- **Batch Processing**: Process multiple samples together
- **Mixed Precision**: Use float16 where appropriate
- **Model Quantization**: Reduce precision for faster inference
- **Pruning**: Remove unimportant weights
- **Knowledge Distillation**: Teach smaller model from larger one

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Ng, A. (2017). Machine Learning Yearning
