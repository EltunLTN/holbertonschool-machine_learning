# Transfer Learning

This directory contains transfer learning implementations from the Holberton machine learning curriculum. Transfer learning is a powerful technique that leverages pre-trained models to solve new tasks efficiently, especially when the new task has limited training data.

## Learning Objectives

- Understand transfer learning principles and when to use it
- Load and utilize pre-trained models (ResNet, VGG, Inception, MobileNet)
- Implement feature extraction from pre-trained networks
- Fine-tune pre-trained models on new tasks
- Adapt models for different input sizes and datasets
- Use transfer learning for computer vision tasks
- Combine transfer learning with other techniques (data augmentation, regularization)
- Evaluate transfer learning performance

## Files

- `0-transfer.py`: Defines `Transfer(data, labels, learning_rate, epochs)` — complete transfer learning implementation

## Requirements

- Python 3.x
- TensorFlow/Keras 2.x or higher
- NumPy
- scikit-learn (for metrics)
- `pycodestyle` style compliance where required

## Key Concepts

### What is Transfer Learning?

Transfer learning is the practice of using a model trained on one task to improve learning on a new, related task. Instead of training from scratch, we leverage pre-trained weights that have already learned useful features.

### Types of Transfer Learning

1. **Feature Extraction (Fine-tuning)**
   - Freeze early layers, train last layers on new data
   - Keeps learned low-level features, adapts high-level features
   - Use when new dataset is small but similar to original

2. **Fine-tuning**
   - Start with pre-trained weights
   - Train all layers with very low learning rate
   - Use when new dataset is similar but not identical

3. **Domain Adaptation**
   - Adapt model from source domain to target domain
   - Handle distribution shift between domains
   - Use techniques like batch normalization adjustment

4. **Multi-task Learning**
   - Train on multiple related tasks simultaneously
   - Share representations between tasks
   - Improves generalization

### Pre-trained Models

Common pre-trained models for image classification (trained on ImageNet):

| Model | Accuracy | Speed | Size | Use Case |
|-------|----------|-------|------|----------|
| **VGG16** | 71.3% | Slow | 138 MB | Feature extraction, simple tasks |
| **ResNet50** | 74.9% | Medium | 102 MB | Balanced performance, deeper understanding |
| **InceptionV3** | 77.9% | Fast | 92 MB | High accuracy, efficient |
| **MobileNetV2** | 71.3% | Very Fast | 12 MB | Mobile/edge deployment |
| **EfficientNet** | 77.1% | Fast | 38 MB | Efficient scaling |
| **DenseNet** | 76.9% | Medium | 110 MB | Dense connections |

### Advantages of Transfer Learning

1. **Faster Training**: Starts with pre-trained weights instead of random initialization
2. **Better Performance**: Especially with small datasets (< 1000 samples)
3. **Fewer Samples Needed**: Reduces data requirements significantly
4. **Lower Computational Cost**: Can use smaller models with good performance
5. **Better Generalization**: Benefits from features learned on ImageNet

### When to Use Transfer Learning

- ✅ New task is related to pre-training task (e.g., ImageNet → your images)
- ✅ New dataset is small (< 10K samples)
- ✅ Computational resources are limited
- ✅ Time-to-market is critical
- ❌ New task is very different from pre-training task
- ❌ New dataset is very large (> 1M samples) and different

## Typical Transfer Learning Workflow

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Step 1: Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# Step 2: Freeze base model layers
base_model.trainable = False

# Step 3: Add custom layers for new task
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# Step 4: Compile with low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train on new task
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Step 6: Fine-tune (optional)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False  # Freeze all but last 20 layers

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

## Feature Extraction Strategy

Use when you have limited data (< 1000 samples):

```python
# Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False

# Extract features
def extract_features(X):
    return base_model.predict(X, verbose=0)

# Extract for all data
X_train_features = extract_features(X_train)
X_val_features = extract_features(X_val)

# Train simple classifier on extracted features
classifier = Dense(num_classes, activation='softmax')
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train_features, y_train, epochs=20, validation_data=(X_val_features, y_val))
```

## Fine-tuning Strategy

Use when you have moderate data (1K - 100K samples):

```python
# Load base model
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom head
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Stage 1: Freeze base model
base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Stage 2: Unfreeze last layers and fine-tune
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

## Data Augmentation with Transfer Learning

Augmentation is especially important when using transfer learning with limited data:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create augmentation pipeline
train_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Train with augmentation
train_generator = train_augmentation.flow(X_train, y_train, batch_size=32)
model.fit(train_generator, epochs=20, validation_data=(X_val, y_val))
```

## Common Issues and Solutions

### Issue: Training Accuracy High but Validation Low
```python
# Solutions:
# 1. Increase regularization
model.add(Dropout(0.5))

# 2. Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(..., callbacks=[early_stop])

# 3. Use more augmentation
augmentation_strength = 0.3  # Increase this
```

### Issue: Very Slow Learning
```python
# Solutions:
# 1. Increase learning rate (if stable)
optimizer = Adam(learning_rate=0.001)  # Increase from 0.0001

# 2. Unfreeze more layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

# 3. Use better initialization
# (pre-trained weights are already good)
```

### Issue: Model Not Generalizing
```python
# Solutions:
# 1. Use smaller/simpler custom architecture
# 2. Add more data augmentation
# 3. Reduce model capacity
# 4. Increase regularization strength
```

## Performance Benchmarks

Training time comparison (1000 images, ResNet50):

| Approach | Time | Accuracy |
|----------|------|----------|
| Training from scratch | 2 hours | 75% |
| Feature extraction | 5 minutes | 82% |
| Fine-tuning (frozen base) | 30 minutes | 85% |
| Fine-tuning (unfrozen) | 1 hour | 88% |

## Domain-Specific Transfer Learning

### Medical Imaging
- Use models pre-trained on medical datasets
- Consider domain shift (e.g., CT vs. X-ray)
- Often use ChexPert or other medical models

### Remote Sensing
- Satellite image classification
- Pre-train on large remote sensing datasets
- Adapt for specific applications (agriculture, urban planning)

### Object Detection
- Use pre-trained detectors (YOLO, Faster R-CNN)
- Fine-tune on custom object classes
- Transfer bounding box regression layers

## Advanced Techniques

### 1. Knowledge Distillation
Transfer knowledge from large model to small model:

```python
# Teacher model (large, accurate)
teacher = large_pretrained_model

# Student model (small, efficient)
student = build_small_model()

# Distillation loss balances task loss and KL divergence
distillation_loss = categorical_crossentropy(y, student_pred) + \
                    KL_divergence(teacher_pred_soft, student_pred_soft)
```

### 2. Adapter Modules
Add small trainable modules to pre-trained networks:

```python
# Insert adapters between frozen layers
x = frozen_layer(x)
x = adapter_layer(x)  # Small trainable module
x = next_frozen_layer(x)
```

### 3. Layer-wise Adaptation
Different learning rates for different layers:

```python
# Lower learning rate for earlier layers
model.get_layer('base_model').set_learning_rate(0.00001)
model.get_layer('dense_1').set_learning_rate(0.0001)
model.get_layer('output').set_learning_rate(0.001)
```

## Evaluation and Testing

```python
# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Analyze results
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predicted_classes))
print(confusion_matrix(y_test, predicted_classes))
```

## Best Practices

1. **Start with feature extraction**: Quick baseline with minimal training
2. **Monitor validation metrics**: Early stopping to prevent overfitting
3. **Use appropriate learning rates**: Lower for pre-trained parts, higher for new parts
4. **Data augmentation**: Critical when data is limited
5. **Gradual unfreezing**: Unfreeze layers progressively during fine-tuning
6. **Preserve pre-trained weights**: Don't corrupt learned features early
7. **Evaluate thoroughly**: Test on representative data

## References

- Yosinski, J., Clune, J., Bengio, Y., & Liphardt, H. (2014). How transferable are features in deep neural networks?
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning
- Long, M., Cao, Y., Wang, J., & Jordan, M. I. (2013). Learning transferable features with deep adaptation networks
