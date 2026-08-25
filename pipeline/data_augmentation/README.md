# Pipeline - Data Augmentation

This directory contains image data augmentation techniques from the Holberton machine learning curriculum. Data augmentation is used to artificially increase the size of a training dataset by applying random transformations to images, improving model generalization.

## Learning Objectives

- Apply common image transformations (flip, rotate, crop)
- Adjust image properties (brightness, contrast, hue)
- Use TensorFlow for efficient augmentation
- Implement Principal Component Analysis (PCA) for augmentation
- Understand how augmentation improves model robustness

## Files

- `0-flip.py`: Defines `flip_image(image)` — randomly flip image horizontally
- `1-crop.py`: Defines `crop_image(image, size)` — randomly crop image to specified size
- `2-rotate.py`: Defines `rotate_image(image, angle)` — rotate image by specified angle
- `3-contrast.py`: Defines `adjust_contrast(images, lower, upper)` — adjust image contrast
- `4-brightness.py`: Defines `adjust_brightness(images, delta)` — adjust image brightness
- `5-hue.py`: Defines `adjust_hue(images, delta)` — adjust image hue (color shift)
- `100-pca.py`: Defines `change_brightness(images, hue_shift)` — PCA-based augmentation using Albumentations

## Requirements

- Python 3.x
- TensorFlow/Keras 2.x or higher
- NumPy
- Albumentations (for PCA-based augmentation)
- `pycodestyle` style compliance where required

## Key Concepts

**Geometric Transformations**: Flip, rotate, and crop operations that change image geometry while preserving content

**Color Transformations**: Brightness, contrast, and hue adjustments that modify pixel values

**PCA-based Augmentation**: Uses principal component analysis to add correlated noise that mimics natural image variations

**Data Pipeline**: Augmentation is typically applied on-the-fly during training to save memory

## Usage

```python
import tensorflow as tf
from data_augmentation_module import flip_image, adjust_brightness

# Create sample image
image = tf.random.normal([256, 256, 3])

# Apply augmentations
flipped = flip_image(image)
brightened = adjust_brightness(tf.expand_dims(flipped, 0), 0.5)

# Use in training pipeline
train_dataset = train_images.map(lambda x: augment_image(x))
train_dataset = train_dataset.batch(32).shuffle(1000)
```

## Best Practices

- Apply augmentation only to training data, not validation/test data
- Use moderate augmentation strength to avoid degrading training
- Combine multiple augmentations for better regularization
- Monitor validation accuracy to ensure augmentation helps rather than hurts

EltunLTN