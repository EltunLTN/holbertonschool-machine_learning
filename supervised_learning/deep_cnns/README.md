# Deep Convolutional Neural Networks

This directory contains implementations of deep CNN architectures from the Holberton machine learning curriculum. It focuses on building deep residual networks (ResNets) and understanding the components of modern deep learning architectures.

## Learning Objectives

- Understand the ResNet architecture and residual connections
- Build identity blocks and projection blocks for deep networks
- Implement skip connections to enable training of very deep networks
- Use ResNet50 for feature extraction and transfer learning

## Files

- `2-identity_block.py`: Defines `identity_block(A_prev, filters)` — builds an identity residual block
- `3-projection_block.py`: Defines `projection_block(A_prev, filters, s=2)` — builds a projection block with stride
- `4-resnet50.py`: Defines `ResNet50()` — builds the complete ResNet50 architecture

## Requirements

- Python 3.x
- TensorFlow/Keras 2.x or higher
- NumPy
- `pycodestyle` style compliance where required

## Key Concepts

**Identity Blocks**: Skip connections where the input dimension matches the output dimension, allowing gradients to flow directly through the network.

**Projection Blocks**: Skip connections with stride and convolution to adjust dimensions, used when changing spatial dimensions or filter counts.

**ResNet50**: A 50-layer deep residual network consisting of convolutional blocks, identity blocks, and projection blocks stacked together.

## Usage

```python
from tensorflow import keras as K
import numpy as np
from your_module import identity_block, projection_block, ResNet50

# Build ResNet50 model
model = ResNet50()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use for classification
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition (ResNet paper)
