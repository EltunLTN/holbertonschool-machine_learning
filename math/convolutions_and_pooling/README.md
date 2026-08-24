# Math - Convolutions and Pooling

This directory contains mathematical implementations of convolution and pooling operations from the Holberton machine learning curriculum. These are fundamental operations in Convolutional Neural Networks (CNNs), implemented from scratch using NumPy.

## Learning Objectives

- Understand convolution operations (valid, same, and full)
- Implement convolutions with and without padding and stride
- Apply convolutions to grayscale and color images
- Implement pooling operations (max pooling)
- Work with single and multi-channel convolutions

## Files

- `0-convolve_grayscale_valid.py`: Defines `convolve_grayscale_valid(images, kernel)` — valid convolution on grayscale images
- `1-convolve_grayscale_same.py`: Defines `convolve_grayscale_same(images, kernel)` — "same" convolution with padding
- `2-convolve_grayscale_padding.py`: Defines `convolve_grayscale_padding(images, kernel, padding)` — convolution with custom padding
- `3-convolve_grayscale.py`: Defines `convolve_grayscale(images, kernel, padding, stride)` — general grayscale convolution with stride
- `4-convolve_channels.py`: Defines `convolve_channels(images, kernel, padding, stride)` — multi-channel convolution
- `5-convolve.py`: Defines `convolve(images, kernels, padding, stride)` — multi-kernel convolution for output feature maps
- `6-pool.py`: Defines `pool(images, kernel_shape, stride, mode)` — pooling operation (max or average)

## Requirements

- Python 3.x
- NumPy
- `pycodestyle` style compliance where required

## Key Concepts

**Valid Convolution**: No padding applied; output size = (input_size - kernel_size) / stride + 1

**Same Convolution**: Padding applied to maintain output size equal to (input_size - 1) / stride + 1

**Convolution with Stride**: Moves the kernel by `stride` pixels instead of 1

**Multi-Channel Convolution**: Each kernel has depth matching input channels; produces one output per kernel

**Pooling**: Downsamples feature maps using max or average pooling

## Usage

```python
import numpy as np
from convolution_module import convolve_grayscale_valid, pool

# Create sample image batch
images = np.random.randn(10, 28, 28)  # 10 images of 28x28
kernel = np.random.randn(3, 3)  # 3x3 kernel

# Apply convolution
output = convolve_grayscale_valid(images, kernel)
print(output.shape)  # (10, 26, 26)

# Apply max pooling
pooled = pool(output[:, np.newaxis, :, :], (2, 2), 2, 'max')
```
