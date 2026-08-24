# Neural Style Transfer

This directory contains implementation of neural style transfer from the Holberton machine learning curriculum. Neural style transfer is a technique that applies the artistic style of one image to the content of another using deep convolutional neural networks.

## Learning Objectives

- Understand feature extraction with pre-trained CNN models (VGG19)
- Compute style and content losses for neural style transfer
- Implement gradient descent optimization for style transfer
- Apply artistic styles from reference images to content images
- Use TensorFlow and Keras for neural style transfer

## Files

- `0-neural_style.py`: Defines `NST(style_image, content_image, alpha=1e-2, beta=1)` — main Neural Style Transfer class

## Requirements

- Python 3.x
- TensorFlow/Keras 2.x or higher
- NumPy
- OpenCV or PIL (for image loading)
- `pycodestyle` style compliance where required

## Key Concepts

**Content Loss**: Measures how well the generated image preserves the content of the content image (typically calculated from higher layers)

**Style Loss**: Measures how well the generated image matches the artistic style of the style image using Gram matrices (typically calculated from multiple layers)

**Gram Matrix**: Represents the style/texture of an image by computing correlations between feature maps

**Optimization**: Uses gradient descent (usually Adam or L-BFGS) to minimize combined style and content losses

**Alpha/Beta Parameters**: Control the trade-off between preserving content (alpha) and matching style (beta)

## Usage

```python
from tensorflow import keras as K
from neural_style_transfer import NST

# Load images
style_image = load_image('style.jpg')
content_image = load_image('content.jpg')

# Create NST instance
nst = NST(style_image, content_image, alpha=1e-2, beta=1)

# Generate stylized image
output = nst.generate(iterations=1000)
```

## Process

1. Load pre-trained VGG19 model (trained on ImageNet)
2. Extract features from style and content images
3. Initialize generated image (typically from content image)
4. Iteratively optimize generated image to:
   - Minimize content loss (preserve content structure)
   - Minimize style loss (match artistic style)
5. Return optimized image with style transferred to content

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style
