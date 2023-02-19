# Precure StyleGAN

Yet another StyleGAN 1.0 implementation with Chainer

We tried out to generate facial images of a specific Precure (Japanese Anime) character.

This project is finished and [will be continued here for better quality with StyleGAN 2.0 ADA](https://github.com/curegit/precure-stylegan-ada).

## Overview

StyleGAN is a generative adversarial network introduced by NVIDIA researchers.
Like PGGAN, its output image resolutions grow progressively during training.
This implementation supports generation ranging from 4x4 px (stage 1) to 1024x1024 px (stage 9) images.

Most of the implementation follows the original paper, but we have added some enhancements.
For example, we implemented an alternative least-squares objective introduced in LSGAN.
We trained the models with facial images of Cure Beauty (Smile Pretty Cure!, 2012) and other common datasets.

## Requirements

- Python >= 3.6
- Chainer >= 7.0
- Pillow >= 7.1
- Numpy
- H5py
- Matplotlib

### Optional

- Cupy
- Pydot (Graphviz)

## Script Synopses

- `train.py` trains the models of StyleGAN.
  Use `-h` option for more details.
- `generate.py` generates images from a trained model.
  Use `-h` option for more details.
- `animate.py` makes the animation of the analogy from a trained model.
  Use `-h` option for more details.
- `visualize.py` draws an example of a computation graph for debugging (Pydot and Graphviz are required).
  It takes no command-line arguments.
- `check.py` analyzes the Chainer environment.
  It takes no command-line arguments.

## Results

### Cure Beauty (Curated)

![Cure Beauty](examples/beauty.png)

## Other Dataset Results

### MNIST (Uncurated)

![MNIST](examples/mnist.png)

### CIFAR-10 (Uncurated)

![CIFAR-10](examples/cifar-10.png)

### Anime Face (Uncurated)

![Anime Face](examples/anime.png)

## Bibliography

### Papers

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
- [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)

### Implementation used as reference

- [Chainer implementation of Style-based Generator](https://github.com/pfnet-research/chainer-stylegan)
- [Chainer-StyleBasedGAN](https://github.com/RUTILEA/Chainer-StyleBasedGAN)

### Datasets for testing

- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Anime-Face-Dataset](https://github.com/Mckinsey666/Anime-Face-Dataset)

## License

[CC BY-NC 4.0](LICENSE)
