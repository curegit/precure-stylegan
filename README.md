# Precure StyleGAN

Yet another StyleGAN implementation we tried out the generation of Precure (Japanese anime) images

## Overview

StyleGAN is a generative adversarial network introduced by NVIDIA researchers.
Its output image resolutions grow progressively during training like PGGAN.
This implementation supports 4x4 px (stage 1) to 1024x1024 px (stage 9) images.
Most of the implementation follows the original paper, but we installed some enhancements.
For instance, we implemented an alternate least-squares objective introduced in LSGAN.

## Requirements

- Python >= 3.6
- Chainer >= 7.0
- Pillow >= 7.1
- Numpy
- H5py

### Optional

- Cupy
- Pydot (Graphviz)

## Script Synopses

- `train.py`

  train StyleGAN

- `generate.py`

  generate images of trained model

- `animate.py`

  make analogy animation

- `visualize.py`

  draw an example of computation graph (debug) (Pydot and Graphviz is required)

- `check.py`

  analyze chainer environment

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

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
- [Chainer implementation of Style-based Generator](https://github.com/pfnet-research/chainer-stylegan)
- [Chainer-StyleBasedGAN](https://github.com/RUTILEA/Chainer-StyleBasedGAN)
- [Anime-Face-Dataset](https://github.com/Mckinsey666/Anime-Face-Dataset)

## License

[CC BY-NC 4.0](LICENSE)
