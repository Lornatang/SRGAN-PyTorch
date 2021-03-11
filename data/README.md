# Dataset Information

## *This experiment uses the `test` data set in ImageNet2012 (delete image size less 96 pixel)*

## Creating pairs of superset datasets

### Method 1: Blind Super-Resolution Kernel Estimation using an Internal-GAN

#### Abstract

Super resolution (SR) methods typically assume that the low-resolution (LR) image
was downscaled from the unknown high-resolution (HR) image by a fixed ‘ideal’
downscaling kernel (e.g. Bicubic downscaling). However, this is rarely the case
in real LR images, in contrast to synthetically generated SR datasets. When
the assumed downscaling kernel deviates from the true one, the performance of
SR methods significantly deteriorates. This gave rise to Blind-SR – namely, SR
when the downscaling kernel (“SR-kernel”) is unknown. It was further shown
that the true SR-kernel is the one that maximizes the recurrence of patches across
scales of the LR image. In this paper we show how this powerful cross-scale
recurrence property can be realized using Deep Internal Learning. We introduce
“KernelGAN”, an image-specific Internal-GAN, which trains solely on the LR
test image at test time, and learns its internal distribution of patches. Its Generator
is trained to produce a downscaled version of the LR test image, such that its
Discriminator cannot distinguish between the patch distribution of the downscaled
image, and the patch distribution of the original LR image. The Generator, once
trained, constitutes the downscaling operation with the correct image-specific
SR-kernel. KernelGAN is fully unsupervised, requires no training data other than
the input image itself, and leads to state-of-the-art results in Blind-SR when plugged
into existing SR algorithms. 

#### Usage

```text
# Clone here
$ git clone https://github.com/Lornatang/KernelGAN
cd KernelGAN

# Put the data sets you need to process in this place
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --4x --input-dir <your-images-dir>

# Using the image distribution algorithm learned in kernelgan to construct pairwise hyperspectral data
$ python create_dataset_for_kernelGAN.py --input-dir <Low-resolution-folder> --target-dir <High-resolution-folder> 
```

### Method 2: Low resolution image generation using bicubic simple down sampling

#### Usage

```text
$ python create_dataset_for_bicubic.py --input-dir <Low-resolution-folder> --target-dir <High-resolution-folder> 
```

### Method 3: In LR region, the HR area is cut out (the difficulty coefficient is very high, and the error is also very large)

#### Usage

```text
$ python crop_dataset.py
```

## Dataset Download

This project uses the voc2012 dataset.
[baiduclouddisk](https://pan.baidu.com/s/1CbBm7_xDkQEI17cHQS6Fbg) access: `9ape`
