# SRGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

### Table of contents
1. [About Super Resolution Generative Adversarial Networks](#about-super-resolution-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights)
    * [Download dataset](#download-dataset)
4. [Test](#test)
    * [Basic test](#basic-test)
    * [Test benchmark](#test-benchmark)
    * [Test image](#test-image)
4. [Train](#train-eg-div2k)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Super Resolution Generative Adversarial Networks

If you're new to SRGAN, here's an abstract straight from the paper:

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and 
deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover 
the finer texture details when we super-resolve at large upscaling factors? The behavior of 
optimization-based super-resolution methods is principally driven by the choice of the objective function. 
Recent work has largely focused on minimizing the mean squared reconstruction error. 
The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and 
are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. 
In this paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). 
To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. 
To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. 
The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained 
to differentiate between the super-resolved images and original photo-realistic images. In addition, 
we use a content loss motivated by perceptual similarity instead of similarity in pixel space. 
Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. 
An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. 
The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained 
with any state-of-the-art method.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. 
It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is 
a discriminant network that discriminates whether an image is real. The input is x, x is a picture, 
and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, 
and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/SRGAN-PyTorch.git
$ cd SRGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights

```bash
$ cd weights/
$ bash download_weights.sh
```

#### Download dataset

```bash
$ cd data/
$ bash download_dataset.sh
```

### Test

Using pre training model to generate pictures.

#### Basic test

```text
usage: test.py [-h] [--dataroot DATAROOT] [-j N] [--image-size IMAGE_SIZE]
               [--scale-factor SCALE_FACTOR] [--cuda] [--weights WEIGHTS]
               [--outf OUTF] [--manualSeed MANUALSEED]

PyTorch Super Resolution GAN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to dataset. (default:`./data/Set5`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:96)
  --scale-factor SCALE_FACTOR
                        Low to high resolution scaling factor. (default:4).
  --cuda                Enables cuda
  --weights WEIGHTS     Path to weights (default:`./weights/SRGAN_X4.pth`).
  --outf OUTF           folder to output images. (default:`./result`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:0)

# Example
$ python test.py --dataroot ./data/Set5 --cuda --weights ./weights/SRGAN_X4.pth
```

#### Test benchmark

```text
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N]
                         [--image-size IMAGE_SIZE] [--scale-factor {4,8,16}]
                         [--cuda] --weights WEIGHTS [--outf OUTF]
                         [--manualSeed MANUALSEED]

PyTorch Super Resolution GAN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/DIV2K`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:96)
  --scale-factor {4,8,16}
                        Low to high resolution scaling factor. (default:4).
  --cuda                Enables cuda
  --weights WEIGHTS     Path to weights. (default:`./weights/SRGAN_X4.pth`).
  --outf OUTF           folder to output images. (default:`./result`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:0)

# Example
$ python test_benchmark.py --dataroot ./data/DIV2K --cuda --weights ./weights/SRGAN_X4.pth
```

#### Test image

```text
usage: test_image.py [-h] [--file FILE] [--weights WEIGHTS] [--cuda]
                     [--image-size IMAGE_SIZE] [--scale-factor SCALE_FACTOR]

PyTorch Super Resolution GAN.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution image name.
                        (default:`./assets/baby.png`)
  --weights WEIGHTS     Generator model name. (default:`weights/SRGAN_X4.pth`)
  --cuda                Enables cuda
  --image-size IMAGE_SIZE
                        size of the data crop (squared assumed). (default:96)
  --scale-factor SCALE_FACTOR
                        Super resolution upscale factor

# Example
$ python test_image.py --file ./assets/baby.png --cuda --weights ./weights/SRGAN_X4.pth
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--epochs N]
                [--image-size IMAGE_SIZE] [-b N] [--lr LR]
                [--scale-factor {4,8,16}] [-p N] [--cuda] [--netG NETG]
                [--netD NETD] [--outf OUTF] [--manualSeed MANUALSEED]

PyTorch Super Resolution GAN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/DIV2K`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --epochs N            Number of total epochs to run. (default:2000)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:96)
  -b N, --batch-size N  mini-batch size (default: 16), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:0.0001)
  --scale-factor {4,8,16}
                        Low to high resolution scaling factor. (default:4).
  -p N, --print-freq N  Print frequency. (default:5)
  --cuda                Enables cuda
  --netG NETG           Path to netG (to continue training).
  --netD NETD           Path to netD (to continue training).
  --outf OUTF           folder to output images. (default:`./output`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:0)

# Example (e.g DIV2K)
$ python train.py --dataroot ./data/DIV2K --cuda --scale-factor 4
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python train.py --dataroot ./data/DIV2K \
                  --cuda                  \
                  --scale-factor 4        \
                  --netG ./weights/SRGAN_G_epoch_50.pth \
                  --netD ./weights/SRGAN_D_epoch_50.pth 
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
_Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi_ <br>

**Abstract** <br>
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and 
deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover 
the finer texture details when we super-resolve at large upscaling factors? The behavior of 
optimization-based super-resolution methods is principally driven by the choice of the objective function. 
Recent work has largely focused on minimizing the mean squared reconstruction error. 
The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and 
are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. 
In this paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). 
To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. 
To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. 
The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained 
to differentiate between the super-resolved images and original photo-realistic images. In addition, 
we use a content loss motivated by perceptual similarity instead of similarity in pixel space. 
Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. 
An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. 
The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained 
with any state-of-the-art method.

[[Paper]](https://arxiv.org/pdf/1609.04802)

```
@InProceedings{srgan,
    author = {Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi},
    title = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
    booktitle = {arXiv},
    year = {2016}
}
```
