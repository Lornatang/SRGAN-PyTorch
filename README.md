# SRGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

### Table of contents
1. [About Super Resolution Generative Adversarial Networks](#about-super-resolution-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download dataset](#download-dataset)
4. [Test](#test)
    * [Test benchmark](#test-benchmark)
    * [Test image](#test-image)
    * [Test video](#test-video)
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

#### Download dataset

```bash
$ cd data/
$ bash download_dataset.sh
```

### Test

#### Test benchmark

```text
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N]
                         [--upscale-factor {2,4}] [--model-path PATH]
                         [--device DEVICE]

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weights/SRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).


# Example
$ python test_benchmark.py --dataroot ./data/DIV2K --upscale-factor 4 --model-path ./weight/SRGAN_X4.pth --device 0
```

#### Test image

```text
usage: test_image.py [-h] [--lr LR] [--hr HR] [--upscale-factor {2,4}]
                     [--model-path PATH] [--device DEVICE]

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weight/SRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).

# Example
$ python test_image.py --lr ./lr.png --hr ./hr.png --upscale-factor 4 --model-path ./weight/SRGAN_X4.pth --device 0
```

#### Test video

```text
usage: test_video.py [-h] --file FILE [--upscale-factor {2,4}]
                     [--model-path PATH] [--device DEVICE] [--view]

SRGAN algorithm is applied to video files.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weight/SRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).
  --view                Super resolution real time to show.

# Example
$ python test_video.py --file ./lr.mp4 --upscale-factor 4 --model-path ./weight/SRGAN_X4.pth --device 0
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--start-epoch N]
                [--psnr-iters N] [--iters N] [-b N] [--lr LR]
                [--upscale-factor {2,4,8}] [--resume_PSNR] [--resume]
                [--manualSeed MANUALSEED] [--device DEVICE]

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --start-epoch N       manual epoch number (useful on restarts)
  --psnr-iters N        The number of iterations is needed in the training of
                        PSNR model. (default:1e6)
  --iters N             The training of srgan model requires the number of
                        iterations. (default:2e5)
  -b N, --batch-size N  mini-batch size (default: 16), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:1e-4)
  --upscale-factor {2,4,8}
                        Low to high resolution scaling factor. (default:4).
  --resume_PSNR         Path to latest checkpoint for PSNR model.
  --resume              Path to latest checkpoint for Generator.
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:10000)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ``).

# Example (e.g DIV2K)
$ python train.py --dataroot ./data/DIV2K --upscale-factor 4
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python train.py --dataroot ./data/DIV2K \
                  --upscale-factor 4        \
                  --resume_PSNR \
                  --resume
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
