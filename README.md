# SRGAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

### Table of contents

1. [About Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](#about-photo-realistic-single-image-super-resolution-using-a-generative-adversarial-network)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download dataset](#download-dataset)
4. [Test](#test)
    * [Test benchmark](#test-benchmark)
    * [Test image](#test-image)
    * [Test video](#test-video)
    * [Test model performance](#test-model-performance)
4. [Train](#train-eg-div2k)
5. [Contributing](#contributing)
6. [Credit](#credit)

### About Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

If you're new to SRGAN, here's an abstract straight from the paper:

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central
problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of
optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on
minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking
high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this
paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable
of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an
adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is
trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by
perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily
downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN.
The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates
images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is
a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

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
usage: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. [-h] [-a ARCH] [-j N] [-b N] [--sampler-frequency N] [--image-size IMAGE_SIZE] [--upscale-factor {2,4,8}] [--model-path PATH] [--pretrained]
                                                                                             [--seed SEED] [--gpu GPU]
                                                                                             DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  Model architecture: srgan | srgan_2x2 | srgan_8x8. (Default: `srgan`)
  -j N, --workers N     Number of data loading workers. (Default: 8)
  -b N, --batch-size N  mini-batch size (default: 32), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --sampler-frequency N
                        If there are many datasets, this method can be used to increase the number of epochs. (Default:1)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 96)
  --upscale-factor {2,4,8}
                        Low to high resolution scaling factor. Optional: [2, 4, 8]. (Default: 4)
  --model-path PATH     Path to latest checkpoint for model. (Default: ``)
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training. (Default: 666)
  --gpu GPU             GPU id to use.
                     
# Example
$ python3 test_benchmark.py -a srgan --pretrained --gpu 0 [image-folder with train and val folders]
```

#### Test image

```text
usage: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. [-h] --lr LR [--hr HR] [-a ARCH] [--upscale-factor {2,4,8}] [--model-path PATH] [--pretrained] [--seed SEED] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  -a ARCH, --arch ARCH  Model architecture: srgan | srgan_2x2 | srgan_8x8. (Default: `srgan`)
  --upscale-factor {2,4,8}
                        Low to high resolution scaling factor. Optional: [2, 4, 8]. (Default: 4)
  --model-path PATH     Path to latest checkpoint for model. (Default: `./weights/GAN.pth`)
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training. (Default: 666)
  --gpu GPU             GPU id to use.

# Example
$ python3 test_image.py -a srgan --lr [path-to-lr-image] --hr [Optional, path-to-hr-image] --pretrained --gpu 0
```

#### Test video

```text
usage: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. [-h] --file FILE [-a ARCH] [--upscale-factor {2,4,8}] [--model-path PATH] [--pretrained] [--seed SEED] [--gpu GPU] [--view]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  -a ARCH, --arch ARCH  Model architecture: srgan | srgan_2x2 | srgan_8x8. (Default: `srgan`)
  --upscale-factor {2,4,8}
                        Low to high resolution scaling factor. Optional: [2, 4, 8]. (Default: 4)
  --model-path PATH     Path to latest checkpoint for model. (Default: ``)
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training. (Default: 666)
  --gpu GPU             GPU id to use.
  --view                Do you want to show SR video synchronously.
                        
# Example
$ python3 test_video.py -a srgan --file [path-to-video] --pretrained --gpu 0 --view 
```

#### Test model performance

|      Model      | Params | FLOPs | CPU Speed | GPU Speed |
|:---------------:|:------:|:-----:|:---------:|:---------:|
|      srgan      | 1.55M  |  2.6G |      9ms  |      7ms  |
|    srgan_2x2    | 1.40M  |  1.7G |     15ms  |      3ms  |
|    srgan_8x8    | 1.70M  |  6.1G |     18ms  |      3ms  |

```text
usage: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. [-h] [-i IMAGE_SIZE] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_SIZE, --image-size IMAGE_SIZE
                        Image size of low-resolution. (Default: 24)
  --gpu GPU             GPU id to use.
  
# Example (CPU: Intel i9-10900X/GPU: Nvidia GeForce RTX 2080Ti)
$ python3 setup.py install
$ python3 scripts/cal_model_complexity.py --gpu 0
```

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png" alt=""></span>

### Train (e.g DIV2K)

```text
usage: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. [-h] [-a ARCH] [-j N] [--psnr-epochs N] [--start-psnr-epoch N] [--gan-epochs N] [--start-gan-epoch N] [-b N] [--sampler-frequency N] [--psnr-lr PSNR_LR]
                                                                                             [--gan-lr GAN_LR] [--image-size IMAGE_SIZE] [--upscale-factor {2,4,8}] [--model-path PATH] [--resume_psnr PATH] [--resume_d PATH] [--resume_g PATH]
                                                                                             [--pretrained] [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
                                                                                             [--multiprocessing-distributed]
                                                                                             DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  Model architecture: srgan | srgan_2x2 | srgan_8x8. (Default: `srgan`)
  -j N, --workers N     Number of data loading workers. (Default: 4)
  --psnr-epochs N       Number of total psnr epochs to run. (Default: 20000)
  --start-psnr-epoch N  Manual psnr epoch number (useful on restarts). (Default: 0)
  --gan-epochs N        Number of total gan epochs to run. (Default: 4000)
  --start-gan-epoch N   Manual gan epoch number (useful on restarts). (Default: 0)
  -b N, --batch-size N  Mini-batch size (default: 16), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel.
  --sampler-frequency N
                        If there are many datasets, this method can be used to increase the number of epochs. (Default:1)
  --psnr-lr PSNR_LR     Learning rate for psnr-oral. (Default: 0.0001)
  --gan-lr GAN_LR       Learning rate for gan-oral. (default: 0.0001)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 96)
  --upscale-factor {2,4,8}
                        Low to high resolution scaling factor. Optional: [2, 4, 8] (Default: 4)
  --model-path PATH     Path to latest checkpoint for model.
  --resume_psnr PATH    Path to latest psnr-oral checkpoint.
  --resume_d PATH       Path to latest -oral checkpoint.
  --resume_g PATH       Path to latest psnr-oral checkpoint.
  --pretrained          Use pre-trained model.
  --world-size WORLD_SIZE
                        Number of nodes for distributed training.
  --rank RANK           Node rank for distributed training. (Default: -1)
  --dist-url DIST_URL   url used to set up distributed training. (Default: `tcp://59.110.31.55:12345`)
  --dist-backend DIST_BACKEND
                        Distributed backend. (Default: `nccl`)
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.
                   
# Example (e.g DIV2K)
$ python3 train.py -a srgan --gpu 0 [image-folder with train and val folders]
# Multi-processing Distributed Data Parallel Training
$ python3 train.py -a srgan --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [image-folder with train and val folders]
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a srgan --start-psnr-epoch 10 --resume-psnr weights/PSNR_epoch10.pth [image-folder with train and val folders] 
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

_Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan
Wang, Wenzhe Shi_ <br>

**Abstract** <br>
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central
problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of
optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on
minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking
high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this
paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable
of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an
adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is
trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by
perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily
downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN.
The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

[[Paper]](https://arxiv.org/pdf/1609.04802)

```
@InProceedings{srgan,
    author = {Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi},
    title = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
    booktitle = {arXiv},
    year = {2016}
}
```
