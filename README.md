# SRGAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

### Table of contents

- [SRGAN-PyTorch](#srgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Installation](#installation)
      - [Clone and install requirements](#clone-and-install-requirements)
      - [Download dataset](#download-dataset)
    - [Train (e.g DIV2K)](#train-eg-div2k)
    - [Test](#test)
      - [Test benchmark](#test-benchmark)
      - [Test image](#test-image)
      - [Test video](#test-video)
      - [Test model performance](#test-model-performance)
    - [Contributing](#contributing)
    - [Credit](#credit)
      - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](#photo-realistic-single-image-super-resolution-using-a-generative-adversarial-network)

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

### Train (e.g DIV2K)
```text
usage: train.py [-h] [--start-psnr-epoch START_PSNR_EPOCH] [--start-gan-epoch START_GAN_EPOCH] [--netD NETD]
                [--netG NETG] [--pretrained] [--seed SEED] [--gpu GPU]
                DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  --start-psnr-epoch START_PSNR_EPOCH
                        Manual psnr epoch number (useful on restarts). (Default: 0)
  --start-gan-epoch START_GAN_EPOCH
                        Manual gan epoch number (useful on restarts). (Default: 0)
  --netD NETD           Path to Discriminator checkpoint.
  --netG NETG           Path to Generator checkpoint.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.

# Example
python train.py --gpu 0 [image-folder with train and val folders]
# If you want to load weights that you've trained before, run the following command.
python train.py --start-psnr-epoch 10 --netG weights/PSNR_epoch10.pth --gpu 0 [image-folder with train and val folders]
```

### Test

#### Test benchmark

```text
usage: test_benchmark.py [-h] [--model-path MODEL_PATH] [--pretrained] [--gpu GPU] DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --gpu GPU             GPU id to use.
                     
# Example
python test_benchmark.py --pretrained --gpu 0 [image-folder with train and val folders]
```

#### Test image

```text
usage: test_image.py [-h] --lr LR [--hr HR] [--model-path MODEL_PATH] [--pretrained] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --gpu GPU             GPU id to use.

# Example
python test_image.py --lr [path-to-lr-image] --hr [Optional, path-to-hr-image] --pretrained --gpu 0
```

#### Test video

```text
usage: test_video.py [-h] --file FILE [--model-path MODEL_PATH] [--pretrained] [--gpu GPU] [--view]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --gpu GPU             GPU id to use.
  --view                Do you want to show SR video synchronously.
                        
# Example
python test_video.py --file [path-to-video] --pretrained --gpu 0 --view 
```

#### Test model performance

| Model | Params | FLOPs | CPU Speed | GPU Speed |
| :---: | :----: | :---: | :-------: | :-------: |
| srgan | 1.55M  | 2.6G  |    9ms    |    7ms    |

```text
# Example (CPU: Intel i9-10900X/GPU: Nvidia GeForce RTX 2080Ti)
python setup.py install --user --prefix=""
python scripts/cal_model_complexity.py --gpu 0
```

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png" alt=""></span>

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

_Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken,
Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi_ <br>

**Abstract** <br>
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional
neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we
super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally
driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared
reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking
high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at
the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image
super-resolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images
for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss
and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network
that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we
use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is
able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive
mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained
with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art
method.

[[Paper]](https://arxiv.org/pdf/1609.04802)

```
@InProceedings{srgan,
    author = {Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi},
    title = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
    booktitle = {arXiv},
    year = {2016}
}
```
