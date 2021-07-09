# SRGAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
.

### Table of contents

- [SRGAN-PyTorch](#srgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Installation](#installation)
      - [Clone and install requirements](#clone-and-install-requirements)
      - [Download dataset](#download-dataset)
    - [Test (e.g Set5)](#test-eg-set5)
    - [Test video](#test-video)
    - [Train (e.g DIV2K)](#train-eg-div2k)
    - [Model performance](#model-performance)
    - [Result](#result)
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

### Test (e.g Set5)

Set5 dataset:

- [baiduclouddisk](https://pan.baidu.com/s/1p2h2XTWMD3FLuvKzwpB3Bg) access: `llot`
- [gooleclouddisk](https://drive.google.com/file/d/13cG8rBaqY3H2xkFSLOgMvUGhwcBJyKoT/view?usp=sharing)

Set14 dataset:

- [baiduclouddisk](https://pan.baidu.com/s/10HqYjvlHSVso_-PkXYA85w) access: `llot`
- [googlecloudisk](https://drive.google.com/file/d/1nwmlu4xeLpSLoP89gFcj-557ymWwKZV2/view?usp=sharing)

```text
usage: test.py [-h] [--pretrained] [--model-path MODEL_PATH] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --pretrained          Use pre-trained model.
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --cuda                Enables cuda.
  
# Example (Set5 dataset)
python test.py --pretrained
```

### Test video

```text
usage: test_video.py [-h] --file FILE [--pretrained] [--model-path MODEL_PATH] [--cuda] [--view]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --pretrained          Use pre-trained model.
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --cuda                Enables cuda.
  --view                Do you want to show SR video synchronously.
 # Example
 python test_video.py --file [path-to-video] --pretrained
```

### Train (e.g DIV2K)

```text
usage: test_video.py [-h] --file FILE [--pretrained] [--model-path MODEL_PATH] [--cuda] [--view]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --pretrained          Use pre-trained model.
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --cuda                Enables cuda.
  --view                Do you want to show SR video synchronously.

C:\code\SRGAN-PyTorch>python train.py -h
usage: train.py [-h] [--pretrained] [--model-path MODEL_PATH] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --pretrained          Use pre-trained model.
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --cuda                Enables cuda.

# Example
python train.py --cuda
# If you want to load weights that you've trained before, run the following command.
python train.py --netG weights/P_epoch10.pth --cuda
```

### Model performance

| Model | Params | FLOPs | CPU Speed | GPU Speed |
| :---: | :----: | :---: | :-------: | :-------: |
| srgan | 1.55M  | 2.6G  |    9ms    |    7ms    |

```text
# Example (CPU: Intel i9-10900X/GPU: Nvidia GeForce RTX 2080Ti)
python setup.py install --user --prefix=""
python scripts/cal_model_complexity.py --gpu 0
```

### Result

Source of original paper results: https://arxiv.org/pdf/1609.04802v5.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       |        SSIM        |     MOS     |
| :-----: | :---: | :--------------: | :----------------: | :---------: |
|  Set5   |   4   | 29.40(**29.88**) | 0.8472(**0.8504**) | 3.58(**-**) |
|  Set14  |   4   | 26.02(**26.58**) | 0.7397(**0.7452**) | 3.72(**-**) |
| BSDS100 |   4   | 25.16(**25.50**) | 0.6688(**0.7284**) | 3.56(**-**) |

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png" alt=""></span>

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request.
Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

_Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham,
Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang,
Wenzhe Shi_ <br>

**Abstract** <br>
Despite the breakthroughs in accuracy and speed of single image super-resolution
using faster and deeper convolutional neural networks, one central problem
remains largely unsolved: how do we recover the finer texture details when we
super-resolve at large upscaling factors? The behavior of optimization-based
super-resolution methods is principally driven by the choice of the objective
function. Recent work has largely focused on minimizing the mean squared
reconstruction error. The resulting estimates have high peak signal-to-noise
ratios, but they are often lacking high-frequency details and are perceptually
unsatisfying in the sense that they fail to match the fidelity expected at the
higher resolution. In this paper, we present SRGAN, a generative adversarial
network (GAN) for image super-resolution (SR). To our knowledge, it is the first
framework capable of inferring photo-realistic natural images for 4x upscaling
factors. To achieve this, we propose a perceptual loss function which consists
of an adversarial loss and a content loss. The adversarial loss pushes our
solution to the natural image manifold using a discriminator network that is
trained to differentiate between the super-resolved images and original
photo-realistic images. In addition, we use a content loss motivated by
perceptual similarity instead of similarity in pixel space. Our deep residual
network is able to recover photo-realistic textures from heavily downsampled
images on public benchmarks. An extensive mean-opinion-score (MOS) test shows
hugely significant gains in perceptual quality using SRGAN. The MOS scores
obtained with SRGAN are closer to those of the original high-resolution images
than to those obtained with any state-of-the-art method.

[[Paper]](https://arxiv.org/pdf/1609.04802)

```
@InProceedings{srgan,
    author = {Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi},
    title = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
    booktitle = {arXiv},
    year = {2016}
}
```
