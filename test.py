# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from srgan_pytorch import Discriminator
from srgan_pytorch import FeatureExtractor
from srgan_pytorch import Generator
from srgan_pytorch import weights_init

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                    help="Number of data loading workers. (default:8)")
parser.add_argument("--epochs", default=2000, type=int, metavar="N",
                    help="Number of total epochs to run. (default:2000)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Size of the data crop (squared assumed). (default:96)")
parser.add_argument("-b", "--batch-size", default=8, type=int,
                    metavar="N",
                    help="mini-batch size (default: 8), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (default:0.0001)")
parser.add_argument("--up-sampling", type=int, default=4,
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--netG", default="", help="Path to netG (to continue training).")
parser.add_argument("--netD", default="", help="Path to netD (to continue training).")
parser.add_argument("--outf", default="./results",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")