# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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
import logging
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import InterpolationMode

import srgan_pytorch.models as models
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.estimate import iqa
from srgan_pytorch.utils.transform import process_image

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.")
parser.add_argument("--lr", type=str, required=True,
                    help="Test low resolution image name.")
parser.add_argument("--hr", type=str,
                    help="Raw high resolution image name.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         ". (Default: `srgan`)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4, 8],
                    help="Low to high resolution scaling factor. Optional: [2, 4, 8]. (Default: 4)")
parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (Default: ``)")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--seed", default=666, type=int,
                    help="Seed for initializing training. (Default: 666)")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")


def main():
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for testing.")

    model = configure(args)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Set eval mode.
    model.eval()

    cudnn.benchmark = True

    # Get image filename.
    filename = os.path.basename(args.lr)

    # Read all pictures.
    lr = Image.open(args.lr)
    bicubic = transforms.Resize((lr.size[1] * args.upscale_factor, lr.size[0] * args.upscale_factor), InterpolationMode.BICUBIC)(lr)
    lr = process_image(lr, args.gpu)
    bicubic = process_image(bicubic, args.gpu)

    with torch.no_grad():
        sr = model(lr)

    if args.hr:
        hr = process_image(Image.open(args.hr), args.gpu)
        vutils.save_image(hr, os.path.join("tests", f"hr_{filename}"))
        images = torch.cat([bicubic, sr, hr], dim=-1)

        value = iqa(sr, hr, args.gpu)
        print(f"Performance avg results:\n")
        print(f"indicator Score\n")
        print(f"--------- -----\n")
        print(f"MSE       {value[0]:6.4f}\n"
              f"RMSE      {value[1]:6.4f}\n"
              f"PSNR      {value[2]:6.2f}\n"
              f"SSIM      {value[3]:6.4f}\n"
              f"LPIPS     {value[4]:6.4f}\n"
              f"GMSD      {value[5]:6.4f}\n")
    else:
        images = torch.cat([bicubic, sr], dim=-1)

    vutils.save_image(lr, os.path.join("tests", f"lr_{filename}"))
    vutils.save_image(bicubic, os.path.join("tests", f"bicubic_{filename}"))
    vutils.save_image(sr, os.path.join("tests", f"sr_{filename}"))
    vutils.save_image(images, os.path.join("tests", f"compare_{filename}"), padding=10)


if __name__ == "__main__":
    print("##################################################\n")
    print("Run Testing Engine.\n")

    create_folder("tests")

    logger.info("TestingEngine:")
    print("\tAPI version .......... 0.2.2")
    print("\tBuild ................ 2021.04.23")
    print("##################################################\n")
    main()

    logger.info("Test single image performance evaluation completed successfully.\n")
