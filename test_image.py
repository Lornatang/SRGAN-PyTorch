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
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import InterpolationMode as Mode

import srgan_pytorch.models as models
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.estimate import iqa
from srgan_pytorch.utils.transform import process_image

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using "
                                 "a Generative Adversarial Network.")
parser.add_argument("--lr", type=str, required=True,
                    help="Test low resolution image name.")
parser.add_argument("--hr", type=str,
                    help="Raw high resolution image name.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         " (default: srgan)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4, 8],
                    help="Low to high resolution scaling factor. Optional: [2, 4, 8] (default: 4)")
parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model.")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--seed", default=None, type=int,
                    help="Seed for initializing training.")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for training.")

    model = configure(args)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Set eval mode.
    model.eval()

    cudnn.benchmark = True

    # Get image filename.
    filename = os.path.basename(args.lr)

    # Read all pictures.
    lr = Image.open(args.lr)
    bicubic = transforms.Resize((lr.size[1] * args.upscale_factor, lr.size[0] * args.upscale_factor), Mode.BICUBIC)(lr)
    lr = process_image(lr, args.gpu)
    bicubic = process_image(bicubic, args.gpu)

    with torch.no_grad():
        c_time = time.time()
        sr = model(lr)
        print('Total time is %f sec. and FPS is %d'%((time.time()- c_time),1/(time.time()- c_time)))

    if args.hr:
        hr = process_image(Image.open(args.hr), args.gpu)
        vutils.save_image(hr, os.path.join("test", f"hr_{filename}"))
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

    vutils.save_image(lr, os.path.join("test", f"lr_{filename}"))
    vutils.save_image(bicubic, os.path.join("test", f"bicubic_{filename}"))
    vutils.save_image(sr, os.path.join("test", f"sr_{filename}"))
    vutils.save_image(images, os.path.join("test", f"compare_{filename}"), padding=10)


if __name__ == "__main__":
    print("##################################################\n")
    print("Run Testing Engine.\n")

    create_folder("test")

    logger.info("TestingEngine:")
    print("\tAPI version .......... 0.1.0")
    print("\tBuild ................ 2021.03.23")
    print("##################################################\n")
    main()

    logger.info("Test single image performance evaluation completed successfully.\n")
