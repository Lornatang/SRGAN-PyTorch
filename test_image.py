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
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import InterpolationMode

import srgan_pytorch.models as models
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.estimate import iqa
from srgan_pytorch.utils.transform import process_image

# Find all available models.
model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(args):
    if args.seed is not None:
        # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set
        # convolution algorithm.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.warning("You have chosen to seed training. "
                       "This will turn on the CUDNN deterministic setting, "
                       "which can slow down your training considerably! "
                       "You may see unexpected behavior when restarting "
                       "from checkpoints.")
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    # Build a super-resolution model, if model path is defined, the specified model weight will be loaded.
    model = configure(args)
    # If special choice model path.
    if args.model_path is not None:
        logger.info(f"You loaded the specified weight. Load weights from `{os.path.abspath(args.model_path)}`.")
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    # Switch model to eval mode.
    model.eval()

    # If the GPU is available, load the model into the GPU memory. This speed.
    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # Setting this flag allows the built-in auto tuner of cudnn to automatically find the most efficient
        # algorithm suitable for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    # Get image filename.
    filename = os.path.basename(args.lr)

    # Read the low resolution image and enlarge the low resolution image with bicubic method.
    # The purpose of bicubic method is to compare the reconstruction results.
    lr = Image.open(args.lr)
    bicubic_image_size = (lr.size[1] * args.upscale_factor, lr.size[0] * args.upscale_factor)
    bicubic = transforms.Resize(bicubic_image_size, InterpolationMode.BICUBIC)(lr)
    lr = process_image(lr, norm=False, gpu=args.gpu)
    bicubic = process_image(bicubic, norm=True, gpu=args.gpu)

    # Needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        sr = model(lr)

    # If there is a reference image, a series of evaluation indexes will be output.
    if args.hr:
        hr = process_image(Image.open(args.hr), norm=True, gpu=args.gpu)
        vutils.save_image(hr, os.path.join("tests", f"hr_{filename}"), normalize=True)
        # Merge three images into one line for visualization.
        images = torch.cat([bicubic, sr, hr], dim=-1)

        # The reconstructed image and the reference image are evaluated once.
        value = iqa(sr, hr, args.gpu)
        print(f"Performance avg results:\n")
        print(f"Indicator score\n")
        print(f"--------- -----\n")
        print(f"MSE       {value[0]:6.4f}\n"
              f"RMSE      {value[1]:6.4f}\n"
              f"PSNR      {value[2]:6.2f}\n"
              f"SSIM      {value[3]:6.4f}\n"
              f"GMSD      {value[4]:6.4f}\n")
    else:
        # Merge two images into one line for visualization.
        images = torch.cat([bicubic, sr], dim=-1)

    # Save a series of reconstruction results.
    vutils.save_image(lr, os.path.join("tests", f"lr_{filename}"))
    vutils.save_image(bicubic, os.path.join("tests", f"bicubic_{filename}"), normalize=True)
    vutils.save_image(sr, os.path.join("tests", f"sr_{filename}"), normalize=True)
    vutils.save_image(images, os.path.join("tests", f"compare_{filename}"), padding=10, normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="srgan", type=str, choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `srgan`)")
    parser.add_argument("--lr", type=str, required=True,
                        help="Test low resolution image name.")
    parser.add_argument("--hr", type=str,
                        help="Raw high resolution image name.")
    parser.add_argument("--upscale-factor", default=4, type=int, choices=[4],
                        help="Low to high resolution scaling factor. Optional: [4]. (Default: 4)")
    parser.add_argument("--model-path", default="", type=str,
                        help="Path to latest checkpoint for model.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for initializing training.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    create_folder("tests")

    logger.info("TestEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.06.13")

    main(args)

    logger.info("Test single image performance evaluation completed successfully.\n")
