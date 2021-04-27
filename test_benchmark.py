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
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

import srgan_pytorch.models as models
from srgan_pytorch.dataset import BaseTestDataset
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.estimate import iqa

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.")
parser.add_argument("data", metavar="DIR",
                    help="Path to dataset.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         ". (Default: `srgan`)")
parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                    help="Number of data loading workers. (Default: 8)")
parser.add_argument("-b", "--batch-size", default=32, type=int,
                    metavar="N",
                    help="mini-batch size (default: 32), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--sampler-frequency", default=1, type=int, metavar="N",
                    help="If there are many datasets, this method can be used "
                         "to increase the number of epochs. (Default:1)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Image size of high resolution image. (Default: 96)")
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

total_mse_value = 0.0
total_rmse_value = 0.0
total_psnr_value = 0.0
total_ssim_value = 0.0
total_lpips_value = 0.0
total_gmsd_value = 0.0


def main():
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global total_mse_value, total_rmse_value, total_psnr_value, total_ssim_value, total_lpips_value, total_gmsd_value
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for testing.")

    model = configure(args)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    logger.info("Load testing dataset.")
    # Selection of appropriate treatment equipment.
    dataset = BaseTestDataset(os.path.join(args.data, "test"), args.image_size, args.upscale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers)
    logger.info(f"Dataset information:\n"
                f"\tPath:              {os.getcwd()}/{args.data}/test\n"
                f"\tNumber of samples: {len(dataset)}\n"
                f"\tNumber of batches: {len(dataloader)}\n"
                f"\tShuffle:           False\n"
                f"\tSampler:           None\n"
                f"\tWorkers:           {args.workers}")

    cudnn.benchmark = True

    # Set eval mode.
    model.eval()

    with torch.no_grad():
        # Start evaluate model performance.
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (lr, bicubic, hr) in progress_bar:
            # Move data to special device.
            if args.gpu is not None:
                lr = lr.cuda(args.gpu, non_blocking=True)
                bicubic = bicubic.cuda(args.gpu, non_blocking=True)
                hr = hr.cuda(args.gpu, non_blocking=True)

            sr = model(lr)

            # Evaluate performance
            value = iqa(sr, hr, args.gpu)

            total_mse_value += value[0]
            total_rmse_value += value[1]
            total_psnr_value += value[2]
            total_ssim_value += value[3]
            total_lpips_value += value[4]
            total_gmsd_value += value[5]

            progress_bar.set_description(f"[{i + 1}/{len(dataloader)}] "
                                         f"PSNR: {total_psnr_value / (i + 1):6.2f} "
                                         f"SSIM: {total_ssim_value / (i + 1):6.4f}")

            images = torch.cat([bicubic, sr, hr], dim=-1)
            vutils.save_image(images, os.path.join("benchmarks", f"{i + 1}.bmp"), padding=10)

    print(f"Performance average results:\n")
    print(f"indicator Score\n")
    print(f"--------- -----\n")
    print(f"MSE       {total_mse_value / len(dataloader):6.4f}\n"
          f"RMSE      {total_rmse_value / len(dataloader):6.4f}\n"
          f"PSNR      {total_psnr_value / len(dataloader):6.2f}\n"
          f"SSIM      {total_ssim_value / len(dataloader):6.4f}\n"
          f"LPIPS     {total_lpips_value / len(dataloader):6.4f}\n"
          f"GMSD      {total_gmsd_value / len(dataloader):6.4f}")


if __name__ == "__main__":
    print("##################################################\n")
    print("Run Testing Engine.\n")

    create_folder("benchmarks")

    logger.info("TestingEngine:")
    print("\tAPI version .......... 0.2.2")
    print("\tBuild ................ 2021.04.27")
    print("##################################################\n")
    main()

    logger.info("Test dataset performance evaluation completed successfully.")
