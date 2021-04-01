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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from tqdm import tqdm

import srgan_pytorch.models as models
from srgan_pytorch.dataset import BaseTestDataset
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.estimate import iqa

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network")
parser.add_argument("data", metavar="DIR",
                    help="path to dataset")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         " (default: srgan)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default: 4)")
parser.add_argument("-b", "--batch-size", default=16, type=int,
                    metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--sampler-frequency", default=1, type=int, metavar="N",
                    help="If there are many datasets, this method can be used "
                         "to increase the number of epochs. (default:1)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Image size of high resolution image. (default: 96)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4, 8],
                    help="Low to high resolution scaling factor. Optional: [2, 4, 8] (default: 4)")
parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model.")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--world-size", default=-1, type=int,
                    help="Number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int,
                    help="Node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://59.110.31.55:12345", type=str,
                    help="url used to set up distributed training. (default: tcp://59.110.31.55:12345)")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="Distributed backend. (default: nccl)")
parser.add_argument("--seed", default=None, type=int,
                    help="Seed for initializing training.")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training")

total_mse_value = 0.0
total_rmse_value = 0.0
total_psnr_value = 0.0
total_ssim_value = 0.0
total_lpips_value = 0.0
total_gmsd_value = 0.0


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

    if args.gpu is not None:
        logger.warning("You have chosen a specific GPU. This will completely disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global total_mse_value, total_rmse_value, total_psnr_value
    global total_ssim_value, total_lpips_value, total_gmsd_value
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for testing.")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    model = configure(args)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = nn.parallel.DistributedDataParallel(module=model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    logger.info("Load testing dataset")
    # Selection of appropriate treatment equipment.
    dataset = BaseTestDataset(root=os.path.join(args.data, "test"), image_size=args.image_size, upscale_factor=args.upscale_factor)
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

    # Start evaluate model performance.
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, (lr, bicubic, hr) in progress_bar:
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            bicubic = bicubic.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
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
    print("\tAPI version .......... 0.1.0")
    print("\tBuild ................ 2021.04.01")
    print("##################################################\n")
    main()

    logger.info("Test dataset performance evaluation completed successfully.")
