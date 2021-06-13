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
import random
import time

import prettytable
import torch
import torch.backends.cudnn as cudnn
from thop import profile

import srgan_pytorch.models as models

# Find all available models.
model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def inference(arch, cpu_data, cuda_data, args) -> [float, float, float, float]:
    logger.info(f"Use `{arch}` model start testing...")
    # Build a super-resolution model, if model_ If path is defined, the specified model weight will be loaded.
    cpu_model = models.__dict__[arch]()
    # Switch model to eval mode.
    cpu_model.eval()

    # If the GPU is available, load the model into the GPU memory. This speed.
    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
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

    # Calculate all parameters of the model and format them.
    params = sum(x.numel() for x in cpu_model.parameters()) / 1E6

    # Cal flops and parameters.
    flops = profile(model=cpu_model, inputs=(cpu_data,), verbose=False)[0] / 1E9 * 2

    # It only needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    # By default, they are all processed on the CPU.
    with torch.no_grad():
        start_time = time.time()
        for _ in range(args.batch_size):
            _ = cpu_model(cpu_data)
        cpu_speed = (time.time() - start_time) / args.batch_size * 1E3
        cuda_speed = 0.

        # If a GPU is used, perform the same test on the GPU.
        if args.gpu is not None:
            cuda_model = cpu_model.cuda(args.gpu)
            start_time = time.time()
            for _ in range(args.batch_size):
                _ = cuda_model(cuda_data)
            cuda_speed = (time.time() - start_time) / args.batch_size * 1E3

    return params, flops, cpu_speed, cuda_speed


def main(args) -> None:
    # For visual table analysis.
    table = prettytable.PrettyTable()

    # Create an image that conforms to the normal distribution.
    cpu_data = torch.randn([1, 3, args.image_size, args.image_size], requires_grad=False)
    if args.gpu is not None:
        cuda_data = cpu_data.cuda(args.gpu)
    else:
        cuda_data = None

    # List or tuple of field names.
    table.field_names = ["Model", "Params", "FLOPs", "CPU Speed", "GPU Speed"]

    # Add the data to the table in turn.
    for i in range(len(model_names)):
        value = inference(model_names[i], cpu_data, cuda_data, args)
        table.add_row([f"{model_names[i].center(15)}",
                       f"{value[0]:4.2f}M",
                       f"{value[1]:4.2f}G",
                       f"{int(value[2]):4d}ms",
                       f"{int(value[3]):4d}ms"])

    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", default=24, type=int,
                        help="Image size of low-resolution. (Default: 24)")
    parser.add_argument("--batch-size", default=128, type=int,
                        help="In order to ensure the fairness test, many experiments are carried out. (Default: 128)")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.06.13")

    main(args)
