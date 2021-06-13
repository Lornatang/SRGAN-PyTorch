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

import torch
import torch.backends.cudnn as cudnn

import srgan_pytorch.models as models
from srgan_pytorch.utils.common import configure

# Find all available models.
model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)


def main(args) -> None:
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
    # Switch model to eval mode.
    model.eval()

    # Create an image that conforms to the normal distribution.
    data = torch.randn([1, 3, args.image_size, args.image_size], requires_grad=False)

    # If there is a GPU, the data will be loaded into the GPU memory.
    if args.gpu is not None:
        data = data.cuda(args.gpu, non_blocking=True)

    # It only needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        start = time.time()
        _ = model(data)
        # Waits for all kernels in all streams on a CUDA device to complete.
        torch.cuda.synchronize()
        print(f"Time:{(time.time() - start) * 1000:.2f}ms.")

    # Context manager that manages autograd profiler state and holds a summary of results.
    with torch.autograd.profiler.profile(enabled=True, use_cuda=args.gpu) as profile:
        _ = model(data)
    print(profile.table())
    # Open Chrome browser and enter in the address bar `chrome://tracing`
    profile.export_chrome_trace("profile.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="pmigan", type=str, choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `pmigan`)")
    parser.add_argument("-i", "--image-size", default=54, type=int,
                        help="Image size of low-resolution. (Default: 54)")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for initializing training.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.06.13")

    main(args)
