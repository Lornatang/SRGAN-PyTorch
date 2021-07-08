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
import logging
import torch
from thop import profile
from time import perf_counter

from srgan_pytorch.models import srgan

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

# The test image is an RGB three-channel image.
IMAGE_CHANNELS = 3
# Low resolution image resolution is 96x96.
IMAGE_SIZE = 96
# A total of 64 runs and averaged.
BATCH_SIZE = 64


def main():
    # Build a super-resolution model, if model path is defined, the specified model weight will be loaded.
    model = srgan(pretrained=False)
    # Switch model to eval mode.
    model.eval()

    # Create an image that conforms to the normal distribution.
    data = torch.randn([1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE])

    # Calculate all parameters of the model and format them.
    params = sum(x.numel() for x in model.parameters()) / 1E6

    # Cal flops and parameters.
    flops = profile(model=model, inputs=(data,), verbose=False)[0] / 1E9 * 2

    # Needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        start_time = perf_counter()

        # Take the average time with 64 cycles of testing.
        for _ in range(BATCH_SIZE):
            _ = model(data)

        stop_time = perf_counter()
        cost_time = stop_time - start_time
        # Waits for all kernels in all streams on a CUDA device to complete.
        print(f"\nModel: `srgan`.\n")
        print(f"Parameters: {params:.2f}M.\n")
        print(f"FLOPs: {flops:.2f}G.\n")
        print(f"Cost time: {cost_time / BATCH_SIZE:.3f}s.")


if __name__ == "__main__":
    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.07.02")

    main()
