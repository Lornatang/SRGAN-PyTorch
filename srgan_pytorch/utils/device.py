# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
"""Select the specified device for processing"""
import logging
import os

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def select_device(device: str = "", batch_size: int = 1) -> torch.device:
    r""" Choose the right equipment.

    Args:
        device (optional, str): Use CPU or CUDA. (Default: ````)
        batch_size (optional, int): Data batch size, cannot be less than the number of devices. (Default: 1).

    Returns:
        torch.device.
    """
    # device = "cpu" or "cuda:0,1,2,3".
    only_cpu = device.lower() == "cpu"
    if device and not only_cpu:  # if device requested other than "cpu".
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable.
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if only_cpu else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB.
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and batch_size:  # check that batch_size is compatible with device_count.
            assert batch_size % gpu_count == 0, f"batch-size {batch_size} not multiple of GPU count {gpu_count}"
        x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
        s = "Using CUDA "
        for i in range(0, gpu_count):
            if i == 1:
                s = " " * len(s)
            logger.info(f"{s}\n\t+ device:{i} (name=`{x[i].name}`, total_memory={int(x[i].total_memory / c)}MB)")
    else:
        logger.info("Using CPU.")

    return torch.device("cuda:0" if cuda else "cpu")
