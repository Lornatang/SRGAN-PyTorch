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
"""Select the specified device for processing"""
import logging
import os

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


# Reference https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py#select_device
def select_device(device="", batch_size=None):
    r""" Choose the right equipment.

    Args:
        device (optional, str): Use CPU or CUDA. (Default: ````)
        batch_size (optional, int): Data batch size, cannot be less than the number of devices. (Default: 1).

    Returns:
        torch.device.
    """
    # device = "cpu" or "cuda:0,1,2,3".
    s = ""
    cpu = device.lower() == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += "CPU\n"

    logger.info("\n" + s)  # skip a line
    return torch.device("cuda" if cuda else "cpu")
