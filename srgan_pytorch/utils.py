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
import logging
import os

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

__all__ = [
    "img2tensor", "init_torch_seeds", "load_checkpoint", "select_device",
    "tensor2img",
]

logger = logging.getLogger(__name__)


def img2tensor():
    r"""Read array image into tensor format."""
    return transforms.ToTensor()


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a

    Args:
        seed (int): The desired seed.
    """
    torch.manual_seed(seed)

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Adam = torch.optim.Adam,
                    file: str = None) -> int:
    r""" Quick loading model functions

    Args:
        model (nn.Module): Neural network model.
        optimizer (torch.optim): Model optimizer. (Default: torch.optim.Adam)
        file (str): Model file.

    Returns:
        How much epoch to start training from.
    """
    if os.path.isfile(file):
        logger.info(f"[*] Loading checkpoint `{file}`.")
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"[*] Loaded checkpoint `{file}` (epoch {checkpoint['epoch']})")
    else:
        logger.info(f"[!] no checkpoint found at '{file}'")
        epoch = 0

    return epoch


def select_device(device: str = None, batch_size: int = 1) -> torch.device:
    r""" Choose the right equipment.

    Args:
        device (str): Use CPU or CUDA. (Default: None)
        batch_size (int, optional): Data batch size, cannot be less than the number of devices. (Default: 1).

    Returns:
        torch.device.
    """
    # device = "cpu" or "cuda:0,1,2,3"
    only_cpu = device.lower() == "cpu"
    if device and not only_cpu:  # if device requested other than "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if only_cpu else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % gpu_count == 0, f"batch-size {batch_size} not multiple of GPU count {gpu_count}"
        x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
        s = "Using CUDA "
        for i in range(0, gpu_count):
            if i == 1:
                s = " " * len(s)
            logger.info(f"{s}\n\t+ device:{i} (name=`{x[i].name}`, "
                        f"total_memory={int(x[i].total_memory / c)}MB)")
    else:
        logger.info("Using CPU")

    logger.info("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def tensor2img():
    r"""Read tensor format  into image array."""
    return transforms.ToPILImage()
