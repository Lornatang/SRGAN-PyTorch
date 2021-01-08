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
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import srgan_pytorch.models as models
from .device import select_device

__all__ = [
    "create_folder", "configure", "inference", "init_torch_seeds", "save_checkpoint", "weights_init",
    "AverageMeter", "ProgressMeter"
]

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def create_folder(folder):
    try:
        os.makedirs(folder)
        logger.info(f"Create `{os.path.join(os.getcwd(), folder)}` directory successful.")
    except OSError:
        logger.warning(f"Directory `{os.path.join(os.getcwd(), folder)}` already exists!")
        pass


def configure(args):
    """Global profile.

    Args:
        args (argparse.ArgumentParser.parse_args): Use argparse library parse command.
    """
    # Selection of appropriate treatment equipment
    device = select_device(args.device, batch_size=1)

    # Create model
    if args.pretrained:
        logger.info(f"Using pre-trained model `{args.arch}`")
        model = models.__dict__[args.arch](pretrained=True, upscale_factor=args.upscale_factor).to(device)
    else:
        logger.info(f"Creating model `{args.arch}`")
        model = models.__dict__[args.arch](upscale_factor=args.upscale_factor).to(device)
        if args.model_path:
            logger.info(f"You loaded the specified weight. Load weights from `{args.model_path}`")
            model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

    return model, device


def inference(model, lr, statistical_time=False):
    r"""General inference method.

    Args:
        model (nn.Module): Neural network model.
        lr (Torch.Tensor): Picture in pytorch format (N*C*H*W).
        statistical_time (optional, bool): Is reasoning time counted. (default: ``False``).

    Returns:
        super resolution image, time consumption of super resolution image (if `statistical_time` set to `True`).
    """
    # Set eval model.
    model.eval()

    if statistical_time:
        start_time = time.time()
        with torch.no_grad():
            sr = model(lr)
        use_time = time.time() - start_time
        return sr, use_time
    else:
        with torch.no_grad():
            sr = model(lr)
        return sr


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a

    Args:
        seed (int): The desired seed.
    """

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    logger.info("Initialize random seed.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, is_best: bool, source_filename: str, target_filename: str):
    torch.save(state, source_filename)
    if is_best:
        torch.save(state["state_dict"], target_filename)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + '/' + fmt.format(num_batches) + "]"
