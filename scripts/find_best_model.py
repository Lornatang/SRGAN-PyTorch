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
from glob import glob

import torch
import torch.backends.cudnn as cudnn
from PIL import Image

import srgan_pytorch.models as models
from srgan_pytorch.utils.estimate import iqa
from srgan_pytorch.utils.transform import process_image

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.")
parser.add_argument("--lr", type=str, required=True,
                    help="Test low resolution image name.")
parser.add_argument("--hr", type=str, required=True,
                    help="Raw high resolution image name.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         ". (Default: srgan)")
parser.add_argument("--model-dir", default="", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model.")
parser.add_argument("--seed", default=666, type=int,
                    help="Seed for initializing training. (default: 666).")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")

best_mse_value = 0.0
best_rmse_value = 0.0
best_psnr_value = 0.0
best_ssim_value = 0.0
best_lpips_value = 1.0
best_gmsd_value = 1.0


def main():
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn("You have chosen to seed training. "
                  "This will turn on the CUDNN deterministic setting, "
                  "which can slow down your training considerably! "
                  "You may see unexpected behavior when restarting "
                  "from checkpoints.")

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_mse_value, best_rmse_value, best_psnr_value, best_ssim_value, best_lpips_value, best_gmsd_value
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for testing.")

    cudnn.benchmark = True

    model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")

    # Read all pictures.
    lr = process_image(Image.open(args.lr), args.gpu)
    hr = process_image(Image.open(args.hr), args.gpu)

    model_paths = glob(os.path.join(f"{args.model_dir}", "Generator_epoch*.pth"))
    best_model = model_paths[0]

    for model_path in model_paths:
        print(f"Process `{model_path}`")
        value = inference(lr, hr, model, model_path, gpu)

        is_best = value[2] > best_psnr_value

        if is_best:
            best_model = os.path.basename(model_path)
            best_mse_value = value[0]
            best_rmse_value = value[1]
            best_psnr_value = value[2]
            best_ssim_value = value[3]
            best_lpips_value = value[4]
            best_gmsd_value = value[5]

    print("\n##################################################")
    print(f"Best model: `{best_model}`.")
    print(f"indicator Score")
    print(f"--------- -----")
    print(f"MSE       {best_mse_value:6.4f}\n"
          f"RMSE      {best_rmse_value:6.2f}\n"
          f"PSNR      {best_psnr_value:6.2f}\n"
          f"SSIM      {best_ssim_value:6.4f}\n"
          f"LPIPS     {best_lpips_value:6.4f}\n"
          f"GMSD      {best_gmsd_value:6.4f}")
    print(f"--------- -----")
    print("##################################################\n")


def inference(lr, hr, model, model_path, gpu: int = None):
    model.load_state_dict(torch.load(model_path)["state_dict"])

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    # Set eval mode.
    model.eval()

    with torch.no_grad():
        sr = model(lr)
        value = iqa(sr, hr, gpu)

    return value


if __name__ == "__main__":
    main()
