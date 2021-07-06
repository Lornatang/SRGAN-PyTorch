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

import torch
import torch.utils.data

from srgan_pytorch.dataset import BaseTestDataset
from srgan_pytorch.models import srgan
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.estimate import test

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(args):
    # Build a super-resolution model, if model path is defined, the specified model weight will be loaded.
    model = srgan(pretrained=args.pretrained)
    # Switch model to eval mode.
    model.eval()

    # If special choice model path.
    if args.model_path is not None:
        logger.info(f"You loaded the specified weight. Load weights from `{os.path.abspath(args.model_path)}`.")
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))

    # Selection of appropriate treatment equipment.
    dataset = BaseTestDataset(os.path.join(args.data, "test"), 96, 4)
    dataloader = torch.utils.data.DataLoader(dataset, 32, pin_memory=True)

    # Needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        # The reconstructed image and the reference image are evaluated once.
        psnr_value, ssim_value = test(dataloader, model, args.gpu)

    print(f"Performance average results:\n")
    print(f"Indicator score\n")
    print(f"--------- -----\n")
    print(f"PSNR      {psnr_value:6.2f}\n"
          f"SSIM      {ssim_value:6.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", metavar="DIR",
                        help="Path to dataset.")
    parser.add_argument("--model-path", default="", type=str,
                        help="Path to latest checkpoint for model.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    create_folder("benchmarks")

    logger.info("TestEngine:")
    logger.info("\tAPI version .......... 0.3.1")
    logger.info("\tBuild ................ 2021.07.06")

    main(args)

    logger.info("Test dataset performance evaluation completed successfully.")
