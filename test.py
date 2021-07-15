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
# ============================================================================
import logging
import os
from argparse import ArgumentParser

import torch
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from srgan_pytorch.model import generator
from srgan_pytorch.utils import create_folder

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--model-path", default="", type=str,
                    help="Path to latest checkpoint for model.")
parser.add_argument("--cuda", dest="cuda", action="store_true",
                    help="Enables cuda.")
args = parser.parse_args()

# Set whether to use CUDA.
device = torch.device("cuda:0" if args.cuda else "cpu")


def sr(model, lr_filename, sr_filename):
    r""" Turn low resolution into super resolution.

    Args:
        model (torch.nn.Module): SR model.
        lr_filename (str): Low resolution image address.
        sr_filename (srt): Super resolution image address.
    """
    with torch.no_grad():
        lr = Image.open(lr_filename).convert("RGB")
        lr_tensor = ToTensor()(lr).unsqueeze(0).to(device)
        sr_tensor = model(lr_tensor)
        save_image(sr_tensor.detach(), sr_filename, normalize=True)


def iqa(sr_filename, hr_filename):
    r""" Image quality evaluation function.

    Args:
        sr_filename (str): Super resolution image address.
        hr_filename (srt): High resolution image address.

    Returns:
        PSNR value(float), SSIM value(float).
    """
    sr_image = imread(sr_filename)
    hr_image = imread(hr_filename)

    # Delete 4 pixels around the image to facilitate PSNR calculation.
    sr_image = sr_image[4:-4, 4:-4, ...]
    hr_image = hr_image[4:-4, 4:-4, ...]

    # Calculate the Y channel of the image. Use the Y channel to calculate PSNR
    # and SSIM instead of using RGB three channels.
    sr_image = sr_image / 255.0
    hr_image = hr_image / 255.0
    sr_image = rgb2ycbcr(sr_image)[:, :, 0:1]
    hr_image = rgb2ycbcr(hr_image)[:, :, 0:1]
    # Because rgb2ycbcr() outputs a floating point type and the range is [0, 255],
    # it needs to be renormalized to [0, 1].
    sr_image = sr_image / 255.0
    hr_image = hr_image / 255.0

    psnr = peak_signal_noise_ratio(sr_image, hr_image)
    ssim = structural_similarity(sr_image,
                                 hr_image,
                                 win_size=11,
                                 gaussian_weights=True,
                                 multichannel=True,
                                 data_range=1.0,
                                 K1=0.01,
                                 K2=0.03,
                                 sigma=1.5)

    return psnr, ssim


def main():
    # Initialize the image quality evaluation index.
    avg_psnr = 0.0
    avg_ssim = 0.0

    # Load model and weights.
    model = generator(args.pretrained).to(device).eval()
    if args.model_path != "":
        logger.info(f"Loading weights from `{args.model_path}`.")
        model.load_state_dict(torch.load(args.model_path))

    # Get test image file index.
    filenames = os.listdir(os.path.join("data", "Set5", "LRbicx4"))

    for index in range(len(filenames)):
        lr_filename = os.path.join("data", "Set5", "LRbicx4", filenames[index])
        sr_filename = os.path.join("tests", "Set5", filenames[index])
        hr_filename = os.path.join("data", "Set5", "GTmod12", filenames[index])

        # Process low-resolution images into super-resolution images.
        sr(model, lr_filename, sr_filename)

        # Test the image quality difference between the super-resolution image
        # and the original high-resolution image.
        psnr, ssim = iqa(sr_filename, hr_filename)
        avg_psnr += psnr
        avg_ssim += ssim

    # Calculate the average index value of the image quality of the test dataset.
    avg_psnr = avg_psnr / len(filenames)
    avg_ssim = avg_ssim / len(filenames)

    logger.info(f"Avg PSNR: {avg_psnr:.2f}dB.")
    logger.info(f"Avg SSIM: {avg_ssim:.4f}.")


if __name__ == "__main__":
    create_folder("tests")
    create_folder(os.path.join("tests", "Set5"))

    logger.info("TrainEngine:")
    logger.info("\tAPI version .......... 0.4.0")
    logger.info("\tBuild ................ 2021.07.09")

    main()

    logger.info("All training has been completed successfully.\n")
