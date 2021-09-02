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

# ==============================================================================
# File description: Realize the verification function after model training.
# ==============================================================================
import shutil
from typing import Tuple

import skimage.color
import skimage.io
import skimage.metrics
import torchvision.utils
from PIL import Image

from config import *
from imgproc import *


def cal_psnr_and_ssim(sr_image, hr_image) -> Tuple[float, float]:
    # Test for Y channel.
    sr = normalize(sr_image)
    hr = normalize(hr_image)
    sr = skimage.color.rgb2ycbcr(sr)[:, :, 0:1]
    hr = skimage.color.rgb2ycbcr(hr)[:, :, 0:1]
    sr = normalize(sr)
    hr = normalize(hr)

    psnr = skimage.metrics.peak_signal_noise_ratio(sr, hr, data_range=1.0)
    ssim = skimage.metrics.structural_similarity(sr,
                                                 hr,
                                                 win_size=11,
                                                 gaussian_weights=True,
                                                 multichannel=True,
                                                 data_range=1.0,
                                                 K1=0.01,
                                                 K2=0.03,
                                                 sigma=1.5)
    return psnr, ssim


def image_quality_assessment(sr_path: str, hr_path: str) -> Tuple[float, float]:
    """Image quality evaluation function.

    Args:
        sr_path (str): Super-resolution image address.
        hr_path (srt): High resolution image address.

    Returns:
        PSNR value(float), SSIM value(float), Spectrum value(float)
    """
    sr_image = skimage.io.imread(sr_path)
    hr_image = skimage.io.imread(hr_path)

    psnr, ssim = cal_psnr_and_ssim(sr_image, hr_image)

    return psnr, ssim


def main() -> None:
    # Create a super-resolution experiment result folder.
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    # Load model weights.
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the image evaluation index.
    total_psnr = 0.0
    total_ssim = 0.0

    # Get a list of test image file names.
    filenames = os.listdir(lr_dir)
    # Get the number of test image files.
    total_files = len(filenames)

    for index in range(total_files):
        lr_path = os.path.join(lr_dir, filenames[index])
        sr_path = os.path.join(sr_dir, filenames[index])
        hr_path = os.path.join(hr_dir, filenames[index])
        # Process low-resolution images into super-resolution images.
        lr = Image.open(lr_path).convert("RGB")
        lr_tensor = image2tensor(lr).unsqueeze(0)
        lr_tensor = lr_tensor.half()
        lr_tensor = lr_tensor.to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
            torchvision.utils.save_image(sr_tensor, sr_path, normalize=True)

        # Test the image quality difference between the super-resolution image
        # and the original high-resolution image.
        print(f"Test `{os.path.abspath(lr_path)}`.")
        psnr, ssim = image_quality_assessment(sr_path, hr_path)
        total_psnr += psnr
        total_ssim += ssim

    print(f"PSNR: {total_psnr / total_files:.2f}.\n"
          f"SSIM: {total_ssim / total_files:.4f}.")


if __name__ == "__main__":
    main()
