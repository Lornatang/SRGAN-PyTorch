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
import warnings
from typing import Tuple

import cv2
import numpy as np
import skimage.color
import skimage.io
import skimage.metrics
import torchvision.utils
from PIL import Image
from skimage import img_as_ubyte

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


def cal_spectrum(sr_image, hr_image) -> float:
    # Scikit-image format is converted to OpenCV format.
    sr = img_as_ubyte(sr_image)
    hr = img_as_ubyte(hr_image)
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2GRAY)
    hr = cv2.cvtColor(hr, cv2.COLOR_RGB2GRAY)

    n = sr.shape[0]

    # 1. Calculate the image gray histogram horizontally.
    all_hist_sr = []
    all_hist_hr = []
    for hist_height in range(n):
        # Calculate each line of gray histogram.
        hist_sr = cv2.calcHist([sr[hist_height, :]], [0], None, [n], [0, 255])
        hist_hr = cv2.calcHist([hr[hist_height, :]], [0], None, [n], [0, 255])
        all_hist_sr.append(hist_sr)
        all_hist_hr.append(hist_hr)

    # 2. 1D Fourier transform (cut one-sided data).
    all_spectrum_sr = []
    all_spectrum_hr = []
    for index in range(n):
        # Fast Fourier Transform
        fft_sr = np.fft.fft(all_hist_sr[index])
        fft_hr = np.fft.fft(all_hist_hr[index])
        # Take the absolute value of the complex number, that is, the modulus of the complex number (bilateral spectrum).
        spectrum_sr = np.abs(fft_sr)
        spectrum_hr = np.abs(fft_hr)
        # Due to symmetry, only half of the interval (one-sided spectrum) is taken.
        spectrum_sr = spectrum_sr[range(n // 2)]
        spectrum_hr = spectrum_hr[range(n // 2)]
        all_spectrum_sr.append(spectrum_sr)
        all_spectrum_hr.append(spectrum_hr)

    # 3. Find the average of the spectrum.
    avg_spectrum_sr = []
    avg_spectrum_hr = []
    # Traverse the spectrum values in the range of 0~(N//2) in N spectra.
    for spectrum in range(n // 2):
        total_spectrum_sr = 0
        total_spectrum_hr = 0
        for index in range(n):
            total_spectrum_sr += all_spectrum_sr[index][spectrum]
            total_spectrum_hr += all_spectrum_hr[index][spectrum]
        avg_spectrum_sr.append(total_spectrum_sr / n)
        avg_spectrum_hr.append(total_spectrum_hr / n)

    # 4. Use the formula to find the difference.
    diff = 0.
    for index in range(n // 2):
        diff += (avg_spectrum_hr[index] - avg_spectrum_sr[index]) ** 2

    spectrum = float(np.sqrt(diff / (n / 2)))

    return spectrum


def image_quality_assessment(sr_path: str, hr_path: str) -> Tuple[float, float, float]:
    """Image quality evaluation function.

    Args:
        sr_path (str): Super-resolution image address.
        hr_path (srt): High resolution image address.

    Returns:
        PSNR value(float), SSIM value(float), Spectrum value(float)
    """
    sr_image = skimage.io.imread(sr_path)
    hr_image = skimage.io.imread(hr_path)

    if sr_image.shape != hr_image.shape:
        warnings.warn("Image size not equal! Possible errors in the calculation of the spectrum!")
    if sr_image.shape[0] != sr_image.shape[1]:
        warnings.warn("Image width and height is not equal! Possible errors in the calculation of the spectrum!")

    psnr, ssim = cal_psnr_and_ssim(sr_image, hr_image)
    spectrum = cal_spectrum(sr_image, hr_image)

    return psnr, ssim, spectrum


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
    total_spectrum = 0.0

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
        psnr, ssim, spectrum = image_quality_assessment(sr_path, hr_path)
        total_psnr += psnr
        total_ssim += ssim
        total_spectrum += spectrum

    print(f"PSNR:    {total_psnr / total_files:.2f}.\n"
          f"SSIM:    {total_ssim / total_files:.4f}.\n"
          f"Spectrum {total_spectrum / total_files:.4f}")


if __name__ == "__main__":
    main()
