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

import skimage.color
import skimage.io
import skimage.metrics
import torchvision.utils
from PIL import Image

from config import *
from imgproc import *


def image_quality_assessment(sr_path: str, hr_path: str) -> Tuple[float, float]:
    sr_image = skimage.io.imread(sr_path)
    hr_image = skimage.io.imread(hr_path)
    
    # Test for Y channel.
    sr_image = normalize(sr_image)
    hr_image = normalize(hr_image)
    sr_image = skimage.color.rgb2ycbcr(sr_image)[:, :, 0:1]
    hr_image = skimage.color.rgb2ycbcr(hr_image)[:, :, 0:1]
    sr_image = normalize(sr_image)
    hr_image = normalize(hr_image)

    psnr = skimage.metrics.peak_signal_noise_ratio(sr_image, hr_image, data_range=1.0)
    ssim = skimage.metrics.structural_similarity(sr_image,
                                                 hr_image,
                                                 win_size=11,
                                                 gaussian_weights=True,
                                                 multichannel=True,
                                                 data_range=1.0,
                                                 K1=0.01,
                                                 K2=0.03,
                                                 sigma=1.5)

    return psnr, ssim


def main() -> None:
    net.half()
    net.eval()

    total_psnr = 0.0
    total_ssim = 0.0

    filenames = os.listdir(lr_dir)
    total_files = len(filenames)

    for index in range(total_files):
        lr_path = os.path.join(lr_dir, filenames[index])
        sr_path = os.path.join(sr_dir, filenames[index])
        hr_path = os.path.join(hr_dir, filenames[index])
        # LR to SR.
        lr_tensor = image2tensor(Image.open(lr_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = net(lr_tensor.half())
            torchvision.utils.save_image(sr_tensor, sr_path)
            
        # Test PSNR and SSIM.
        print(f"Test `{lr_path}`.")
        psnr, ssim = image_quality_assessment(sr_path, hr_path)
        total_psnr += psnr
        total_ssim += ssim
    
    print(f"PSNR: {total_psnr / total_files:.2f}\n"
          f"SSIM: {total_ssim / total_files:.4f}.")


if __name__ == "__main__":
    main()
