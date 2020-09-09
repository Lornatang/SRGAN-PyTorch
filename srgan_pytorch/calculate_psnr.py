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
import math

from .calculate_mse import cal_mse


def cal_psnr(src_img, dst_img):
    r"""Python simply calculates the maximum signal noise ratio.

    Args:
        src_img (np.array): Prediction image format read by OpenCV.
        dst_img (np.array): Target image format read by OpenCV.

    ..math:
        10 \cdot \log _{10}\left(\frac{MAX_{I}^{2}}{MSE}\right)

    Returns:
        Maximum signal to noise ratio between two images.
    """
    mse = cal_mse(src_img, dst_img)

    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)
