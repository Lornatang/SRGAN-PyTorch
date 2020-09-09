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
import cv2
import numpy as np


def cal_mse(src_img, dst_img):
    r"""Python calculates mean square error.

        Args:
            src_img (np.array): Prediction image format read by OpenCV.
            dst_img (np.array): Target image format read by OpenCV.

        ..math:
            \frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}

        Returns:
            Mean square error.
        """

    return np.mean((src_img - dst_img) ** 2)


def cal_rmse(src_img, dst_img):
    r"""Python calculates root mean square error.

    Args:
        src_img (np.array): Prediction image format read by OpenCV.
        dst_img (np.array): Target image format read by OpenCV.

    ..math:
        \sqrt{\operatorname{MSE}(\hat{\theta})}

    Returns:
        Root mean square error.
    """

    return np.sqrt(cal_mse(src_img, dst_img))
