# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import warnings

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import imgproc

__all__ = [
    "psnr", "ssim",
    "PSNR", "SSIM",
]


# The following is the implementation of IQA method in Python, using CPU as processing device
def _check_image(raw_image: np.ndarray, dst_image: np.ndarray):
    """Check whether the size and type of the two images are the same

    Args:
        raw_image (np.ndarray): image data to be compared, BGR format, data range [0, 255]
        dst_image (np.ndarray): reference image data, BGR format, data range [0, 255]

    """
    # check image scale
    assert raw_image.shape == dst_image.shape, \
        f"Supplied images have different sizes {str(raw_image.shape)} and {str(dst_image.shape)}"

    # check image type
    if raw_image.dtype != dst_image.dtype:
        warnings.warn(f"Supplied images have different dtypes{str(raw_image.shape)} and {str(dst_image.shape)}")


def psnr(raw_image: np.ndarray, dst_image: np.ndarray, crop_border: int, only_test_y_channel: bool) -> float:
    """Python implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Args:
        raw_image (np.ndarray): image data to be compared, BGR format, data range [0, 255]
        dst_image (np.ndarray): reference image data, BGR format, data range [0, 255]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image.

    Returns:
        psnr_metrics (np.float64): PSNR metrics

    """
    # Check if two images are similar in scale and type
    _check_image(raw_image, dst_image)

    # crop border pixels
    if crop_border > 0:
        raw_image = raw_image[crop_border:-crop_border, crop_border:-crop_border, ...]
        dst_image = dst_image[crop_border:-crop_border, crop_border:-crop_border, ...]

    # If you only test the Y channel, you need to extract the Y channel data of the YCbCr channel data separately
    if only_test_y_channel:
        raw_image = imgproc.expand_y(raw_image)
        dst_image = imgproc.expand_y(dst_image)

    # Convert data type to numpy.float64 bit
    raw_image = raw_image.astype(np.float64)
    dst_image = dst_image.astype(np.float64)

    psnr_metrics = 10 * np.log10((255.0 ** 2) / np.mean((raw_image - dst_image) ** 2) + 1e-8)

    return psnr_metrics


def _ssim(raw_image: np.ndarray, dst_image: np.ndarray) -> float:
    """Python implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_image (np.ndarray): The image data to be compared, in BGR format, the data range is [0, 255]
        dst_image (np.ndarray): reference image data, BGR format, data range is [0, 255]

    Returns:
        ssim_metrics (float): SSIM metrics for single channel

    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel_window = np.outer(kernel, kernel.transpose())

    raw_mean = cv2.filter2D(raw_image, -1, kernel_window)[5:-5, 5:-5]
    dst_mean = cv2.filter2D(dst_image, -1, kernel_window)[5:-5, 5:-5]
    raw_mean_square = raw_mean ** 2
    dst_mean_square = dst_mean ** 2
    raw_dst_mean = raw_mean * dst_mean
    raw_variance = cv2.filter2D(raw_image ** 2, -1, kernel_window)[5:-5, 5:-5] - raw_mean_square
    dst_variance = cv2.filter2D(dst_image ** 2, -1, kernel_window)[5:-5, 5:-5] - dst_mean_square
    raw_dst_covariance = cv2.filter2D(raw_image * dst_image, -1, kernel_window)[5:-5, 5:-5] - raw_dst_mean

    ssim_molecular = (2 * raw_dst_mean + c1) * (2 * raw_dst_covariance + c2)
    ssim_denominator = (raw_mean_square + dst_mean_square + c1) * (raw_variance + dst_variance + c2)

    ssim_metrics = ssim_molecular / ssim_denominator
    ssim_metrics = np.mean(ssim_metrics)

    return ssim_metrics


def ssim(raw_image: np.ndarray, dst_image: np.ndarray, crop_border: int, only_test_y_channel: bool) -> float:
    """Python implements the SSIM (Structural Similarity) function, which calculates single/multi-channel data

    Args:
        raw_image (np.ndarray): The image data to be compared, in BGR format, the data range is [0, 255]
        dst_image (np.ndarray): reference image data, BGR format, data range is [0, 255]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image

    Returns:
        ssim_metrics (float): SSIM metrics for single channel

    """
    # Check if two images are similar in scale and type
    _check_image(raw_image, dst_image)

    # crop border pixels
    if crop_border > 0:
        raw_image = raw_image[crop_border:-crop_border, crop_border:-crop_border, ...]
        dst_image = dst_image[crop_border:-crop_border, crop_border:-crop_border, ...]

    # If you only test the Y channel, you need to extract the Y channel data of the YCbCr channel data separately
    if only_test_y_channel:
        raw_image = imgproc.expand_y(raw_image)
        dst_image = imgproc.expand_y(dst_image)

    # Convert data type to numpy.float64 bit
    raw_image = raw_image.astype(np.float64)
    dst_image = dst_image.astype(np.float64)

    channels_ssim_metrics = []
    for channel in range(raw_image.shape[2]):
        ssim_metrics = _ssim(raw_image[..., channel], dst_image[..., channel])
        channels_ssim_metrics.append(ssim_metrics)
    ssim_metrics = np.mean(np.asarray(channels_ssim_metrics))

    return ssim_metrics


# The following is the IQA method implemented by PyTorch, using CUDA as the processing device
def _check_tensor_shape(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    """Check if the dimensions of the two tensors are the same

    Args:
        raw_tensor (np.ndarray or torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (np.ndarray or torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]

    """
    # Check if tensor scales are consistent
    assert raw_tensor.shape == dst_tensor.shape, \
        f"Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"


def _psnr_torch(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor, crop_border: int,
                only_test_y_channel: bool) -> float:
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Args:
        raw_tensor (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image

    Returns:
        psnr_metrics (torch.Tensor): PSNR metrics

    """
    # Check if two tensor scales are similar
    _check_tensor_shape(raw_tensor, dst_tensor)

    # crop border pixels
    if crop_border > 0:
        raw_tensor = raw_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]

    # Convert RGB tensor data to YCbCr tensor, and extract only Y channel data
    if only_test_y_channel:
        raw_tensor = imgproc.rgb2ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = imgproc.rgb2ycbcr_torch(dst_tensor, only_use_y_channel=True)

    # Convert data type to torch.float64 bit
    raw_tensor = raw_tensor.to(torch.float64)
    dst_tensor = dst_tensor.to(torch.float64)

    mse_value = torch.mean((raw_tensor * 255.0 - dst_tensor * 255.0) ** 2 + 1e-8, dim=[1, 2, 3])
    psnr_metrics = 10 * torch.log10_(255.0 ** 2 / mse_value)

    return psnr_metrics


class PSNR(nn.Module):
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Attributes:
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image

    Returns:
        psnr_metrics (torch.Tensor): PSNR metrics

    """

    def __init__(self, crop_border: int, only_test_y_channel: bool) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> torch.Tensor:
        psnr_metrics = _psnr_torch(raw_tensor, dst_tensor, self.crop_border, self.only_test_y_channel)

        return psnr_metrics


def _ssim_torch(raw_tensor: torch.Tensor,
                dst_tensor: torch.Tensor,
                window_size: int,
                gaussian_kernel_window: np.ndarray) -> float:
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_tensor (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 255]
        dst_tensor (torch.Tensor): reference image tensorflow, RGB format, data range [0, 255]
        window_size (int): Gaussian filter size
        gaussian_kernel_window (np.ndarray): Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    gaussian_kernel_window = torch.from_numpy(gaussian_kernel_window).view(1, 1, window_size, window_size)
    gaussian_kernel_window = gaussian_kernel_window.expand(raw_tensor.size(1), 1, window_size, window_size)
    gaussian_kernel_window = gaussian_kernel_window.to(device=raw_tensor.device, dtype=raw_tensor.dtype)

    raw_mean = F.conv2d(raw_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=raw_tensor.shape[1])
    dst_mean = F.conv2d(dst_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=dst_tensor.shape[1])
    raw_mean_square = raw_mean ** 2
    dst_mean_square = dst_mean ** 2
    raw_dst_mean = raw_mean * dst_mean
    raw_variance = F.conv2d(raw_tensor * raw_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0),
                            groups=raw_tensor.shape[1]) - raw_mean_square
    dst_variance = F.conv2d(dst_tensor * dst_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0),
                            groups=raw_tensor.shape[1]) - dst_mean_square
    raw_dst_covariance = F.conv2d(raw_tensor * dst_tensor, gaussian_kernel_window, stride=1, padding=(0, 0),
                                  groups=raw_tensor.shape[1]) - raw_dst_mean

    ssim_molecular = (2 * raw_dst_mean + c1) * (2 * raw_dst_covariance + c2)
    ssim_denominator = (raw_mean_square + dst_mean_square + c1) * (raw_variance + dst_variance + c2)

    ssim_metrics = ssim_molecular / ssim_denominator
    ssim_metrics = torch.mean(ssim_metrics, [1, 2, 3])

    return ssim_metrics


def _ssim_single_torch(raw_tensor: torch.Tensor,
                       dst_tensor: torch.Tensor,
                       crop_border: int,
                       only_test_y_channel: bool,
                       window_size: int,
                       gaussian_kernel_window: torch.Tensor) -> torch.Tensor:
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_tensor (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        window_size (int): Gaussian filter size
        gaussian_kernel_window (torch.Tensor): Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """
    # Check if two tensor scales are similar
    _check_tensor_shape(raw_tensor, dst_tensor)

    # crop border pixels
    if crop_border > 0:
        raw_tensor = raw_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]

    # Convert RGB tensor data to YCbCr tensor, and extract only Y channel data
    if only_test_y_channel:
        raw_tensor = imgproc.rgb2ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = imgproc.rgb2ycbcr_torch(dst_tensor, only_use_y_channel=True)

    # Convert data type to torch.float64 bit
    raw_tensor = raw_tensor.to(torch.float64)
    dst_tensor = dst_tensor.to(torch.float64)

    ssim_metrics = _ssim_torch(raw_tensor * 255.0, dst_tensor * 255.0, window_size, gaussian_kernel_window)

    return ssim_metrics


class SSIM(nn.Module):
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        crop_border (int): crop border a few pixels
        only_only_test_y_channel (bool): Whether to test only the Y channel of the image
        window_size (int): Gaussian filter size
        gaussian_sigma (float): sigma parameter in Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """

    def __init__(self, crop_border: int,
                 only_only_test_y_channel: bool,
                 window_size: int = 11,
                 gaussian_sigma: float = 1.5) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_only_test_y_channel
        self.window_size = window_size

        gaussian_kernel = cv2.getGaussianKernel(window_size, gaussian_sigma)
        self.gaussian_kernel_window = np.outer(gaussian_kernel, gaussian_kernel.transpose())

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> torch.Tensor:
        ssim_metrics = _ssim_single_torch(raw_tensor,
                                          dst_tensor,
                                          self.crop_border,
                                          self.only_test_y_channel,
                                          self.window_size,
                                          self.gaussian_kernel_window)

        return ssim_metrics
