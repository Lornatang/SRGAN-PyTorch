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
import collections.abc
import math
import typing
import warnings
from itertools import repeat
from typing import Any

import cv2
import numpy as np
import torch
from numpy import ndarray
from scipy.io import loadmat
from scipy.ndimage.filters import convolve
from scipy.special import gamma
from torch import nn
from torch.nn import functional as F

from imgproc import image_resize, expand_y, bgr_to_ycbcr, rgb_to_ycbcr_torch

__all__ = [
    "psnr", "ssim", "niqe",
    "PSNR", "SSIM", "NIQE",
]

_I = typing.Optional[int]
_D = typing.Optional[torch.dtype]


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
        raw_image = expand_y(raw_image)
        dst_image = expand_y(dst_image)

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
    ssim_metrics = float(np.mean(ssim_metrics))

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
        raw_image = expand_y(raw_image)
        dst_image = expand_y(dst_image)

    # Convert data type to numpy.float64 bit
    raw_image = raw_image.astype(np.float64)
    dst_image = dst_image.astype(np.float64)

    channels_ssim_metrics = []
    for channel in range(raw_image.shape[2]):
        ssim_metrics = _ssim(raw_image[..., channel], dst_image[..., channel])
        channels_ssim_metrics.append(ssim_metrics)
    ssim_metrics = np.mean(np.asarray(channels_ssim_metrics))

    return float(ssim_metrics)


def _estimate_aggd_parameters(vector: np.ndarray) -> [np.ndarray, float, float]:
    """Python implements the NIQE (Natural Image Quality Evaluator) function,
    This function is used to estimate an asymmetric generalized Gaussian distribution

    Reference papers:
        `Estimation of shape parameter for generalized Gaussian distributions in subband decompositions of video`

    Args:
        vector (np.ndarray): data vector

    Returns:
        aggd_parameters (np.ndarray): asymmetric generalized Gaussian distribution
        left_beta (float): symmetric left data vector variance mean product
        right_beta (float): symmetric right side data vector variance mean product

    """
    # The following is obtained according to the formula and the method provided in the paper on WIki encyclopedia
    vector = vector.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(vector[vector < 0] ** 2))
    right_std = np.sqrt(np.mean(vector[vector > 0] ** 2))
    gamma_hat = left_std / right_std
    rhat = (np.mean(np.abs(vector))) ** 2 / np.mean(vector ** 2)
    rhat_norm = (rhat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2)
    array_position = np.argmin((r_gam - rhat_norm) ** 2)

    aggd_parameters = gam[array_position]
    left_beta = left_std * np.sqrt(gamma(1 / aggd_parameters) / gamma(3 / aggd_parameters))
    right_beta = right_std * np.sqrt(gamma(1 / aggd_parameters) / gamma(3 / aggd_parameters))

    return aggd_parameters, left_beta, right_beta


def _get_mscn_feature(image: np.ndarray) -> list[float | Any]:
    """Python implements the NIQE (Natural Image Quality Evaluator) function,
    This function is used to calculate the MSCN feature map

    Reference papers:
        `Estimation of shape parameter for generalized Gaussian distributions in subband decompositions of video`

    Args:
        image (np.ndarray): Grayscale image of MSCN feature to be calculated, BGR format, data range is [0, 255]

    Returns:
        mscn_feature (np.ndarray): MSCN feature map of the image

    """
    mscn_feature = []
    # Calculate the asymmetric generalized Gaussian distribution
    aggd_parameters, left_beta, right_beta = _estimate_aggd_parameters(image)
    mscn_feature.extend([aggd_parameters, (left_beta + right_beta) / 2])

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(image, shifts[i], axis=(0, 1))
        # Calculate the asymmetric generalized Gaussian distribution
        aggd_parameters, left_beta, right_beta = _estimate_aggd_parameters(image * shifted_block)
        mean = (right_beta - left_beta) * (gamma(2 / aggd_parameters) / gamma(1 / aggd_parameters))
        mscn_feature.extend([aggd_parameters, mean, left_beta, right_beta])

    return mscn_feature


def _fit_mscn_ipac(image: np.ndarray,
                   mu_pris_param: np.ndarray,
                   cov_pris_param: np.ndarray,
                   gaussian_window: np.ndarray,
                   block_size_height: int,
                   block_size_width: int) -> float:
    """Python implements the NIQE (Natural Image Quality Evaluator) function,
    This function is used to fit the inner product of adjacent coefficients of MSCN

    Reference papers:
        `Estimation of shape parameter for generalized Gaussian distributions in subband decompositions of video`

    Args:
        image (np.ndarray): The image data of the NIQE to be tested, in BGR format, the data range is [0, 255]
        mu_pris_param (np.ndarray): Mean of predefined multivariate Gaussians, model computed on original dataset.
        cov_pris_param (np.ndarray): Covariance of predefined multivariate Gaussian model computed on original dataset.
        gaussian_window (np.ndarray): 7x7 Gaussian window for smoothing the image
        block_size_height (int): the height of the block into which the image is divided
        block_size_width (int): The width of the block into which the image is divided

    Returns:
        niqe_metric (np.ndarray): NIQE score

    """
    image_height, image_width = image.shape
    num_block_height = math.floor(image_height / block_size_height)
    num_block_width = math.floor(image_width / block_size_width)
    image = image[0:num_block_height * block_size_height, 0:num_block_width * block_size_width]

    features_parameters = []
    for scale in (1, 2):
        mu = convolve(image, gaussian_window, mode="nearest")
        sigma = np.sqrt(np.abs(convolve(np.square(image), gaussian_window, mode="nearest") - np.square(mu)))
        image_norm = (image - mu) / (sigma + 1)

        feature = []
        for idx_w in range(num_block_width):
            for idx_h in range(num_block_height):
                vector = image_norm[
                         idx_h * block_size_height // scale:(idx_h + 1) * block_size_height // scale,
                         idx_w * block_size_width // scale:(idx_w + 1) * block_size_width // scale]
                feature.append(_get_mscn_feature(vector))

        features_parameters.append(np.array(feature))

        if scale == 1:
            image = image_resize(image / 255., scale_factor=0.5, antialiasing=True)
            image = image * 255.

    features_parameters = np.concatenate(features_parameters, axis=1)

    # Fitting a multivariate Gaussian kernel model to distorted patch features
    mu_distparam = np.nanmean(features_parameters, axis=0)
    distparam_no_nan = features_parameters[~np.isnan(features_parameters).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    niqe_metric = np.matmul(np.matmul((mu_pris_param - mu_distparam), invcov_param),
                            np.transpose((mu_pris_param - mu_distparam)))

    niqe_metric = np.sqrt(niqe_metric)
    niqe_metric = float(np.squeeze(niqe_metric))

    return niqe_metric


def niqe(image: np.ndarray,
         crop_border: int,
         niqe_model_path: str,
         block_size_height: int = 96,
         block_size_width: int = 96) -> float:
    """Python implements the NIQE (Natural Image Quality Evaluator) function,
    This function computes single/multi-channel data

    Args:
        image (np.ndarray): The image data to be compared, in BGR format, the data range is [0, 255]
        crop_border (int): crop border a few pixels
        niqe_model_path: NIQE estimator model address
        block_size_height (int): The height of the block the image is divided into. Default: 96
        block_size_width (int): The width of the block the image is divided into. Default: 96

    Returns:
        niqe_metrics (float): NIQE indicator under single channel

    """
    # crop border pixels
    if crop_border > 0:
        image = image[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Defining the NIQE Feature Extraction Model
    niqe_model = np.load(niqe_model_path)

    mu_pris_param = niqe_model["mu_pris_param"]
    cov_pris_param = niqe_model["cov_pris_param"]
    gaussian_window = niqe_model["gaussian_window"]

    # NIQE only tests on Y channel images and needs to convert the images
    y_image = bgr_to_ycbcr(image, only_use_y_channel=True)

    # Convert data type to numpy.float64 bit
    y_image = y_image.astype(np.float64)

    niqe_metric = _fit_mscn_ipac(y_image,
                                 mu_pris_param,
                                 cov_pris_param,
                                 gaussian_window,
                                 block_size_height,
                                 block_size_width)

    return niqe_metric


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
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, only_use_y_channel=True)

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

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> float:
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
    ssim_metrics = torch.mean(ssim_metrics, [1, 2, 3]).float()

    return ssim_metrics


def _ssim_single_torch(raw_tensor: torch.Tensor,
                       dst_tensor: torch.Tensor,
                       crop_border: int,
                       only_test_y_channel: bool,
                       window_size: int,
                       gaussian_kernel_window: ndarray) -> float:
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_tensor (Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensorflow, RGB format, data range [0, 1]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        window_size (int): Gaussian filter size
        gaussian_kernel_window (ndarray): Gaussian filter

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
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, only_use_y_channel=True)

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

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> float:
        ssim_metrics = _ssim_single_torch(raw_tensor,
                                          dst_tensor,
                                          self.crop_border,
                                          self.only_test_y_channel,
                                          self.window_size,
                                          self.gaussian_kernel_window)

        return ssim_metrics


def _fspecial_gaussian_torch(window_size: int, sigma: float, channels: int):
    """PyTorch implements the fspecial_gaussian() function in MATLAB

    Args:
        window_size (int): Gaussian filter size
        sigma (float): sigma parameter in Gaussian filter
        channels (int): number of input image channels

    Returns:
        gaussian_kernel_window (torch.Tensor): Gaussian filter in Tensor format

    """
    if type(window_size) is int:
        shape = (window_size, window_size)
    else:
        shape = window_size
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()

    if sumh != 0:
        h /= sumh

    gaussian_kernel_window = torch.from_numpy(h).float().repeat(channels, 1, 1, 1)

    return gaussian_kernel_window


def _to_tuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def _excact_padding_2d(tensor: torch.Tensor,
                       kernel: torch.Tensor | tuple,
                       stride: int = 1,
                       dilation: int = 1,
                       mode: str = "same") -> torch.Tensor:
    assert len(tensor.shape) == 4, f"Only support 4D tensor input, but got {tensor.shape}"
    kernel = _to_tuple(2)(kernel)
    stride = _to_tuple(2)(stride)
    dilation = _to_tuple(2)(dilation)
    b, c, h, w = tensor.shape
    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    pad_l, pad_r, pad_t, pad_b = (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2)

    mode = mode if mode != "same" else "constant"
    if mode != "symmetric":
        tensor = F.pad(tensor, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    elif mode == "symmetric":
        sym_h = torch.flip(tensor, [2])
        sym_w = torch.flip(tensor, [3])
        sym_hw = torch.flip(tensor, [2, 3])

        row1 = torch.cat((sym_hw, sym_h, sym_hw), dim=3)
        row2 = torch.cat((sym_w, tensor, sym_w), dim=3)
        row3 = torch.cat((sym_hw, sym_h, sym_hw), dim=3)

        whole_map = torch.cat((row1, row2, row3), dim=2)

        tensor = whole_map[:, :, h - pad_t:2 * h + pad_b, w - pad_l:2 * w + pad_r, ]

    return tensor


class ExactPadding2d(nn.Module):
    r"""This function calculate exact padding values for 4D tensor inputs,
    and support the same padding mode as tensorflow.

    Args:
        kernel (int or tuple): kernel size.
        stride (int or tuple): stride size.
        dilation (int or tuple): dilation size, default with 1.
        mode (srt): padding mode can be ('same', 'symmetric', 'replicate', 'circular')

    """

    def __init__(self, kernel, stride=1, dilation=1, mode="same") -> None:
        super().__init__()
        self.kernel = _to_tuple(2)(kernel)
        self.stride = _to_tuple(2)(stride)
        self.dilation = _to_tuple(2)(dilation)
        self.mode = mode

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return _excact_padding_2d(tensor, self.kernel, self.stride, self.dilation, self.mode)


def _image_filter(tensor: torch.Tensor,
                  weight: torch.Tensor,
                  bias=None,
                  stride: int = 1,
                  padding: str = "same",
                  dilation: int = 1,
                  groups: int = 1):
    """PyTorch implements the imfilter() function in MATLAB

    Args:
        tensor (torch.Tensor): Tensor image data
        weight (torch.Tensor): filter weight
        padding (str): how to pad pixels. Default: ``same``
        dilation (int): convolution dilation scale
        groups (int): number of grouped convolutions

    """
    kernel_size = weight.shape[2:]
    exact_padding_2d = ExactPadding2d(kernel_size, stride, dilation, mode=padding)

    return F.conv2d(exact_padding_2d(tensor), weight, bias, stride, dilation=dilation, groups=groups)


def _reshape_input_torch(tensor: torch.Tensor) -> typing.Tuple[torch.Tensor, _I, _I, int, int]:
    if tensor.dim() == 4:
        b, c, h, w = tensor.size()
    elif tensor.dim() == 3:
        c, h, w = tensor.size()
        b = None
    elif tensor.dim() == 2:
        h, w = tensor.size()
        b = c = None
    else:
        raise ValueError('{}-dim Tensor is not supported!'.format(tensor.dim()))

    tensor = tensor.view(-1, 1, h, w)
    return tensor, b, c, h, w


def _reshape_output_torch(tensor: torch.Tensor, b: _I, c: _I) -> torch.Tensor:
    rh = tensor.size(-2)
    rw = tensor.size(-1)
    # Back to the original dimension
    if b is not None:
        tensor = tensor.view(b, c, rh, rw)  # 4-dim
    else:
        if c is not None:
            tensor = tensor.view(c, rh, rw)  # 3-dim
        else:
            tensor = tensor.view(rh, rw)  # 2-dim

    return tensor


def _cast_input_torch(tensor: torch.Tensor) -> typing.Tuple[torch.Tensor, _D]:
    if tensor.dtype != torch.float32 or tensor.dtype != torch.float64:
        dtype = tensor.dtype
        tensor = tensor.float()
    else:
        dtype = None

    return tensor, dtype


def _cast_output_torch(tensor: torch.Tensor, dtype: _D) -> torch.Tensor:
    if dtype is not None:
        if not dtype.is_floating_point:
            tensor = tensor.round()
        # To prevent over/underflow when converting types
        if dtype is torch.uint8:
            tensor = tensor.clamp(0, 255)

        tensor = tensor.to(dtype=dtype)

    return tensor


def _cubic_contribution_torch(tensor: torch.Tensor, a: float = -0.5) -> torch.Tensor:
    ax = tensor.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=tensor.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=tensor.dtype)

    cont = cont_01 + cont_12
    return cont


def _gaussian_contribution_torch(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    range_3sigma = (x.abs() <= 3 * sigma + 1)
    # Normalization will be done after
    cont = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    cont = cont * range_3sigma.to(dtype=x.dtype)
    return cont


def _reflect_padding_torch(tensor: torch.Tensor, dim: int, pad_pre: int, pad_post: int) -> torch.Tensor:
    """
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.
    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    """
    b, c, h, w = tensor.size()
    if dim == 2 or dim == -2:
        padding_buffer = tensor.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(tensor)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(tensor[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(tensor[..., -(p + 1), :])
    else:
        padding_buffer = tensor.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(tensor)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(tensor[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(tensor[..., -(p + 1)])

    return padding_buffer


def _padding_torch(tensor: torch.Tensor,
                   dim: int,
                   pad_pre: int,
                   pad_post: int,
                   padding_type: typing.Optional[str] = 'reflect') -> torch.Tensor:
    if padding_type is None:
        return tensor
    elif padding_type == 'reflect':
        x_pad = _reflect_padding_torch(tensor, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))

    return x_pad


def _get_padding_torch(tensor: torch.Tensor, kernel_size: int, x_size: int) -> typing.Tuple[int, int, torch.Tensor]:
    tensor = tensor.long()
    r_min = tensor.min()
    r_max = tensor.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        tensor += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, tensor


def _get_weight_torch(tensor: torch.Tensor,
                      kernel_size: int,
                      kernel: str = "cubic",
                      sigma: float = 2.0,
                      antialiasing_factor: float = 1) -> torch.Tensor:
    buffer_pos = tensor.new_zeros(kernel_size, len(tensor))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(tensor - idx)

    # Expand (downsampling) / Shrink (upsampling) the receptive field.
    buffer_pos *= antialiasing_factor
    if kernel == 'cubic':
        weight = _cubic_contribution_torch(buffer_pos)
    elif kernel == 'gaussian':
        weight = _gaussian_contribution_torch(buffer_pos, sigma=sigma)
    else:
        raise ValueError('{} kernel is not supported!'.format(kernel))

    weight /= weight.sum(dim=0, keepdim=True)
    return weight


def _reshape_tensor_torch(tensor: torch.Tensor, dim: int, kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = tensor.size(-2) - kernel_size + 1
        w_out = tensor.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = tensor.size(-2)
        w_out = tensor.size(-1) - kernel_size + 1

    unfold = F.unfold(tensor, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold


def _resize_1d_torch(tensor: torch.Tensor,
                     dim: int,
                     size: int,
                     scale: float,
                     kernel: str = 'cubic',
                     sigma: float = 2.0,
                     padding_type: str = 'reflect',
                     antialiasing: bool = True) -> torch.Tensor:
    """
    Args:
        tensor (torch.Tensor): A torch.Tensor of dimension (B x C, 1, H, W).
        dim (int):
        scale (float):
        size (int):
    Return:
    """
    # Identity case
    if scale == 1:
        return tensor

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel == 'cubic':
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sizes
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        pos = torch.linspace(
            0,
            size - 1,
            steps=size,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        pos = (pos + 0.5) / scale - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = _get_weight_torch(
            dist,
            kernel_size,
            kernel=kernel,
            sigma=sigma,
            antialiasing_factor=antialiasing_factor,
        )
        pad_pre, pad_post, base = _get_padding_torch(base, kernel_size, tensor.size(dim))

    # To back-propagate through x
    x_pad = _padding_torch(tensor, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = _reshape_tensor_torch(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    tensor = sample * weight
    tensor = tensor.sum(dim=1, keepdim=True)
    return tensor


def _downsampling_2d_torch(tensor: torch.Tensor, k: torch.Tensor, scale: int,
                           padding_type: str = 'reflect') -> torch.Tensor:
    c = tensor.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)

    k = k.to(dtype=tensor.dtype, device=tensor.device)
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e

    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    tensor = _padding_torch(tensor, -2, pad_h, pad_h, padding_type=padding_type)
    tensor = _padding_torch(tensor, -1, pad_w, pad_w, padding_type=padding_type)
    y = F.conv2d(tensor, k, padding=0, stride=scale)
    return y


def _cov_torch(tensor, rowvar=True, bias=False):
    r"""Estimate a covariance matrix (np.cov)
    Ref: https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2)


def _nancov_torch(x):
    r"""Calculate nancov for batched tensor, rows that contains nan value
    will be removed.
    Args:
        x (tensor): (B, row_num, feat_dim)
    Return:
        cov (tensor): (B, feat_dim, feat_dim)
    """
    assert len(x.shape) == 3, f'Shape of input should be (batch_size, row_num, feat_dim), but got {x.shape}'
    b, rownum, feat_dim = x.shape
    nan_mask = torch.isnan(x).any(dim=2, keepdim=True)
    x_no_nan = x.masked_select(~nan_mask).reshape(b, -1, feat_dim)
    cov_x = _cov_torch(x_no_nan, rowvar=False)
    return cov_x


def _nanmean_torch(v, *args, inplace=False, **kwargs):
    r"""nanmean same as matlab function: calculate mean values by removing all nan.
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def _symm_pad_torch(im: torch.Tensor, padding: [int, int, int, int]):
    """Symmetric padding same as tensorflow.
    Ref: https://discuss.pytorch.org/t/symmetric-padding/19866/3
    """
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def _blockproc_torch(x, kernel, fun, border_size=None, pad_partial=False, pad_method='zero'):
    r"""blockproc function like matlab

    Difference:
        - Partial blocks is discarded (if exist) for fast GPU process.

    Args:
        x (tensor): shape (b, c, h, w)
        kernel (int or tuple): block size
        func (function): function to process each block
        border_size (int or tuple): border pixels to each block
        pad_partial: pad partial blocks to make them full-sized, default False
        pad_method: [zero, replicate, symmetric] how to pad partial block when pad_partial is set True

    Return:
        results (tensor): concatenated results of each block

    """
    assert len(x.shape) == 4, f'Shape of input has to be (b, c, h, w) but got {x.shape}'
    kernel = _to_tuple(2)(kernel)
    if pad_partial:
        b, c, h, w = x.shape
        stride = kernel
        h2 = math.ceil(h / stride[0])
        w2 = math.ceil(w / stride[1])
        pad_row = (h2 - 1) * stride[0] + kernel[0] - h
        pad_col = (w2 - 1) * stride[1] + kernel[1] - w
        padding = (0, pad_col, 0, pad_row)
        if pad_method == 'zero':
            x = F.pad(x, padding, mode='constant')
        elif pad_method == 'symmetric':
            x = _symm_pad_torch(x, padding)
        else:
            x = F.pad(x, padding, mode=pad_method)

    if border_size is not None:
        raise NotImplementedError('Blockproc with border is not implemented yet')
    else:
        b, c, h, w = x.shape
        block_size_h, block_size_w = kernel
        num_block_h = math.floor(h / block_size_h)
        num_block_w = math.floor(w / block_size_w)

        # extract blocks in (row, column) manner, i.e., stored with column first
        blocks = F.unfold(x, kernel, stride=kernel)
        blocks = blocks.reshape(b, c, *kernel, num_block_h, num_block_w)
        blocks = blocks.permute(5, 4, 0, 1, 2, 3).reshape(num_block_h * num_block_w * b, c, *kernel)

        results = fun(blocks)
        results = results.reshape(num_block_h * num_block_w, b, *results.shape[1:]).transpose(0, 1)
        return results


def _image_resize_torch(tensor: torch.Tensor,
                        scale_factor: typing.Optional[float] = None,
                        sizes: typing.Optional[typing.Tuple[int, int]] = None,
                        kernel: typing.Union[str, torch.Tensor] = "cubic",
                        sigma: float = 2,
                        rotation_degree: float = 0,
                        padding_type: str = "reflect",
                        antialiasing: bool = True) -> torch.Tensor:
    """
    Args:
        tensor (torch.Tensor):
        scale_factor (float):
        sizes (tuple(int, int)):
        kernel (str, default='cubic'):
        sigma (float, default=2):
        rotation_degree (float, default=0):
        padding_type (str, default='reflect'):
        antialiasing (bool, default=True):
    Return:
        torch.Tensor:
    """
    scales = (scale_factor, scale_factor)

    if scale_factor is None and sizes is None:
        raise ValueError('One of scale or sizes must be specified!')
    if scale_factor is not None and sizes is not None:
        raise ValueError('Please specify scale or sizes to avoid conflict!')

    tensor, b, c, h, w = _reshape_input_torch(tensor)

    if sizes is None and scale_factor is not None:
        '''
        # Check if we can apply the convolution algorithm
        scale_inv = 1 / scale
        if isinstance(kernel, str) and scale_inv.is_integer():
            kernel = discrete_kernel(kernel, scale, antialiasing=antialiasing)
        elif isinstance(kernel, torch.Tensor) and not scale_inv.is_integer():
            raise ValueError(
                'An integer downsampling factor '
                'should be used with a predefined kernel!'
            )
        '''
        # Determine output size
        sizes = (math.ceil(h * scale_factor), math.ceil(w * scale_factor))
        scales = (scale_factor, scale_factor)

    if scale_factor is None and sizes is not None:
        scales = (sizes[0] / h, sizes[1] / w)

    tensor, dtype = _cast_input_torch(tensor)

    if isinstance(kernel, str) and sizes is not None:
        # Core resizing module
        tensor = _resize_1d_torch(
            tensor,
            -2,
            size=sizes[0],
            scale=scales[0],
            kernel=kernel,
            sigma=sigma,
            padding_type=padding_type,
            antialiasing=antialiasing)
        tensor = _resize_1d_torch(
            tensor,
            -1,
            size=sizes[1],
            scale=scales[1],
            kernel=kernel,
            sigma=sigma,
            padding_type=padding_type,
            antialiasing=antialiasing)
    elif isinstance(kernel, torch.Tensor) and scale_factor is not None:
        tensor = _downsampling_2d_torch(tensor, kernel, scale=int(1 / scale_factor))

    tensor = _reshape_output_torch(tensor, b, c)
    tensor = _cast_output_torch(tensor, dtype)
    return tensor


def _estimate_aggd_parameters_torch(tensor: torch.Tensor,
                                    get_sigma: bool) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch implements the BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) function
    This function is used to estimate an asymmetric generalized Gaussian distribution

    Reference papers:
        `No-Reference Image Quality Assessment in the Spatial Domain`
        `Referenceless Image Spatial Quality Evaluation Engine`

    Args:
        tensor (torch.Tensor): data vector
        get_sigma (bool): whether to return the covariance mean

    Returns:
        aggd_parameters (torch.Tensor): asymmetric generalized Gaussian distribution
        left_std (torch.Tensor): symmetric left data vector variance mean
        right_std (torch.Tensor): Symmetric right side data vector variance mean

    """
    # The following is obtained according to the formula and the method provided in the paper on WIki encyclopedia
    aggd = torch.arange(0.2, 10 + 0.001, 0.001).to(tensor)
    r_gam = (2 * torch.lgamma(2. / aggd) - (torch.lgamma(1. / aggd) + torch.lgamma(3. / aggd))).exp()
    r_gam = r_gam.repeat(tensor.size(0), 1)

    mask_left = tensor < 0
    mask_right = tensor > 0
    count_left = mask_left.sum(dim=(-1, -2), dtype=torch.float32)
    count_right = mask_right.sum(dim=(-1, -2), dtype=torch.float32)

    left_std = torch.sqrt_((tensor * mask_left).pow(2).sum(dim=(-1, -2)) / (count_left + 1e-8))
    right_std = torch.sqrt_((tensor * mask_right).pow(2).sum(dim=(-1, -2)) / (count_right + 1e-8))
    gamma_hat = left_std / right_std
    rhat = tensor.abs().mean(dim=(-1, -2)).pow(2) / tensor.pow(2).mean(dim=(-1, -2))
    rhat_norm = (rhat * (gamma_hat.pow(3) + 1) * (gamma_hat + 1)) / (gamma_hat.pow(2) + 1).pow(2)

    array_position = (r_gam - rhat_norm).abs().argmin(dim=-1)
    aggd_parameters = aggd[array_position]

    if get_sigma:
        left_beta = left_std.squeeze(-1) * (
                torch.lgamma(1 / aggd_parameters) - torch.lgamma(3 / aggd_parameters)).exp().sqrt()
        right_beta = right_std.squeeze(-1) * (
                torch.lgamma(1 / aggd_parameters) - torch.lgamma(3 / aggd_parameters)).exp().sqrt()
        return aggd_parameters, left_beta, right_beta

    else:
        left_std = left_std.squeeze_(-1)
        right_std = right_std.squeeze_(-1)
        return aggd_parameters, left_std, right_std


def _get_mscn_feature_torch(tensor: torch.Tensor) -> np.ndarray:
    """PyTorch implements the NIQE (Natural Image Quality Evaluator) function,
    This function is used to calculate the feature map

    Reference papers:
        `Estimation of shape parameter for generalized Gaussian distributions in subband decompositions of video`

    Args:
        tensor (torch.Tensor): The image to be evaluated for NIQE sharpness

    Returns:
        feature (torch.Tensor): image feature map

    """
    batch_size = tensor.shape[0]
    aggd_block = tensor[:, [0]]
    aggd_parameters, left_beta, right_beta = _estimate_aggd_parameters_torch(aggd_block, True)
    feature = [aggd_parameters, (left_beta + right_beta) / 2]

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = torch.roll(aggd_block, shifts[i], dims=(2, 3))
        aggd_parameters, left_beta, right_beta = _estimate_aggd_parameters_torch(aggd_block * shifted_block, True)
        mean = (right_beta - left_beta) * (torch.lgamma(2 / aggd_parameters) - torch.lgamma(1 / aggd_parameters)).exp()
        feature.extend((aggd_parameters, mean, left_beta, right_beta))

    feature = [x.reshape(batch_size, 1) for x in feature]
    feature = torch.cat(feature, dim=-1)

    return feature


def _fit_mscn_ipac_torch(tensor: torch.Tensor,
                         mu_pris_param: torch.Tensor,
                         cov_pris_param: torch.Tensor,
                         block_size_height: int,
                         block_size_width: int,
                         kernel_size: int = 7,
                         kernel_sigma: float = 7. / 6,
                         padding: str = "replicate") -> float:
    """PyTorch implements the NIQE (Natural Image Quality Evaluator) function,
    This function is used to fit the inner product of adjacent coefficients of MSCN

    Reference papers:
        `Estimation of shape parameter for generalized Gaussian distributions in subband decompositions of video`

    Args:
        tensor (torch.Tensor): The image to be evaluated for NIQE sharpness
        mu_pris_param (torch.Tensor): mean of predefined multivariate Gaussians, model computed on original dataset
        cov_pris_param (torch.Tensor): Covariance of predefined multivariate Gaussian model computed on original dataset
        block_size_height (int): the height of the block into which the image is divided
        block_size_width (int): The width of the block into which the image is divided
        kernel_size (int): Gaussian filter size
        kernel_sigma (int): sigma value in Gaussian filter
        padding (str): how to pad pixels. Default: ``replicate``

    Returns:
        niqe_metric (torch.Tensor): NIQE score

    """
    # crop image
    b, c, h, w = tensor.shape
    num_block_height = math.floor(h / block_size_height)
    num_block_width = math.floor(w / block_size_width)
    tensor = tensor[..., 0:num_block_height * block_size_height, 0:num_block_width * block_size_width]

    distparam = []
    for scale in (1, 2):
        kernel = _fspecial_gaussian_torch(kernel_size, kernel_sigma, 1).to(tensor)
        mu = _image_filter(tensor, kernel, padding=padding)
        std = _image_filter(tensor ** 2, kernel, padding=padding)
        sigma = torch.sqrt_((std - mu ** 2).abs() + 1e-8)
        structdis = (tensor - mu) / (sigma + 1)

        distparam.append(_blockproc_torch(structdis,
                                          [block_size_height // scale, block_size_width // scale],
                                          fun=_get_mscn_feature_torch))

        if scale == 1:
            tensor = _image_resize_torch(tensor / 255., scale_factor=0.5, antialiasing=True)
            tensor = tensor * 255.

    distparam = torch.cat(distparam, -1)

    # Fit MVG (Multivariate Gaussian) model to distorted patch features
    mu_distparam = _nanmean_torch(distparam, dim=1)
    cov_distparam = _nancov_torch(distparam)

    invcov_param = torch.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = (mu_pris_param - mu_distparam).unsqueeze(1)
    niqe_metric = torch.bmm(torch.bmm(diff, invcov_param), diff.transpose(1, 2)).squeeze()
    niqe_metric = torch.sqrt(niqe_metric)

    return niqe_metric


def _niqe_torch(tensor: torch.Tensor,
                crop_border: int,
                niqe_model_path: str,
                block_size_height: int = 96,
                block_size_width: int = 96
                ) -> float:
    """PyTorch implements the NIQE (Natural Image Quality Evaluator) function,

    Attributes:
        tensor (torch.Tensor): The image to evaluate the sharpness of the BRISQUE
        crop_border (int): crop border a few pixels
        niqe_model_path (str): NIQE model estimator weight address
        block_size_height (int): The height of the block the image is divided into. Default: 96
        block_size_width (int): The width of the block the image is divided into. Default: 96

    Returns:
        niqe_metrics (torch.Tensor): NIQE metrics

    """
    # crop border pixels
    if crop_border > 0:
        tensor = tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]

    # Load the NIQE feature extraction model
    niqe_model = loadmat(niqe_model_path)

    mu_pris_param = np.ravel(niqe_model["mu_prisparam"])
    cov_pris_param = niqe_model["cov_prisparam"]
    mu_pris_param = torch.from_numpy(mu_pris_param).to(tensor)
    cov_pris_param = torch.from_numpy(cov_pris_param).to(tensor)

    mu_pris_param = mu_pris_param.repeat(tensor.size(0), 1)
    cov_pris_param = cov_pris_param.repeat(tensor.size(0), 1, 1)

    # NIQE only tests on Y channel images and needs to convert the images
    y_tensor = rgb_to_ycbcr_torch(tensor, only_use_y_channel=True)
    y_tensor *= 255.0
    y_tensor = y_tensor.round()

    # Convert data type to torch.float64 bit
    y_tensor = y_tensor.to(torch.float64)

    niqe_metric = _fit_mscn_ipac_torch(y_tensor,
                                       mu_pris_param,
                                       cov_pris_param,
                                       block_size_height,
                                       block_size_width)

    return niqe_metric


class NIQE(nn.Module):
    """PyTorch implements the NIQE (Natural Image Quality Evaluator) function,

    Attributes:
        crop_border (int): crop border a few pixels
        niqe_model_path (str): NIQE model address
        block_size_height (int): The height of the block the image is divided into. Default: 96
        block_size_width (int): The width of the block the image is divided into. Default: 96

    Returns:
        niqe_metrics (torch.Tensor): NIQE metrics

    """

    def __init__(self, crop_border: int,
                 niqe_model_path: str,
                 block_size_height: int = 96,
                 block_size_width: int = 96) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.niqe_model_path = niqe_model_path
        self.block_size_height = block_size_height
        self.block_size_width = block_size_width

    def forward(self, raw_tensor: torch.Tensor) -> float:
        niqe_metrics = _niqe_torch(raw_tensor,
                                   self.crop_border,
                                   self.niqe_model_path,
                                   self.block_size_height,
                                   self.block_size_width)

        return niqe_metrics
