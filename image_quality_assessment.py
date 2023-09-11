# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
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

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch

from imgproc import rgb_to_ycbcr_torch

__all__ = [
    "MSE", "PSNR", "SSIM"
]

_I = typing.Optional[int]
_D = typing.Optional[torch.dtype]


def _to_tuple(dim: int):
    """Convert the input to a tuple


    Args:
        dim (int): the dimension of the input
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, dim))

    return parse


# The following implements the IQA method for PyTorch, using CUDA as the processing device
def _check_tensor_shape(raw_tensor: Tensor, dst_tensor: Tensor):
    """Check if the dimensions of the two tensors are the same

    Args:
        raw_tensor (np.ndarray or Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (np.ndarray or Tensor): reference image tensor flow, RGB format, data range [0, 1]
    """

    # Check if the tensor scale is consistent
    assert raw_tensor.shape == dst_tensor.shape, \
        f"Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"


def _fspecial_gaussian_torch(
        window_size: int,
        sigma: float,
        channels: int = 3,
        filter_type: int = 0,
) -> Tensor:
    """PyTorch implements the fspecial_gaussian() function in MATLAB

    Args:
        window_size (int): Gaussian filter size
        sigma (float): sigma parameter in Gaussian filter
        channels (int): number of image channels, default: ``3``
        filter_type (int): filter type, 0: Gaussian filter, 1: mean filter, default: ``0``

    Returns:
        gaussian_kernel_window (Tensor): Gaussian filter
    """

    # Gaussian filter processing
    if filter_type == 0:
        shape = _to_tuple(2)(window_size)
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        g = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        g[g < np.finfo(g.dtype).eps * g.max()] = 0
        sum_height = g.sum()

        if sum_height != 0:
            g /= sum_height

        g = torch.from_numpy(g).float().repeat(channels, 1, 1, 1)

        return g
    # mean filter processing
    elif filter_type == 1:
        raise NotImplementedError(f"Only support `gaussian filter`, got {filter_type}")


def _reshape_input_torch(tensor: Tensor) -> typing.Tuple[Tensor, _I, _I, int, int]:
    """Reshape the input tensor to 4-dim tensor

    Args:
        tensor (Tensor): shape (b, c, h, w) or (c, h, w) or (h, w)

    Returns:
        tensor (Tensor): shape (b*c, 1, h, w)
    """

    if tensor.dim() == 4:
        b, c, h, w = tensor.size()
    elif tensor.dim() == 3:
        c, h, w = tensor.size()
        b = None
    elif tensor.dim() == 2:
        h, w = tensor.size()
        b = c = None
    else:
        raise ValueError(f"{tensor.dim()}-dim Tensor is not supported!")

    tensor = tensor.view(-1, 1, h, w)

    return tensor, b, c, h, w


def _cubic_contribution_torch(tensor: Tensor, a: float = -0.5) -> Tensor:
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


def _gaussian_contribution_torch(x: Tensor, sigma: float = 2.0) -> Tensor:
    range_3sigma = (x.abs() <= 3 * sigma + 1)
    # Normalization will be done after
    cont = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    cont = cont * range_3sigma.to(dtype=x.dtype)
    return cont


def _reflect_padding_torch(tensor: Tensor, dim: int, pad_pre: int, pad_post: int) -> Tensor:
    """Reflect padding for 2-dim tensor

    Args:
        tensor (Tensor): shape (b, c, h, w)
        dim (int): 2 or -2
        pad_pre (int): padding size before the dim
        pad_post (int): padding size after the dim

    Returns:
        padding_buffer (Tensor): shape (b, c, h + pad_pre + pad_post, w) or (b, c, h, w + pad_pre + pad_post)
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


def _padding_torch(
        tensor: Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int,
        padding_type: typing.Optional[str] = "reflect",
) -> Tensor:
    if padding_type is None:
        return tensor
    elif padding_type == "reflect":
        x_pad = _reflect_padding_torch(tensor, dim, pad_pre, pad_post)
    else:
        raise ValueError(f"{padding_type} padding is not supported!")

    return x_pad


def _get_padding_torch(tensor: Tensor, kernel_size: int, x_size: int) -> typing.Tuple[int, int, Tensor]:
    """Get padding size and padded tensor

    Args:
        tensor (Tensor): shape (b, c, h, w)
        kernel_size (int): kernel size
        x_size (int): input size

    Returns:
        pad_pre (int): padding size before the dim
    """

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


def _get_weight_torch(
        tensor: Tensor,
        kernel_size: int,
        kernel: str = "cubic",
        sigma: float = 2.0,
        antialiasing_factor: float = 1,
) -> Tensor:
    """Get weight for each pixel

    Args:
        tensor (Tensor): shape (b, c, h, w)
        kernel_size (int): kernel size
        kernel (str): kernel type, cubic or gaussian
        sigma (float): sigma for gaussian kernel
        antialiasing_factor (float): antialiasing factor

    Returns:
        weight (Tensor): shape (b, c, k, h, w) or (b, c, h, k, w)
    """

    buffer_pos = tensor.new_zeros(kernel_size, len(tensor))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(tensor - idx)

    # Expand (downsampling) / Shrink (upsampling) the receptive field.
    buffer_pos *= antialiasing_factor
    if kernel == "cubic":
        weight = _cubic_contribution_torch(buffer_pos)
    elif kernel == "gaussian":
        weight = _gaussian_contribution_torch(buffer_pos, sigma=sigma)
    else:
        raise ValueError(f"{kernel} kernel is not supported!")

    weight /= weight.sum(dim=0, keepdim=True)
    return weight


def _reshape_tensor_torch(tensor: Tensor, dim: int, kernel_size: int) -> Tensor:
    """Reshape the tensor to the shape of (B, C, K, H, W) or (B, C, H, K, W) for 1D convolution.

    Args:
        tensor (Tensor): Tensor to be reshaped.
        dim (int): Dimension to be resized.
        kernel_size (int): Size of the kernel.

    Returns:
        Tensor: Reshaped tensor.
    """

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

    unfold = F_torch.unfold(tensor, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)

    return unfold


def _resize_1d_torch(
        tensor: Tensor,
        dim: int,
        size: int,
        scale: float,
        kernel: str = "cubic",
        sigma: float = 2.0,
        padding_type: str = "reflect",
        antialiasing: bool = True,
) -> Tensor:
    """Resize the given tensor to the given size.

    Args:
        tensor (Tensor): Tensor to be resized.
        dim (int): Dimension to be resized.
        size (int): Size of the resized dimension.
        scale (float): Scale factor of the resized dimension.
        kernel (str, optional): Kernel type. Default: ``cubic``
        sigma (float, optional): Sigma of the gaussian kernel. Default: 2.0
        padding_type (str, optional): Padding type. Default: ``reflect``
        antialiasing (bool, optional): Whether to use antialiasing. Default: ``True``

    Returns:
        Tensor: Resized tensor.
    """

    # Identity case
    if scale == 1:
        return tensor

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel == "cubic":
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
            kernel,
            sigma,
            antialiasing_factor,
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


def _downsampling_2d_torch(
        tensor: Tensor,
        k: Tensor,
        scale: int,
        padding_type: str = "reflect",
) -> Tensor:
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
    y = F_torch.conv2d(tensor, k, padding=0, stride=scale)
    return y


def _cast_input_torch(tensor: Tensor) -> typing.Tuple[Tensor, _D]:
    """Casts the input tensor to the correct data type and stores the original data type.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with the correct data type.
    """

    if tensor.dtype != torch.float32 or tensor.dtype != torch.float64:
        dtype = tensor.dtype
        tensor = tensor.float()
    else:
        dtype = None

    return tensor, dtype


def _cast_output_torch(tensor: Tensor, dtype: _D) -> Tensor:
    if dtype is not None:
        if not dtype.is_floating_point:
            tensor = tensor.round()
        # To prevent over/underflow when converting types
        if dtype is torch.uint8:
            tensor = tensor.clamp(0, 255)

        tensor = tensor.to(dtype=dtype)

    return tensor


def _image_resize_torch(
        x: Tensor,
        scale_factor: typing.Optional[float] = None,
        sizes: typing.Optional[typing.Tuple[int, int]] = None,
        kernel: typing.Union[str, Tensor] = "cubic",
        sigma: float = 2,
        padding_type: str = "reflect",
        antialiasing: bool = True,
) -> Tensor:
    """Resize image with given kernel and sigma.

    Args:
        x (Tensor): Input image with shape (b, c, h, w)
        scale_factor (float): Scale factor for resizing
        sizes (tuple): Size of the output image (h, w)
        kernel (str or Tensor, optional): Kernel type or kernel tensor. Default: ``cubic``
        sigma (float): Sigma for Gaussian kernel. Default: 2
        padding_type (str): Padding type for convolution. Default: ``reflect``
        antialiasing (bool): Whether to use antialiasing or not. Default: ``True``

    Returns:
        Tensor: Resized image with shape (b, c, h, w)
    """

    # Only one zoom factor and target size can be selected
    if scale_factor is None and sizes is None:
        raise ValueError("One of scale or sizes must be specified!")
    if scale_factor is not None and sizes is not None:
        raise ValueError("Please specify scale or sizes to avoid conflict!")

    # Reshape the input tensor to 4-dim tensor
    x, b, c, h, w = _reshape_input_torch(x)

    scales = (1.0, 1.0)

    # Determine output size
    if sizes is None and scale_factor is not None:
        sizes = (math.ceil(h * scale_factor), math.ceil(w * scale_factor))
        scales = (scale_factor, scale_factor)

    # Determine output scale
    if scale_factor is None and sizes is not None:
        scales = (sizes[0] / h, sizes[1] / w)

    # Casts the input tensor to the correct data type and stores the original data type.
    x, dtype = _cast_input_torch(x)

    if isinstance(kernel, str) and sizes is not None:
        # Core resizing module
        x = _resize_1d_torch(
            x,
            -2,
            sizes[0],
            scales[0],
            kernel,
            sigma,
            padding_type,
            antialiasing)
        x = _resize_1d_torch(
            x,
            -1,
            sizes[1],
            scales[1],
            kernel,
            sigma,
            padding_type,
            antialiasing)
    elif isinstance(kernel, torch.Tensor) and scale_factor is not None:
        x = _downsampling_2d_torch(x, kernel, scale=int(1 / scale_factor))

    x = _reshape_tensor_torch(x, b, c)
    x = _cast_output_torch(x, dtype)
    return x


def _mse_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        only_test_y_channel: bool,
        data_range: float = 1.0,
        eps: float = 1e-8,
) -> Tensor:
    """PyTorch implements the MSE (Mean Squared Error, mean square error) function

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 1]
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        data_range (float, optional): Maximum value range of images. Default: 1.0
        eps (float, optional): Deviation prevention denominator is 0. Default: 1e-8

    Returns:
        mse_metrics (Tensor): MSE metrics

    """
    # Convert RGB tensor data to YCbCr tensor, and only extract Y channel data
    if only_test_y_channel and raw_tensor.shape[1] == 3 and dst_tensor.shape[1] == 3:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, True)

    mse_metrics = torch.mean((raw_tensor * data_range - dst_tensor * data_range) ** 2 + eps, dim=[1, 2, 3])

    return mse_metrics


def _psnr_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        only_test_y_channel: bool,
        data_range: float = 1.0,
        eps: float = 1e-8,
) -> Tensor:
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 1]
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        data_range (float, optional): Maximum value range of images. Default: 1.0
        eps (float, optional): Deviation prevention denominator is 0. Default: 1e-8

    Returns:
        psnr_metrics (Tensor): PSNR metrics

    """
    # Convert RGB tensor data to YCbCr tensor, and only extract Y channel data
    if only_test_y_channel and raw_tensor.shape[1] == 3 and dst_tensor.shape[1] == 3:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, True)

    mse_metrics = _mse_torch(raw_tensor, dst_tensor, only_test_y_channel, data_range, eps)
    psnr_metrics = 10 * torch.log10_(data_range ** 2 / mse_metrics)

    return psnr_metrics


def _ssim_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        gaussian_kernel_window: Tensor,
        downsampling: bool = False,
        get_ssim_map: bool = False,
        get_cs_map: bool = False,
        get_weight: bool = False,
        only_test_y_channel: bool = True,
        data_range: float = 255.0,
):
    """PyTorch implements SSIM (Structural Similarity) function

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 255]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 255]
        gaussian_kernel_window (Tensor): Gaussian filter
        downsampling (bool): Whether to perform downsampling, default: ``False``
        get_ssim_map (bool): Whether to return SSIM image, default: ``False``
        get_cs_map (bool): whether to return CS image, default: ``False``
        get_weight (bool): whether to return the weight image, default: ``False``
        only_test_y_channel (bool): Whether to test only the Y channel of the image, default: ``True``
        data_range (float, optional): Maximum value range of images. Default: 255.0

    Returns:
        ssim_metrics (Tensor): SSIM metrics
    """

    if data_range != 255.0:
        warnings.warn(f"`data_range` must be 255.0!")
        data_range = 255.0

    # Convert RGB tensor data to YCbCr tensor, and only extract Y channel data
    if only_test_y_channel and raw_tensor.shape[1] == 3 and dst_tensor.shape[1] == 3:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, True)
        raw_tensor = raw_tensor[:, [0], :, :] * data_range
        dst_tensor = dst_tensor[:, [0], :, :] * data_range
        # Round image data
        raw_tensor = raw_tensor - raw_tensor.detach() + raw_tensor.round()
        dst_tensor = dst_tensor - dst_tensor.detach() + dst_tensor.round()
    else:
        raw_tensor = raw_tensor * data_range
        raw_tensor = raw_tensor - raw_tensor.detach() + raw_tensor.round()
        dst_tensor = dst_tensor * data_range
        dst_tensor = dst_tensor - dst_tensor.detach() + dst_tensor.round()

    gaussian_kernel_window = gaussian_kernel_window.to(raw_tensor.device, dtype=raw_tensor.dtype)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # If the image size is large enough, downsample
    downsampling_factor = max(1, round(min(raw_tensor.size()[-2:]) / 256))
    if (downsampling_factor > 1) and downsampling:
        raw_tensor = F_torch.avg_pool2d(raw_tensor, kernel_size=(downsampling_factor, downsampling_factor))
        dst_tensor = F_torch.avg_pool2d(dst_tensor, kernel_size=(downsampling_factor, downsampling_factor))

    mu1 = F_torch.conv2d(raw_tensor,
                         gaussian_kernel_window,
                         stride=(1, 1),
                         padding=(0, 0),
                         groups=raw_tensor.shape[1])
    mu2 = F_torch.conv2d(dst_tensor,
                         gaussian_kernel_window,
                         stride=(1, 1),
                         padding=(0, 0),
                         groups=dst_tensor.shape[1])
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F_torch.conv2d(raw_tensor * raw_tensor,
                               gaussian_kernel_window,
                               stride=(1, 1),
                               padding=(0, 0),
                               groups=(dst_tensor * dst_tensor).shape[1]) - mu1_sq
    sigma2_sq = F_torch.conv2d(dst_tensor * dst_tensor,
                               gaussian_kernel_window,
                               stride=(1, 1),
                               padding=(0, 0),
                               groups=(dst_tensor * dst_tensor).shape[1]) - mu2_sq
    sigma12 = F_torch.conv2d(raw_tensor * dst_tensor,
                             gaussian_kernel_window,
                             stride=(1, 1),
                             padding=(0, 0),
                             groups=(dst_tensor * dst_tensor).shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    # Force ssim output to be non-negative to avoid negative results
    cs_map = F_torch.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    ssim_metrics = ssim_map.mean([1, 2, 3])

    if get_ssim_map:
        return ssim_map

    if get_cs_map:
        return ssim_metrics, cs_map.mean([1, 2, 3])

    if get_weight:
        weights = torch.log((1 + sigma1_sq / c2) * (1 + sigma2_sq / c2))
        return ssim_map, weights

    return ssim_metrics


class MSE(nn.Module):
    """PyTorch implements the MSE (Mean Squared Error, mean square error) function"""

    def __init__(self, crop_border: int = 0, only_test_y_channel: bool = True, **kwargs) -> None:
        """

        Args:
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            mse_metrics (Tensor): MSE metrics
        """

        super(MSE, self).__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.kwargs = kwargs

    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        # Check if two tensor scales are similar
        _check_tensor_shape(raw_tensor, dst_tensor)

        # crop pixel boundaries
        if self.crop_border > 0:
            raw_tensor = raw_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            dst_tensor = dst_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        mse_metrics = _mse_torch(raw_tensor, dst_tensor, self.only_test_y_channel, **self.kwargs)

        return mse_metrics


class PSNR(nn.Module):
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function"""

    def __init__(self, crop_border: int = 0, only_test_y_channel: bool = True, **kwargs) -> None:
        """

        Args:
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            psnr_metrics (Tensor): PSNR metrics
        """
        super(PSNR, self).__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.kwargs = kwargs

    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        # Check if two tensor scales are similar
        _check_tensor_shape(raw_tensor, dst_tensor)

        # crop pixel boundaries
        if self.crop_border > 0:
            raw_tensor = raw_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            dst_tensor = dst_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        psnr_metrics = _psnr_torch(raw_tensor, dst_tensor, self.only_test_y_channel, **self.kwargs)

        return psnr_metrics


class SSIM(nn.Module):
    """PyTorch implements SSIM (Structural Similarity) function"""

    def __init__(
            self,
            window_size: int = 11,
            gaussian_sigma: float = 1.5,
            channels: int = 3,
            downsampling: bool = False,
            get_ssim_map: bool = False,
            get_cs_map: bool = False,
            get_weight: bool = False,
            crop_border: int = 0,
            only_test_y_channel: bool = True,
            **kwargs,
    ) -> None:
        """

        Args:
            window_size (int): Gaussian filter size, must be an odd number, default: ``11``
            gaussian_sigma (float): sigma parameter in Gaussian filter, default: ``1.5``
            channels (int): number of image channels, default: ``3``
            downsampling (bool): Whether to perform downsampling, default: ``False``
            get_ssim_map (bool): Whether to return SSIM image, default: ``False``
            get_cs_map (bool): whether to return CS image, default: ``False``
            get_weight (bool): whether to return the weight image, default: ``False``
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            ssim_metrics (Tensor): SSIM metrics

        """
        super(SSIM, self).__init__()
        if only_test_y_channel and channels != 1:
            channels = 1
        self.gaussian_kernel_window = _fspecial_gaussian_torch(window_size, gaussian_sigma, channels)
        self.downsampling = downsampling
        self.get_ssim_map = get_ssim_map
        self.get_cs_map = get_cs_map
        self.get_weight = get_weight
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.kwargs = kwargs

    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        # Check if two tensor scales are similar
        _check_tensor_shape(raw_tensor, dst_tensor)

        # crop pixel boundaries
        if self.crop_border > 0:
            raw_tensor = raw_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            dst_tensor = dst_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        ssim_metrics = _ssim_torch(raw_tensor,
                                   dst_tensor,
                                   self.gaussian_kernel_window,
                                   self.downsampling,
                                   self.get_ssim_map,
                                   self.get_cs_map,
                                   self.get_weight,
                                   self.only_test_y_channel,
                                   **self.kwargs)

        return ssim_metrics
