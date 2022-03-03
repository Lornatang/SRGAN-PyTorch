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
"""Realize the function of processing the dataset before training."""
import math
import random
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

__all__ = [
    "image2tensor", "tensor2image",
    "rgb2ycbcr", "bgr2ycbcr", "ycbcr2bgr", "ycbcr2rgb",
    "center_crop", "random_crop", "random_rotate", "random_horizontally_flip", "random_vertically_flip",
]


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert ``PIL.Image`` to Tensor.

    Args:
        image (np.ndarray): The image data read by ``PIL.Image``
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        Normalized image data

    Examples:
        >>> image = cv2.imread("image.bmp", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        >>> tensor_image = image2tensor(image, range_norm=False, half=False)
    """

    tensor = F.to_tensor(image)

    if range_norm:
        tensor = tensor.mul_(2.0).sub_(1.0)
    if half:
        tensor = tensor.half()

    return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    """Converts ``torch.Tensor`` to ``PIL.Image``.

    Args:
        tensor (torch.Tensor): The image that needs to be converted to ``PIL.Image``
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        Convert image data to support PIL library

    Examples:
        >>> tensor = torch.randn([1, 3, 128, 128])
        >>> image = tensor2image(tensor, range_norm=False, half=False)
    """

    if range_norm:
        tensor = tensor.add_(1.0).div_(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze_(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).cpu().numpy().astype("uint8")

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def cubic(x: Any):
    """Implementation of `cubic` function in Matlab under Python language.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation.
    """

    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
        ((absx > 1) * (absx <= 2)).type_as(absx))


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def calculate_weights_indices(in_length: int, out_length: int, scale: float, kernel_width: int, antialiasing: bool):
    """Implementation of `calculate_weights_indices` function in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    """

    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def imresize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        np.ndarray: Output image with shape (c, h, w), [0, 1] range, w/o round.
    """
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(in_h, out_h, scale_factor, kernel_width, antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(in_w, out_w, scale_factor, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)

    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def rgb2ycbcr(image: np.ndarray, use_y_channel: bool = False) -> np.ndarray:
    """Implementation of rgb2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in RGB format.
        use_y_channel (bool): Extract Y channel separately. Default: ``False``.

    Returns:
        ndarray: YCbCr image array data.
    """

    if use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def bgr2ycbcr(image: np.ndarray, use_y_channel: bool = False) -> np.ndarray:
    """Implementation of bgr2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in BGR format.
        use_y_channel (bool): Extract Y channel separately. Default: ``False``.

    Returns:
        ndarray: YCbCr image array data.
    """

    if use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def ycbcr2rgb(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2rgb function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        ndarray: RGB image array data.
    """

    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    image = image.astype(image_dtype)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def ycbcr2bgr(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2bgr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        ndarray: BGR image array data.
    """

    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image center area.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        np.ndarray: Small patch image.
    """

    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = (image_height - image_size) // 2
    left = (image_width - image_size) // 2

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        np.ndarray: Small patch image.
    """

    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_rotate(image: np.ndarray, angles: list, center=None, scale_factor: float = 1.0) -> np.ndarray:
    """Rotate an image randomly by a specified angle.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        angles (list): Specify the rotation angle.
        center (tuple[int]): Image rotation center. If the center is None, initialize it as the center of the image. ``Default: None``.
        scale_factor (float): scaling factor. Default: 1.0.

    Returns:
        np.ndarray: Rotated image.
    """

    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random select specific angle
    angle = random.choice(angles)
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return image


def random_horizontally_flip(image: np.ndarray, p=0.5) -> np.ndarray:
    """Flip an image horizontally randomly.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        p (optional, float): rollover probability. (Default: 0.5)

    Returns:
        np.ndarray: Horizontally flip image.
    """

    if random.random() < p:
        image = cv2.flip(image, 1)

    return image


def random_vertically_flip(image: np.ndarray, p=0.5) -> np.ndarray:
    """Flip an image vertically randomly.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        p (optional, float): rollover probability. (Default: 0.5)

    Returns:
        np.ndarray: Vertically flip image.
    """

    if random.random() < p:
        image = cv2.flip(image, 0)

    return image
