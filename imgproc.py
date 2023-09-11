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
import math
import random
from typing import Any

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import functional as F_vision

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "image_resize", "preprocess_one_image",
    "expand_y", "rgb_to_ycbcr", "bgr_to_ycbcr", "ycbcr_to_bgr", "ycbcr_to_rgb",
    "rgb_to_ycbcr_torch", "bgr_to_ycbcr_torch",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
    "center_crop_torch", "random_crop_torch", "random_rotate_torch", "random_vertically_flip_torch",
    "random_horizontally_flip_torch",
]


def _cubic(x: Any) -> Any:
    """Implementation of `cubic` function in Matlab under Python language.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation

    """
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
        ((absx > 1) * (absx <= 2)).type_as(absx))


def _calculate_weights_indices(in_length: int,
                               out_length: int,
                               scale: float,
                               kernel_width: int,
                               antialiasing: bool) -> [np.ndarray, np.ndarray, int, int]:
    """Implementation of `calculate_weights_indices` function in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    Returns:
       weights, indices, sym_len_s, sym_len_e

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
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

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


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def preprocess_one_image(image_path: str, range_norm: bool, half: bool, device: torch.device) -> Tensor:
    # read an image using OpenCV
    image = cv2.imread(image_path).astype(np.float32) / 255.0

    # BGR image channel data to RGB image channel data
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB image channel data to image formats supported by PyTorch
    tensor = image_to_tensor(image, range_norm, half).unsqueeze_(0)

    # Data transfer to the specified device
    tensor = tensor.to(device, non_blocking=True)

    return tensor


def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        out_2 (np.ndarray): Output image with shape (c, h, w), [0, 1] range, w/o round

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
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)
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


def expand_y(image: np.ndarray) -> np.ndarray:
    """Convert BGR channel to YCbCr format,
    and expand Y channel data in YCbCr, from HW to HWC

    Args:
        image (np.ndarray): Y channel image data

    Returns:
        y_image (np.ndarray): Y-channel image data in HWC form

    """
    # Normalize image data to [0, 1]
    image = image.astype(np.float32) / 255.

    # Convert BGR to YCbCr, and extract only Y channel
    y_image = bgr_to_ycbcr(image, only_use_y_channel=True)

    # Expand Y channel
    y_image = y_image[..., None]

    # Normalize the image data to [0, 255]
    y_image = y_image.astype(np.float64) * 255.0

    return y_image


def rgb_to_ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of rgb2ycbcr function in Matlab under Python language

    Args:
        image (np.ndarray): Image input in RGB format.
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def bgr_to_ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of bgr2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in BGR format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2rgb function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): RGB image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def ycbcr_to_bgr(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2bgr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): BGR image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def rgb_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    """Implementation of rgb2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[65.481, -37.797, 112.0],
                         [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def bgr_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    """Implementation of bgr2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = Tensor([[24.966], [128.553], [65.481]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[24.966, 112.0, -18.214],
                         [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image center area.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

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
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_rotate(image,
                  angles: list,
                  center: tuple[int, int] = None,
                  scale_factor: float = 1.0) -> np.ndarray:
    """Rotate an image by a random angle

    Args:
        image (np.ndarray): Image read with OpenCV
        angles (list): Rotation angle range
        center (optional, tuple[int, int]): High resolution image selection center point. Default: ``None``
        scale_factor (optional, float): scaling factor. Default: 1.0

    Returns:
        rotated_image (np.ndarray): image after rotation

    """
    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random select specific angle
    angle = random.choice(angles)
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return rotated_image


def random_horizontally_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image upside down randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Horizontally flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): image after horizontally flip

    """
    if random.random() < p:
        horizontally_flip_image = cv2.flip(image, 1)
    else:
        horizontally_flip_image = image

    return horizontally_flip_image


def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip an image horizontally randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Vertically flip probability. Default: 0.5

    Returns:
        vertically_flip_image (np.ndarray): image after vertically flip

    """
    if random.random() < p:
        vertically_flip_image = cv2.flip(image, 0)
    else:
        vertically_flip_image = image

    return vertically_flip_image


def center_crop_torch(
        gt_images,
        lr_images,
        gt_patch_size: int,
        upscale_factor: int,
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Intercept two images to specify the center area

    Args:
        gt_images (ndarray | Tensor | list[ndarray] | list[Tensor]): ground truth images read by PyTorch
        lr_images (ndarray | Tensor | list[ndarray] | list[Tensor]): Low resolution images read by PyTorch
        gt_patch_size (int): the size of the ground truth image after interception
        upscale_factor (int): the ground truth image size is a magnification of the low resolution image size

    Returns:
        gt_images (ndarray or Tensor or): the intercepted ground truth image
        lr_images (ndarray or Tensor or): low-resolution intercepted images

    """
    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Calculate the size of the low-resolution image that needs to be intercepted
    lr_patch_size = gt_patch_size // upscale_factor

    # Just need to find the top and left coordinates of the image
    lr_top = (lr_image_height - lr_patch_size) // 2
    lr_left = (lr_image_width - lr_patch_size) // 2

    # Capture low-resolution images
    if input_type == "Tensor":
        lr_images = [lr_image[
                     :,
                     :,
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size] for lr_image in lr_images]
    else:
        lr_images = [lr_image[
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size,
                     ...] for lr_image in lr_images]

    # Intercept the ground truth image
    gt_top, gt_left = int(lr_top * upscale_factor), int(lr_left * upscale_factor)

    if input_type == "Tensor":
        gt_images = [v[
                     :,
                     :,
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size] for v in gt_images]
    else:
        gt_images = [v[
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size,
                     ...] for v in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_crop_torch(
        gt_images,
        lr_images,
        gt_patch_size: int,
        upscale_factor: int,
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Randomly intercept two images in the specified area

    Args:
        gt_images (ndarray | Tensor | list[ndarray] | list[Tensor]): ground truth images read by PyTorch
        lr_images (ndarray | Tensor | list[ndarray] | list[Tensor]): Low resolution images read by PyTorch
        gt_patch_size (int): the size of the ground truth image after interception
        upscale_factor (int): the ground truth image size is a magnification of the low resolution image size

    Returns:
        gt_images (ndarray or Tensor or): the intercepted ground truth image
        lr_images (ndarray or Tensor or): low-resolution intercepted images

    """

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Calculate the size of the low-resolution image that needs to be intercepted
    lr_patch_size = gt_patch_size // upscale_factor

    # Just need to find the top and left coordinates of the image
    lr_top = random.randint(0, lr_image_height - lr_patch_size)
    lr_left = random.randint(0, lr_image_width - lr_patch_size)

    # Capture low-resolution images
    if input_type == "Tensor":
        lr_images = [lr_image[
                     :,
                     :,
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size] for lr_image in lr_images]
    else:
        lr_images = [lr_image[
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size,
                     ...] for lr_image in lr_images]

    # Intercept the ground truth image
    gt_top, gt_left = int(lr_top * upscale_factor), int(lr_left * upscale_factor)

    if input_type == "Tensor":
        gt_images = [v[
                     :,
                     :,
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size] for v in gt_images]
    else:
        gt_images = [v[
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size,
                     ...] for v in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_rotate_torch(
        gt_images,
        lr_images,
        upscale_factor: int,
        angles: list,
        gt_center: tuple = None,
        lr_center: tuple = None,
        rotate_scale_factor: float = 1.0
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Randomly rotate the image

    Args:
        gt_images (ndarray | Tensor | list[ndarray] | list[Tensor]): ground truth images read by the PyTorch library
        lr_images (ndarray | Tensor | list[ndarray] | list[Tensor]): low-resolution images read by the PyTorch library
        angles (list): List of random rotation angles
        upscale_factor (int): the ground truth image size is a magnification of the low resolution image size
        gt_center (optional, tuple[int, int]): The center point of the ground truth image selection. Default: ``None``
        lr_center (optional, tuple[int, int]): Low resolution image selection center point. Default: ``None``
        rotate_scale_factor (optional, float): Rotation scaling factor. Default: 1.0

    Returns:
        gt_images (ndarray or Tensor or): ground truth image after rotation
        lr_images (ndarray or Tensor or): Rotated low-resolution images

    """
    # Randomly choose the rotation angle
    angle = random.choice(angles)

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Rotate the low-res image
    if lr_center is None:
        lr_center = [lr_image_width // 2, lr_image_height // 2]

    lr_matrix = cv2.getRotationMatrix2D(lr_center, angle, rotate_scale_factor)

    if input_type == "Tensor":
        lr_images = [F_vision.rotate(lr_image, angle, center=lr_center) for lr_image in lr_images]
    else:
        lr_images = [cv2.warpAffine(lr_image, lr_matrix, (lr_image_width, lr_image_height)) for lr_image in lr_images]

    # rotate the ground truth image
    gt_image_width = int(lr_image_width * upscale_factor)
    gt_image_height = int(lr_image_height * upscale_factor)

    if gt_center is None:
        gt_center = [gt_image_width // 2, gt_image_height // 2]

    gt_matrix = cv2.getRotationMatrix2D(gt_center, angle, rotate_scale_factor)

    if input_type == "Tensor":
        gt_images = [F_vision.rotate(gt_image, angle, center=gt_center) for gt_image in gt_images]
    else:
        gt_images = [cv2.warpAffine(gt_image, gt_matrix, (gt_image_width, gt_image_height)) for gt_image in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_horizontally_flip_torch(
        gt_images,
        lr_images,
        p: float = 0.5
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Randomly flip the image up and down

    Args:
        gt_images (ndarray): ground truth images read by the PyTorch library
        lr_images (ndarray): low resolution images read by the PyTorch library
        p (optional, float): flip probability. Default: 0.5

    Returns:
        gt_images (ndarray or Tensor or): flipped ground truth images
        lr_images (ndarray or Tensor or): flipped low-resolution images

    """
    # Randomly generate flip probability
    flip_prob = random.random()

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            lr_images = [F_vision.hflip(lr_image) for lr_image in lr_images]
            gt_images = [F_vision.hflip(gt_image) for gt_image in gt_images]
        else:
            lr_images = [cv2.flip(lr_image, 1) for lr_image in lr_images]
            gt_images = [cv2.flip(gt_image, 1) for gt_image in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_vertically_flip_torch(
        gt_images,
        lr_images,
        p: float = 0.5
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Randomly flip the image left and right

    Args:
        gt_images (ndarray): ground truth images read by the PyTorch library
        lr_images (ndarray): low resolution images read by the PyTorch library
        p (optional, float): flip probability. Default: 0.5

    Returns:
        gt_images (ndarray or Tensor or): flipped ground truth images
        lr_images (ndarray or Tensor or): flipped low-resolution images

    """
    # Randomly generate flip probability
    flip_prob = random.random()

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            lr_images = [F_vision.vflip(lr_image) for lr_image in lr_images]
            gt_images = [F_vision.vflip(gt_image) for gt_image in gt_images]
        else:
            lr_images = [cv2.flip(lr_image, 0) for lr_image in lr_images]
            gt_images = [cv2.flip(gt_image, 0) for gt_image in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images
