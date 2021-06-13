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
import argparse
import logging
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode as Mode
from tqdm import tqdm

import srgan_pytorch.models as models
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.transform import process_image

# Find all available models.
model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(args):
    if args.seed is not None:
        # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set
        # convolution algorithm.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.warning("You have chosen to seed training. "
                       "This will turn on the CUDNN deterministic setting, "
                       "which can slow down your training considerably! "
                       "You may see unexpected behavior when restarting "
                       "from checkpoints.")
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    # Build a super-resolution model, if model path is defined, the specified model weight will be loaded.
    model = configure(args)
    # Switch model to eval mode.
    model.eval()

    # If the GPU is available, load the model into the GPU memory. This speed.
    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")

    # Get video filename.
    filename = os.path.basename(args.lr)

    # OpenCV video input method open.
    video_capture = cv2.VideoCapture(args.file)
    # Prepare to write the processed image into the video.
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set video window resolution size.
    raw_video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sr_video_size = (raw_video_size[0] * args.upscale_factor, raw_video_size[1] * args.upscale_factor)
    compare_video_size = (sr_video_size[0] * 2 + 10, sr_video_size[1] + 10 + sr_video_size[0] // 5 - 9)
    # Video write loader.
    sr_writer_path = os.path.join("videos", f"sr_{args.upscale_factor}x_{filename}")
    compare_writer_path = os.path.join("videos", f"compare_{args.upscale_factor}x_{filename}")
    sr_writer = cv2.VideoWriter(sr_writer_path, cv2.VideoWriter_fourcc(*"MPEG"), fps, sr_video_size)
    compare_writer = cv2.VideoWriter(compare_writer_path, cv2.VideoWriter_fourcc(*"MPEG"), fps, compare_video_size)

    # read video frame.
    with torch.no_grad():
        success, raw_frame = video_capture.read()
        for _ in tqdm(range(total_frames), desc="[processing video and saving/view result videos]"):
            if success:
                # The low resolution image is reconstructed to the super resolution image.
                sr = model(process_image(raw_frame, norm=False, gpu=args.gpu))

                # Convert N*C*H*W image data to H*W*C image data.
                sr = sr.cpu()
                sr = sr.data[0].numpy()
                sr *= 255.0
                sr = (np.uint8(sr)).transpose((1, 2, 0))
                # Writer sr video to SR video file.
                sr_writer.write(sr)

                # Make compared video and crop shot of left top\right top\center\left bottom\right bottom.
                sr = transforms.ToPILImage()(sr)
                # Five areas are selected as the bottom contrast map.
                crop_sr_images = transforms.FiveCrop(sr.width // 5 - 9)(sr)
                crop_sr_images = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(image)) for image in crop_sr_images]
                sr = transforms.Pad(padding=(5, 0, 0, 5))(sr)
                # Five areas in the contrast map are selected as the bottom contrast map
                compare_image_size = (sr_video_size[1], sr_video_size[0])
                compare_image = transforms.Resize(compare_image_size, interpolation=Mode.BICUBIC)(raw_frame)
                compare_image = transforms.ToPILImage()(compare_image)
                crop_compare_images = transforms.FiveCrop(compare_image.width // 5 - 9)(compare_image)
                crop_compare_images = [np.asarray(transforms.Pad((0, 5, 10, 0))(image)) for image in
                                       crop_compare_images]
                compare_image = transforms.Pad(padding=(0, 0, 5, 5))(compare_image)
                # Concatenate all the pictures to one single picture
                # 1. Mosaic the left and right images of the video.
                top_image = np.concatenate((np.asarray(compare_image), np.asarray(sr)), axis=1)
                # 2. Mosaic the bottom left and bottom right images of the video.
                bottom_image = np.concatenate(crop_compare_images + crop_sr_images, axis=1)
                bottom_image_height = int(top_image.shape[1] / bottom_image.shape[1] * bottom_image.shape[0])
                bottom_image_width = top_image.shape[1]
                # 3. Adjust to the right size.
                bottom_image_size = (bottom_image_height, bottom_image_width)
                bottom_image = np.asarray(transforms.Resize(bottom_image_size)(transforms.ToPILImage()(bottom_image)))
                # 4. Combine the bottom zone with the upper zone.
                images = np.concatenate((top_image, bottom_image))

                # Writer compare video to compare video file.
                compare_writer.write(images)

                # Display compare video.
                if args.view:
                    cv2.imshow("LR video convert SR video ", images)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Read next frame.
                success, raw_frame = video_capture.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="srgan", type=str, choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `srgan`)")
    parser.add_argument("--file", type=str, required=True,
                        help="Test low resolution video name.")
    parser.add_argument("--upscale-factor", default=4, type=int, choices=[4],
                        help="Low to high resolution scaling factor. Optional: [4]. (Default: 4)")
    parser.add_argument("--model-path", default="", type=str,
                        help="Path to latest checkpoint for model.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for initializing training.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    parser.add_argument("--view", dest="view", action="store_true",
                        help="Do you want to show SR video synchronously.")
    args = parser.parse_args()

    create_folder("videos")

    logger.info("TestEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.06.13")

    main(args)

    logger.info("Super-resolution video completed successfully.\n")
