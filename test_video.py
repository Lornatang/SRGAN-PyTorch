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
import logging
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from torchvision.transforms import FiveCrop
from torchvision.transforms import InterpolationMode as Mode
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
from tqdm import tqdm

from srgan_pytorch.model import generator
from srgan_pytorch.utils.common import create_folder

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--model-path", default="", type=str,
                    help="Path to latest checkpoint for model.")
parser.add_argument("--cuda", dest="cuda", action="store_true",
                    help="Enables cuda.")
parser.add_argument("--view", dest="view", action="store_true",
                    help="Do you want to show SR video synchronously.")
args = parser.parse_args()

# Set whether to use CUDA.
device = torch.device("cuda:0" if args.cuda else "cpu")


def main():
    # Load model and weights.
    model = generator(args.pretrained).to(device).eval()
    if args.model_path != "":
        logger.info(f"Loading weights from `{args.model_path}`.")
        model.load_state_dict(torch.load(args.model_path))

    # Get video filename.
    filename = os.path.basename(args.file)

    # OpenCV video input method open.
    video_capture = cv2.VideoCapture(args.file)
    # Prepare to write the processed image into the video.
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set video window resolution size.
    raw_video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sr_video_size = (raw_video_size[0] * 4, raw_video_size[1] * 4)
    compare_video_size = (sr_video_size[0] * 2 + 10, sr_video_size[1] + 10 + sr_video_size[0] // 5 - 9)
    # Video write loader.
    sr_writer_path = os.path.join("videos", f"sr_{4}x_{filename}")
    compare_writer_path = os.path.join("videos", f"compare_{4}x_{filename}")
    sr_writer = cv2.VideoWriter(sr_writer_path, cv2.VideoWriter_fourcc(*"MPEG"), fps, sr_video_size)
    compare_writer = cv2.VideoWriter(compare_writer_path, cv2.VideoWriter_fourcc(*"MPEG"), fps, compare_video_size)

    # read video frame.
    with torch.no_grad():
        success, raw_frame = video_capture.read()
        for _ in tqdm(range(total_frames),
                      desc="[processing video and saving/view result videos]"):
            if success:
                # The low resolution image is reconstructed to the super resolution image.
                sr_tensor = ToTensor()(raw_frame).unsqueeze(0).to(device)
                sr = model(sr_tensor)

                # Convert N*C*H*W image data to H*W*C image data.
                sr = sr.cpu()
                sr = sr.data[0].numpy()
                sr *= 255.0
                sr = (np.uint8(sr)).transpose((1, 2, 0))
                # Writer sr video to SR video file.
                sr_writer.write(sr)

                # Make compared video and crop shot of left top\right top\center\left bottom\right bottom.
                sr = ToPILImage()(sr)
                # Five areas are selected as the bottom contrast map.
                crop_sr_images = FiveCrop(sr.width // 5 - 9)(sr)
                crop_sr_images = [np.asarray(Pad(padding=(10, 5, 0, 0))(image)) for image in crop_sr_images]
                sr = Pad(padding=(5, 0, 0, 5))(sr)
                # Five areas in the contrast map are selected as the bottom contrast map
                compare_image_size = (sr_video_size[1], sr_video_size[0])
                compare_image = Resize(compare_image_size, Mode.BICUBIC)(ToPILImage()(raw_frame))
                # compare_image = ToPILImage()(compare_image)
                crop_compare_images = FiveCrop(compare_image.width // 5 - 9)(compare_image)
                crop_compare_images = [np.asarray(Pad((0, 5, 10, 0))(image)) for image in crop_compare_images]
                compare_image = Pad(padding=(0, 0, 5, 5))(compare_image)
                # Concatenate all the pictures to one single picture
                # 1. Mosaic the left and right images of the video.
                top_image = np.concatenate((np.asarray(compare_image), np.asarray(sr)), axis=1)
                # 2. Mosaic the bottom left and bottom right images of the video.
                bottom_image = np.concatenate(crop_compare_images + crop_sr_images, axis=1)
                bottom_image_height = int(top_image.shape[1] / bottom_image.shape[1] * bottom_image.shape[0])
                bottom_image_width = top_image.shape[1]
                # 3. Adjust to the right size.
                bottom_image_size = (bottom_image_height, bottom_image_width)
                bottom_image = np.asarray(Resize(bottom_image_size)(ToPILImage()(bottom_image)))
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
    create_folder("videos")

    logger.info("TestEngine:")
    logger.info("\tAPI version .......... 0.4.0")
    logger.info("\tBuild ................ 2021.07.09")

    main()

    logger.info("Super-resolution video completed successfully.\n")
