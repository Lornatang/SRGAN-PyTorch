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
import warnings

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode as Mode
from tqdm import tqdm

import srgan_pytorch.models as models
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.transform import process_image

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network")

parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         " (default: srgan)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4, 8],
                    help="Low to high resolution scaling factor. Optional: [2, 4, 8] (default: 4)")
parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model.")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--seed", default=None, type=int,
                    help="Seed for initializing training.")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")
parser.add_argument("--view", dest="view", action="store_true",
                    help="Do you want to show SR video synchronously.")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for testing.")

    model = configure(args)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    cudnn.benchmark = True

    # Set eval mode.
    model.eval()

    # Get video filename.
    filename = os.path.basename(args.file)

    # Image preprocessing operation
    tensor2pil = transforms.ToPILImage()

    video_capture = cv2.VideoCapture(args.file)
    # Prepare to write the processed image into the video.
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set video size
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sr_size = (size[0] * args.upscale_factor, size[1] * args.upscale_factor)
    pare_size = (sr_size[0] * 2 + 10, sr_size[1] + 10 + sr_size[0] // 5 - 9)
    # Video write loader.
    sr_writer = cv2.VideoWriter(os.path.join("video", f"sr_{args.upscale_factor}x_{filename}"),
                                cv2.VideoWriter_fourcc(*"MPEG"),
                                fps,
                                sr_size)
    compare_writer = cv2.VideoWriter(os.path.join("video", f"compare_{args.upscale_factor}x_{filename}"),
                                     cv2.VideoWriter_fourcc(*"MPEG"),
                                     fps,
                                     pare_size)

    # read frame
    success, raw_frame = video_capture.read()
    progress_bar = tqdm(range(total_frames), desc="[processing video and saving/view result videos]")
    for _ in progress_bar:
        if success:
            # Read img to tensor and transfer to the specified device for processing.
            lr = process_image(raw_frame, args.gpu)

            with torch.no_grad():
                sr = model(lr)

            sr = sr.cpu()
            sr = sr.data[0].numpy()
            sr *= 255.0
            sr = (np.uint8(sr)).transpose((1, 2, 0))
            # save sr video
            sr_writer.write(sr)

            # make compared video and crop shot of left top\right top\center\left bottom\right bottom
            sr = tensor2pil(sr)
            # Five areas are selected as the bottom contrast map.
            crop_sr_imgs = transforms.FiveCrop(size=sr.width // 5 - 9)(sr)
            crop_sr_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_sr_imgs]
            sr = transforms.Pad(padding=(5, 0, 0, 5))(sr)
            # Five areas in the contrast map are selected as the bottom contrast map
            compare_img = transforms.Resize((sr_size[1], sr_size[0]), interpolation=Mode.BICUBIC)(tensor2pil(raw_frame))
            crop_compare_imgs = transforms.FiveCrop(size=compare_img.width // 5 - 9)(compare_img)
            crop_compare_imgs = [np.asarray(transforms.Pad((0, 5, 10, 0))(img)) for img in crop_compare_imgs]
            compare_img = transforms.Pad(padding=(0, 0, 5, 5))(compare_img)
            # concatenate all the pictures to one single picture
            # 1. Mosaic the left and right images of the video.
            top_img = np.concatenate((np.asarray(compare_img), np.asarray(sr)), axis=1)
            # 2. Mosaic the bottom left and bottom right images of the video.
            bottom_img = np.concatenate(crop_compare_imgs + crop_sr_imgs, axis=1)
            bottom_img_height = int(top_img.shape[1] / bottom_img.shape[1] * bottom_img.shape[0])
            bottom_img_width = top_img.shape[1]
            # 3. Adjust to the right size.
            bottom_img = np.asarray(
                transforms.Resize((bottom_img_height, bottom_img_width))(tensor2pil(bottom_img)))
            # 4. Combine the bottom zone with the upper zone.
            final_image = np.concatenate((top_img, bottom_img))

            # save compare video
            compare_writer.write(final_image)

            if args.view:
                # display video
                cv2.imshow("LR video convert SR video ", final_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # next frame
            success, raw_frame = video_capture.read()


if __name__ == "__main__":
    print("##################################################\n")
    print("Run SR Engine.\n")

    create_folder("video")

    logger.info("SREngine:")
    print("\tAPI version .......... 0.1.0")
    print("\tBuild ................ 2021.03.23")
    print("##################################################\n")
    main()

    logger.info("Super-resolution video completed successfully.\n")
