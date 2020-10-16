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
import os

import cv2
import numpy as np
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from srgan_pytorch import Generator
from srgan_pytorch import select_device

parser = argparse.ArgumentParser(description="SRGAN algorithm is applied to video files.")
parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--model-path", default="./weight/SRGAN_4x.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: ``./weight/SRGAN_4x.pth``).")
parser.add_argument("--device", default="0",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``CUDA:0``).")
parser.add_argument("--view", action="store_true",
                    help="Super resolution real time to show.")

args = parser.parse_args()
print(args)

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=args.batch_size)

# Construct SRGAN model.
model = Generator(upscale_factor=args.upscale_factor).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set model eval mode
model.eval()

# Image preprocessing operation
pil2tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
tensor2pil = transforms.ToPILImage()

# Open video file
video_name = args.file
print(f"Reading `{os.path.basename(video_name)}`...")
video_capture = cv2.VideoCapture(video_name)
# Prepare to write the processed image into the video.
fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# Set video size
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sr_size = (size[0] * args.upscale_factor, size[1] * args.upscale_factor)
pare_size = (sr_size[0] * 2 + 10, sr_size[1] + 10 + sr_size[0] // 5 - 9)

# Video write loader.
srgan_writer = cv2.VideoWriter(f"srgan_{args.scale_factor}x_{os.path.basename(video_name)}",
                               cv2.VideoWriter_fourcc(*"MPEG"), fps, sr_size)
compare_writer = cv2.VideoWriter(f"compare_{args.scale_factor}x_{os.path.basename(video_name)}",
                                 cv2.VideoWriter_fourcc(*"MPEG"), fps, pare_size)
# read frame
success, raw_frame = video_capture.read()
progress_bar = tqdm(range(total_frames), desc="[processing video and saving/view result videos]")
for index in progress_bar:
    if success:
        img = pil2tensor(raw_frame).unsqueeze(0)
        lr = img.to(device)

        with torch.no_grad():
            sr = model(lr)

        sr = sr.cpu()
        sr = sr.data[0].numpy()
        sr *= 255.0
        sr = (np.uint8(sr)).transpose((1, 2, 0))
        # save sr video
        srgan_writer.write(sr)

        # make compared video and crop shot of left top\right top\center\left bottom\right bottom
        sr = tensor2pil(sr)
        # Five areas are selected as the bottom contrast map.
        crop_sr_imgs = transforms.FiveCrop(size=sr.width // 5 - 9)(sr)
        crop_sr_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_sr_imgs]
        sr = transforms.Pad(padding=(5, 0, 0, 5))(sr)
        # Five areas in the contrast map are selected as the bottom contrast map
        compare_img = transforms.Resize((sr_size[1], sr_size[0]), interpolation=Image.BICUBIC)(tensor2pil(raw_frame))
        crop_compare_imgs = transforms.FiveCrop(size=compare_img.width // 5 - 9)(compare_img)
        crop_compare_imgs = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in crop_compare_imgs]
        compare_img = transforms.Pad(padding=(0, 0, 5, 5))(compare_img)
        # concatenate all the pictures to one single picture
        # 1. Mosaic the left and right images of the video.
        top_img = np.concatenate((np.asarray(compare_img), np.asarray(sr)), axis=1)
        # 2. Mosaic the bottom left and bottom right images of the video.
        bottom_img = np.concatenate(crop_compare_imgs + crop_sr_imgs, axis=1)
        bottom_img_height = int(top_img.shape[1] / bottom_img.shape[1] * bottom_img.shape[0])
        bottom_img_width = top_img.shape[1]
        # 3. Adjust to the right size.
        bottom_img = np.asarray(transforms.Resize((bottom_img_height, bottom_img_width))(tensor2pil(bottom_img)))
        # 4. Combine the bottom zone with the upper zone.
        final_image = np.concatenate((top_img, bottom_img))

        # save compare video
        compare_writer.write(final_image)

        if args.view:
            # display video
            cv2.imshow("LR video convert HR video ", final_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # next frame
        success, raw_frame = video_capture.read()
