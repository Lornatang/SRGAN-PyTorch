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
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from srgan_pytorch import Generator

parser = argparse.ArgumentParser(description="SRGAN algorithm is applied to video files.")
parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("--weights", type=str, required=True,
                    help="Generator model name. ")
parser.add_argument("--scale-factor", type=int, default=4, choices=[4],
                    help="Super resolution upscale factor. (default:4)")
parser.add_argument("--view", action="store_true",
                    help="Super resolution real time to show.")
parser.add_argument("--cuda", action="store_true",
                    help="Enables cuda")

args = parser.parse_args()
print(args)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = Generator(scale_factor=args.scale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set model eval mode
model.eval()

# Image preprocessing operation
pil2tensor = transforms.ToTensor()
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
sr_size = (size[0] * args.scale_factor, size[1] * args.scale_factor)
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
        img = img.to(device)

        with torch.no_grad():
            prediction = model(img)

        prediction = prediction.cpu()
        sr_frame = prediction.data[0].numpy()
        sr_frame *= 255.0
        sr_frame = (np.uint8(sr_frame)).transpose((1, 2, 0))
        # save sr video
        srgan_writer.write(sr_frame)

        # make compared video and crop shot of left top\right top\center\left bottom\right bottom
        sr_frame = tensor2pil(sr_frame)
        # Five areas are selected as the bottom contrast map.
        crop_sr_imgs = transforms.FiveCrop(size=sr_frame.width // 5 - 9)(sr_frame)
        crop_sr_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_sr_imgs]
        sr_frame = transforms.Pad(padding=(5, 0, 0, 5))(sr_frame)
        # Five areas in the contrast map are selected as the bottom contrast map
        compare_img = transforms.Resize((sr_size[1], sr_size[0]), interpolation=Image.BICUBIC)(tensor2pil(raw_frame))
        crop_compare_imgs = transforms.FiveCrop(size=compare_img.width // 5 - 9)(compare_img)
        crop_compare_imgs = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in crop_compare_imgs]
        compare_img = transforms.Pad(padding=(0, 0, 5, 5))(compare_img)
        # concatenate all the pictures to one single picture
        # 1. Mosaic the left and right images of the video.
        top_img = np.concatenate((np.asarray(compare_img), np.asarray(sr_frame)), axis=1)
        # 2. Mosaic the bottom left and bottom right images of the video.
        bottom_img = np.concatenate(crop_compare_imgs + crop_sr_imgs, axis=1)
        bottom_img_height = int(top_img.shape[1] / bottom_img.shape[1] * bottom_img.shape[0])
        bottom_img_width = top_img.shape[1]
        # 3. Adjust to the right size.
        bottom_img = np.asarray(transforms.Resize((bottom_img_height, bottom_img_width))(tensor2pil(bottom_img)))
        # 4. Combine the bottom zone with the upper zone.
        final_image = np.concatenate((top_img, bottom_img))

        if args.view:
            # display video
            cv2.imshow("LR video convert HR video ", final_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # save compare video
            compare_writer.write(final_image)

        # next frame
        success, raw_frame = video_capture.read()
