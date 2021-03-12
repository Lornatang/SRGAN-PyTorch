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
import logging
import os

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import InterpolationMode as Mode
from tqdm import tqdm

from srgan_pytorch.dataset import CustomTestDataset
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.estimate import iqa
from srgan_pytorch.utils.image_quality_assessment import GMSD
from srgan_pytorch.utils.image_quality_assessment import LPIPS
from srgan_pytorch.utils.image_quality_assessment import SSIM
from srgan_pytorch.utils.transform import process_image

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


class Test(object):
    def __init__(self, args):
        self.args = args
        self.model = configure(args)

        # Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
        self.ssim_loss = SSIM().cuda(args.gpu)
        # Reference sources from https://github.com/richzhang/PerceptualSimilarity
        self.lpips_loss = LPIPS(args.gpu).cuda(args.gpu)
        # Reference sources from http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
        self.gmsd_loss = GMSD().cuda(args.gpu)

        logger.info("Load testing dataset")
        dataset = CustomTestDataset(root=os.path.join(args.data, "test"),
                                    image_size=args.image_size)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=args.batch_size,
                                                      pin_memory=True,
                                                      num_workers=args.workers)

        logger.info(f"Dataset information\n"
                    f"\tDataset dir is `{os.getcwd()}/{args.data}/test`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {args.workers}\n"
                    f"\tLoad dataset to CUDA")

    def run(self):
        args = self.args
        # Evaluate algorithm performance.
        total_mse_value = 0.0
        total_rmse_value = 0.0
        total_psnr_value = 0.0
        total_ssim_value = 0.0
        total_mssim_value = 0.0
        total_lpips_value = 0.0
        total_gmsd_value = 0.0

        # Start evaluate model performance.
        progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))

        for i, (lr, bicubic, hr) in progress_bar:
            # Move data to special device.
            if args.gpu is not None:
                lr = lr.cuda(args.gpu, non_blocking=True)
                bicubic = bicubic.cuda(args.gpu, non_blocking=True)
                hr = hr.cuda(args.gpu, non_blocking=True)

            with torch.no_grad():
                sr = self.model(lr)

            # Evaluate performance
            value = iqa(os.path.join("benchmark", "sr.bmp"), os.path.join("benchmark", "hr.bmp"), args.gpu)

            total_mse_value += value[0]
            total_rmse_value += value[1]
            total_psnr_value += value[2]
            total_ssim_value += value[3]
            total_mssim_value += value[4]
            total_lpips_value += value[5]
            total_gmsd_value += value[6]

            progress_bar.set_description(f"[{i + 1}/{len(self.dataloader)}] PSNR: {value[2]:.2f}dB")

            images = torch.cat([bicubic, sr, hr], dim=-1)
            vutils.save_image(images, os.path.join("benchmark", f"{i + 1}.bmp"), padding=10)

        print(f"Performance average results:\n")
        print(f"indicator Score\n")
        print(f"--------- -----\n")
        print(f"MSE       {total_mse_value / len(self.dataloader):.2f}\n"
              f"RMSE      {total_rmse_value / len(self.dataloader):.2f}\n"
              f"PSNR      {total_psnr_value / len(self.dataloader):.2f}\n"
              f"SSIM      {total_ssim_value / len(self.dataloader):.2f}\n"
              f"MS-SSIM   {total_mssim_value / len(self.dataloader):.2f}\n"
              f"LPIPS     {total_lpips_value / len(self.dataloader):.2f}\n"
              f"GMSD      {total_ssim_value}")


class Estimate(object):
    def __init__(self, args):
        self.args = args
        self.model = configure(args)

    def run(self):
        args = self.args
        # Get filename.
        filename = os.path.basename(args.lr)

        # Read all pictures.
        lr = process_image(Image.open(args.lr), args.gpu)
        bicubic = process_image(transforms.Resize(args.image_size, interpolation=Mode.BICUBIC)(lr), args.gpu)

        with torch.no_grad():
            sr = self.model(lr)

        if args.hr:
            hr = process_image(Image.open(args.hr), args.gpu)
            vutils.save_image(hr, os.path.join("test", f"hr_{filename}"))
            images = torch.cat([bicubic, sr, hr], dim=-1)
        else:
            images = torch.cat([bicubic, sr], dim=-1)

        vutils.save_image(lr, os.path.join("test", f"lr_{filename}"))
        vutils.save_image(bicubic, os.path.join("test", f"bicubic_{filename}"))
        vutils.save_image(sr, os.path.join("test", f"sr_{filename}"))
        vutils.save_image(images, os.path.join("test", f"compare_{filename}"), padding=10)
        value = iqa(os.path.join("test", f"sr_{filename}"), args.hr, args.gpu)

        print(f"Performance avg results:\n")
        print(f"indicator Score\n")
        print(f"--------- -----\n")
        print(f"MSE       {value[0]:.2f}\n"
              f"RMSE      {value[1]:.2f}\n"
              f"PSNR      {value[2]:.2f}\n"
              f"SSIM      {value[3]:.2f}\n"
              f"MS-SSIM   {value[4]:.2f}\n"
              f"LPIPS     {value[5]:.2f}\n"
              f"GMSD      {value[6]:.2f}\n")


class Video(object):
    def __init__(self, args):
        self.args = args
        self.model = configure(args)
        # Image preprocessing operation
        self.tensor2pil = transforms.ToPILImage()

        self.video_capture = cv2.VideoCapture(args.file)
        # Prepare to write the processed image into the video.
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Set video size
        self.size = (int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.sr_size = (self.size[0] * args.upscale_factor, self.size[1] * args.upscale_factor)
        self.pare_size = (self.sr_size[0] * 2 + 10, self.sr_size[1] + 10 + self.sr_size[0] // 5 - 9)
        # Video write loader.
        self.sr_writer = cv2.VideoWriter(
            os.path.join("video", f"sr_{args.upscale_factor}x_{os.path.basename(args.file)}"),
            cv2.VideoWriter_fourcc(*"MPEG"),
            self.fps,
            self.sr_size)
        self.compare_writer = cv2.VideoWriter(
            os.path.join("video", f"compare_{args.upscale_factor}x_{os.path.basename(args.file)}"),
            cv2.VideoWriter_fourcc(*"MPEG"),
            self.fps,
            self.pare_size)

    def run(self):
        args = self.args
        # Set eval model.
        self.model.eval()

        # read frame
        success, raw_frame = self.video_capture.read()
        progress_bar = tqdm(range(self.total_frames), desc="[processing video and saving/view result videos]")
        for _ in progress_bar:
            if success:
                # Read img to tensor and transfer to the specified device for processing.
                lr = process_image(Image.open(args.lr), args.gpu)

                with torch.no_grad():
                    sr = self.model(lr)

                sr = sr.cpu()
                sr = sr.data[0].numpy()
                sr *= 255.0
                sr = (np.uint8(sr)).transpose((1, 2, 0))
                # save sr video
                self.sr_writer.write(sr)

                # make compared video and crop shot of left top\right top\center\left bottom\right bottom
                sr = self.tensor2pil(sr)
                # Five areas are selected as the bottom contrast map.
                crop_sr_imgs = transforms.FiveCrop(size=sr.width // 5 - 9)(sr)
                crop_sr_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_sr_imgs]
                sr = transforms.Pad(padding=(5, 0, 0, 5))(sr)
                # Five areas in the contrast map are selected as the bottom contrast map
                compare_img = transforms.Resize((self.sr_size[1], self.sr_size[0]),
                                                interpolation=Image.BICUBIC)(self.tensor2pil(raw_frame))
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
                    transforms.Resize((bottom_img_height, bottom_img_width))(self.tensor2pil(bottom_img)))
                # 4. Combine the bottom zone with the upper zone.
                final_image = np.concatenate((top_img, bottom_img))

                # save compare video
                self.compare_writer.write(final_image)

                if args.view:
                    # display video
                    cv2.imshow("LR video convert SR video ", final_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # next frame
                success, raw_frame = self.video_capture.read()
