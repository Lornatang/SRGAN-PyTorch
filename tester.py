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
import math
import os

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from srgan_pytorch.dataset import CustomTestDataset
from srgan_pytorch.utils.calculate_ssim import ssim
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import inference
from srgan_pytorch.utils.estimate import image_quality_evaluation
from srgan_pytorch.utils.transform import process_image

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


class Test(object):
    def __init__(self, args):
        self.args = args
        self.model, self.device = configure(args)

        logger.info("Load testing dataset")
        dataset = CustomTestDataset(root=f"{args.data}/test",
                                    image_size=args.image_size)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=args.batch_size,
                                                      pin_memory=True,
                                                      num_workers=int(args.workers))

        logger.info(f"Dataset information\n"
                    f"\tDataset dir is `{os.getcwd()}/{args.data}/test`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")

    def run(self):
        args = self.args
        # Evaluate algorithm performance.
        total_mse_value = 0.0
        total_rmse_value = 0.0
        total_psnr_value = 0.0
        total_ssim_value = 0.0
        total_mssim_value = 0.0
        total_niqe_value = 0.0
        total_sam_value = 0.0
        total_vif_value = 0.0
        total_lpips_value = 0.0

        # Start evaluate model performance.
        progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))

        for i, (input, bicubic, target) in progress_bar:
            # Set model gradients to zero
            lr = input.to(self.device)
            bicubic = bicubic.to(self.device)
            hr = target.to(self.device)

            # Super-resolution.
            sr = inference(self.model, lr)

            # Evaluate performance
            if args.detail:
                vutils.save_image(sr, os.path.join("benchmark", "sr.bmp"))  # Save super resolution image.
                vutils.save_image(hr, os.path.join("benchmark", "hr.bmp"))  # Save high resolution image.
                value = image_quality_evaluation(sr_filename=os.path.join("benchmark", "sr.bmp"),
                                                 hr_filename=os.path.join("benchmark", "hr.bmp"),
                                                 device=self.device)

                total_mse_value += value[0]
                total_rmse_value += value[1]
                total_psnr_value += value[2]
                total_ssim_value += value[3][0]
                total_mssim_value += value[4].real
                total_niqe_value += value[5]
                total_sam_value += value[6]
                total_vif_value += value[7]
                total_lpips_value += value[8].item()
                progress_bar.set_description(f"[{i + 1}/{len(self.dataloader)}] "
                                             f"PSNR: {value[2]:.2f}dB "
                                             f"SSIM: {value[3][0]:.4f}")
            else:
                mse_value = ((sr - hr) ** 2).data.mean()
                psnr_value = 10 * math.log10(1. / mse_value)
                ssim_value = ssim(hr, sr)
                total_psnr_value += psnr_value
                total_ssim_value += ssim_value
                progress_bar.set_description(f"[{i + 1}/{len(self.dataloader)}] "
                                             f"PSNR: {psnr_value:.2f}dB SSIM: {ssim_value:.4f}.")

            images = torch.cat([bicubic, sr, hr], dim=-1)
            vutils.save_image(images, os.path.join("benchmark", f"{i + 1}.bmp"), padding=10)

        print(f"Performance avg results:\n")
        print(f"indicator Score\n")
        print(f"--------- -----\n")
        if args.detail:
            print(f"MSE       {total_mse_value / len(self.dataloader):.2f}\n"
                  f"RMSE      {total_rmse_value / len(self.dataloader):.2f}\n"
                  f"PSNR      {total_psnr_value / len(self.dataloader):.2f}\n"
                  f"SSIM      {total_ssim_value / len(self.dataloader):.4f}\n"
                  f"MS-SSIM   {total_mssim_value / len(self.dataloader):.4f}\n"
                  f"NIQE      {total_niqe_value / len(self.dataloader):.2f}\n"
                  f"SAM       {total_sam_value / len(self.dataloader):.4f}\n"
                  f"VIF       {total_vif_value / len(self.dataloader):.4f}\n"
                  f"LPIPS     {total_lpips_value / len(self.dataloader):.4f}\n")
        else:
            print(f"PSNR      {total_psnr_value / len(self.dataloader):.2f}\n"
                  f"SSIM      {total_ssim_value / len(self.dataloader):.4f}\n")


class Estimate(object):
    def __init__(self, args):
        self.args = args
        self.model, self.device = configure(args)

    def run(self):
        args = self.args
        # Read img to tensor and transfer to the specified device for processing.
        img = Image.open(args.lr)
        lr = process_image(img, self.device)

        sr, use_time = inference(self.model, lr, statistical_time=True)
        vutils.save_image(sr, os.path.join("test", f"{args.lr.split('/')[-1]}"))  # Save super resolution image.
        value = image_quality_evaluation(os.path.join("test", f"{args.lr.split('/')[-1]}"), args.hr, self.device)

        print(f"Performance avg results:\n")
        print(f"indicator Score\n")
        print(f"--------- -----\n")
        if self.args.detail:
            print(f"MSE       {value[0]:.2f}\n"
                  f"RMSE      {value[1]:.2f}\n"
                  f"PSNR      {value[2]:.2f}\n"
                  f"SSIM      {value[3][0]:.4f}\n"
                  f"MS-SSIM   {value[4].real:.4f}\n"
                  f"NIQE      {value[5]:.2f}\n"
                  f"SAM       {value[6]:.4f}\n"
                  f"VIF       {value[7]:.4f}\n"
                  f"LPIPS     {value[8].item():.4f}\n"
                  f"Use time: {use_time * 1000:.2f}ms | {use_time:.4f}s")
        else:
            print(f"PSNR      {value[0]:.2f}\n"
                  f"SSIM      {value[1][0]:.2f}\n"
                  f"Use time: {use_time * 1000:.2f}ms | {use_time:.4f}s")


class Video(object):
    def __init__(self, args):
        self.args = args
        self.model, self.device = configure(args)
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
                img = Image.open(args.lr)
                lr = process_image(img, self.device)

                sr = inference(self.model, lr)

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
