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
import random

import cv2
import torch.autograd.profiler as profiler
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sewar.full_ref import mse
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar.full_ref import vifp
from tqdm import tqdm

from srgan_pytorch import Discriminator
from srgan_pytorch import Generator
from srgan_pytorch import ContentLoss_VGG54
from srgan_pytorch import cal_niqe
from srgan_pytorch import DatasetFromFolder

parser = argparse.ArgumentParser(description="Photo-Realistic Single Image Super-Resolution "
                                             "Using a Generative Adversarial Network.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--psnr_iters", default=1e6, type=int, metavar="N",
                    help="The number of iterations is needed in the training of PSNR model. (default:1e6)")
parser.add_argument("--srgan_iters", default=2e5, type=int, metavar="N",
                    help="The training of srgan model requires the number of iterations. (default:2e5)")
parser.add_argument("-b", "--batch-size", default=16, type=int, metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate. (default:1e-4)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--cuda", action="store_true",
                    help="Enables cuda")
parser.add_argument("--netG", default="",
                    help="Path to netG (to continue training).")
parser.add_argument("--netD", default="",
                    help="Path to netD (to continue training).")
parser.add_argument("--outf", default="./output",
                    help="folder to output images. (default:`./output`).")
parser.add_argument("--manualSeed", type=int, default=0,
                    help="Seed for initializing training. (default:0)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs("weights")
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/train/input",
                                  target_dir=f"{args.dataroot}/{args.upscale_factor}x/train/target")
val_dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/val/input",
                                target_dir=f"{args.dataroot}/{args.upscale_factor}x/val/target")

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               num_workers=int(args.workers))
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             pin_memory=True,
                                             num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

netG = Generator(upscale_factor=args.upscale_factor).to(device)
netD = Discriminator().to(device)

if args.netG != "":
    netG.load_state_dict(torch.load(args.netG, map_location=device))
if args.netD != "":
    netD.load_state_dict(torch.load(args.netD, map_location=device))

# define loss function and optimizer
mse_loss = nn.MSELoss().to(device)
content_loss = ContentLoss_VGG54().to(device)
adversarial_loss = nn.BCELoss().to(device)

# According to the requirements of the original paper
optimizer = torch.optim.Adam(netG.parameters(), lr=args.lr)

# Pre-train generator using raw MSE loss
psnr_epochs = int(args.psnr_iters // len(train_dataloader))
save_interval = int(psnr_epochs // 5)
print("[*] Start training PSNR model based on MSE loss.")
print(f"[*] Generator pre-training for {psnr_epochs} epochs.")
print(f"[*] Searching PSNR pretrained model weights.")

# Save the generator model based on MSE pre training to speed up the training time
if os.path.exists(f"./weights/SRResNet_{args.upscale_factor}x.pth"):
    print("[*] Found PSNR pretrained model weights. Skip pre-train.")
    netG.load_state_dict(torch.load(f"./weights/SRResNet_{args.upscale_factor}x.pth", map_location=device))
else:
    print("[!] Not found pretrained weights. Start training PSNR model.")
    for epoch in range(psnr_epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (input, target) in progress_bar:
            # Set generator gradients to zero
            netG.zero_grad()
            # Generate data
            input = input[0].to(device)
            target = target[1].to(device)

            # Generating fake high resolution images from real low resolution images.
            output = netG(input)
            # The MSE of the generated fake high-resolution image and real high-resolution image is calculated.
            loss = mse_loss(output, target)
            # Calculate gradients for generator
            loss.backward()
            # Update generator weights
            optimizer.step()

            # The image is saved every 500 iterations.
            if (i + 1) % 500 == 0:
                vutils.save_image(output, f"{args.outf}/hr_fake_epoch_{epoch}.png", normalize=True)
                vutils.save_image(target, f"{args.outf}/hr_real_epoch_{epoch}.png", normalize=True)

            progress_bar.set_description(f"[{epoch + 1}/{psnr_epochs}][{i + 1}/{len(train_dataloader)}] "
                                         f"MSE loss: {loss.item():.6f}")

        # The model is saved every 200000 iterations.
        if (epoch + 1) % save_interval == 0:
            torch.save(netG.state_dict(), f"./weights/SRResNet_{args.upscale_factor}x_epoch_{epoch}.pth")
    torch.save(netG.state_dict(), f"./weights/SRResNet_{args.upscale_factor}x.pth")
    print(f"[*] Training PSNR model done! Saving PSNR model weight to `./weights/SRResNet_{args.upscale_factor}x.pth`.")

mse_list = []
rmse_list = []
psnr_list = []
ssim_list = []
ms_ssim_list = []
niqe_list = []
sam_list = []
vif_list = []

best_psnr_value = 0.0
best_ssim_value = 0.0

# Alternating training SRGAN network.
srgan_epochs = int(args.srgan_iters // len(train_dataloader))
save_interval = int(srgan_epochs // 5)
optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr)
optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=srgan_epochs // 2, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=srgan_epochs // 2, gamma=0.1)

print(f"[*] Staring training SRGAN model!")
print(f"[*] Training for {srgan_epochs} epochs.")
for epoch in range(0, srgan_epochs):
    # Evaluate algorithm performance
    total_mse_value = 0.0
    total_rmse_value = 0.0
    total_psnr_value = 0.0
    total_ssim_value = 0.0
    total_ms_ssim_value = 0.0
    total_niqe_value = 0.0
    total_sam_value = 0.0
    total_vif_value = 0.0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (input, target) in progress_bar:
        input = input[0].to(device)
        target = target[1].to(device)

        ##############################################
        # (1) Update D network
        ##############################################
        output = netG(input)

        # Set discriminator gradients to zero.
        netD.zero_grad()

        real_out = netD(input).mean()
        fake_out = netD(output).mean()
        d_loss = 1 - real_out + fake_out

        # Calculate gradients for discriminator.
        d_loss.backward(retain_graph=True)
        # Update discriminator weights.
        optimizer_D.step()

        ##############################################
        # (2) Update G network
        ##############################################
        # Set generator gradients to zero
        netG.zero_grad()

        output = netG(input)
        fake_out = netD(output).mean()

        g_loss = generator_loss(fake_out, output, target)
        # Calculate gradients for generator.
        g_loss.backward()
        # Update generator weights.
        optimizer_G.step()

        # Dynamic adjustment of learning rate
        scheduler_G.step()
        scheduler_D.step()

        # The image is saved every 500 iterations.
        if (i + 1) % 500 == 0:
            vutils.save_image(output, f"{args.outf}/hr_fake_epoch_{epoch}.png", normalize=True)
            vutils.save_image(target, f"{args.outf}/hr_real_epoch_{epoch}.png", normalize=True)

        progress_bar.set_description(f"[{epoch + 1}/{args.epochs}][{i + 1}/{len(train_dataloader)}] "
                                     f"Loss_D: {d_loss.item():.6f} "
                                     f"Loss_G: {g_loss.item():.6f}")

        # The model is saved every 20000 iterations.
        if (epoch + 1) % save_interval == 0:
            torch.save(netG.state_dict(), f"./weights/SRResNet_{args.upscale_factor}x_epoch_{epoch}.pth")
    torch.save(netG.state_dict(), f"./weights/SRResNet_{args.upscale_factor}x.pth")
    print(f"[*] Training PSNR model done! Saving PSNR model weight to `./weights/SRResNet_{args.upscale_factor}x.pth`.")

    # Start evaluate model performance
#     with torch.no_grad():
#         netG.eval()
#         netD.eval()
#         for i, data in enumerate(val_dataloader):
#             lr_real_image = data[0].to(device)
#             hr_restore_image = data[1].to(device)
#             hr_real_image = data[2].to(device)
#
#             hr_fake_image = netG(lr_real_image)
#
#             vutils.save_image(lr_real_image, f"{args.outf}/lr_real_epoch_{epoch}.png", normalize=True)
#             vutils.save_image(hr_real_image, f"{args.outf}/hr_real_epoch_{epoch}.png", normalize=True)
#             vutils.save_image(hr_restore_image, f"{args.outf}/hr_restore_epoch_{epoch}.png", normalize=True)
#             vutils.save_image(hr_fake_image, f"{args.outf}/hr_fake_epoch_{epoch}.png", normalize=True)
#
#             src_img = cv2.imread(f"{args.outf}/hr_fake_epoch_{epoch}.png")
#             dst_img = cv2.imread(f"{args.outf}/hr_real_epoch_{epoch}.png")
#
#             mse_value = mse(src_img, dst_img)
#             rmse_value = rmse(src_img, dst_img)
#             psnr_value = psnr(src_img, dst_img)
#             ssim_value = ssim(src_img, dst_img)
#             ms_ssim_value = msssim(src_img, dst_img)
#             niqe_value = cal_niqe(f"{args.outf}/hr_fake_epoch_{epoch}.png")
#             sam_value = sam(src_img, dst_img)
#             vif_value = vifp(src_img, dst_img)
#
#             total_mse_value += mse_value
#             total_rmse_value += rmse_value
#             total_psnr_value += psnr_value
#             total_ssim_value += ssim_value
#             total_ms_ssim_value += ms_ssim_value
#             total_niqe_value += niqe_value
#             total_sam_value += sam_value
#             total_vif_value += vif_value
#
#         # do checkpointing
#         if (epoch + 1) % args.print_freq == 0:
#             print(f"[*] Save SRGAN model!")
#             torch.save(netG.state_dict(), f"./weights/netG_{args.upscale_factor}x_epoch_{epoch + 1}.pth")
#             torch.save(netD.state_dict(), f"./weights/netD_{args.upscale_factor}x_epoch_{epoch + 1}.pth")
#
#     avg_mse_value = total_mse_value / len(val_dataloader)
#     avg_rmse_value = total_rmse_value / len(val_dataloader)
#     avg_psnr_value = total_psnr_value / len(val_dataloader)
#     avg_ssim_value = total_ssim_value / len(val_dataloader)
#     avg_ms_ssim_value = total_ms_ssim_value / len(val_dataloader)
#     avg_niqe_value = total_niqe_value / len(val_dataloader)
#     avg_sam_value = total_sam_value / len(val_dataloader)
#     avg_vif_value = total_vif_value / len(val_dataloader)
#
#     print("\n")
#     print("====================== Performance summary ======================")
#     print(f"======================   Epoch {epoch}    ======================")
#     print("=================================================================")
#     print(f"Avg MSE: {avg_mse_value:.2f}\n"
#           f"Avg RMSE: {avg_rmse_value:.2f}\n"
#           f"Avg PSNR: {avg_psnr_value:.2f}\n"
#           f"Avg SSIM: {avg_ssim_value:.4f}\n"
#           f"Avg MS-SSIM: {avg_ms_ssim_value:.4f}\n"
#           f"Avg NIQE: {avg_niqe_value:.2f}\n"
#           f"Avg SAM: {avg_sam_value:.4f}\n"
#           f"Avg VIF: {avg_vif_value:.4f}")
#     print("============================== End ==============================")
#     print("\n")
#
#     # save best model
#     if best_psnr_value < avg_psnr_value and best_ssim_value < avg_ssim_value:
#         best_psnr_value = avg_psnr_value
#         best_ssim_value = avg_ssim_value
#         torch.save(netG.state_dict(), f"weights/SRGAN_{args.upscale_factor}x.pth")
#
#     mse_list.append(total_mse_value / len(val_dataloader))
#     rmse_list.append(total_rmse_value / len(val_dataloader))
#     psnr_list.append(total_psnr_value / len(val_dataloader))
#     rmse_list.append(total_ssim_value / len(val_dataloader))
#     ms_ssim_list.append(total_ms_ssim_value / len(val_dataloader))
#     niqe_list.append(total_niqe_value / len(val_dataloader))
#     sam_list.append(total_sam_value / len(val_dataloader))
#     vif_list.append(total_vif_value / len(val_dataloader))
#
# plt.figure(figsize=(50, 40))
# plt.title("Model performance")
# plt.plot(mse_list, label="MSE")
# plt.plot(rmse_list, label="RMSE")
# plt.plot(psnr_list, label="PSNR")
# plt.plot(ssim_list, label="SSIM")
# plt.plot(ms_ssim_list, label="MS-SSIM")
# plt.plot(niqe_list, label="NIQE")
# plt.plot(sam_list, label="SAM")
# plt.plot(vif_list, label="VIF")
# plt.xlabel("Epochs")
# plt.ylabel("Value")
# plt.legend()
# plt.savefig("train_performance_result.png")
