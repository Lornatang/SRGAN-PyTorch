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
import shutil

import cv2
import torch.autograd.profiler as profiler
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sewar.full_ref import msssim
from sewar.full_ref import sam
from sewar.full_ref import vifp

from srgan_pytorch import Discriminator
from srgan_pytorch import Generator
from srgan_pytorch import GeneratorLoss
from srgan_pytorch import TrainDatasetFromFolder
from srgan_pytorch import ValDatasetFromFolder
from srgan_pytorch import cal_mse
from srgan_pytorch import cal_niqe
from srgan_pytorch import cal_psnr
from srgan_pytorch import cal_rmse
from srgan_pytorch import cal_ssim
from srgan_pytorch import calculate_valid_crop_size

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data/DIV2K",
                    help="Path to datasets. (default:`./data/DIV2K`)")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="Number of data loading workers. (default:0)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="Number of total epochs to run. (default:200)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Size of the data crop (squared assumed). (default:96)")
parser.add_argument("-b", "--batch-size", default=16, type=int, metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (default:0.0001)")
parser.add_argument("--scale-factor", type=int, default=4, choices=[2, 4, 8, 16],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("-p", "--print-freq", default=5, type=int, metavar="N",
                    help="Print frequency. (default:5)")
parser.add_argument("--cuda", action="store_true",
                    help="Enables cuda")
parser.add_argument("--netG", default="",
                    help="Path to netG (to continue training).")
parser.add_argument("--netD", default="",
                    help="Path to netD (to continue training).")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

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

train_dataset = TrainDatasetFromFolder(dataset_dir=f'{args.dataroot}/train',
                                       crop_size=args.image_size,
                                       upscale_factor=args.scale_factor)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=int(args.workers))

val_dataset = ValDatasetFromFolder(dataset_dir=f'{args.dataroot}/val',
                                   upscale_factor=args.scale_factor)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

netG = Generator(scale_factor=args.scale_factor).to(device)
netD = Discriminator().to(device)

if args.netG != "":
    netG.load_state_dict(torch.load(args.netG, map_location=device))
if args.netD != "":
    netD.load_state_dict(torch.load(args.netD, map_location=device))

# Memory call of analysis model
lr_image_size = calculate_valid_crop_size(args.image_size, args.scale_factor)
lr = torch.randn(args.batch_size, 3, lr_image_size, lr_image_size, device=device)

with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    netG(lr)
print(" # Generator # ")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

hr_image_size = calculate_valid_crop_size(args.image_size, args.scale_factor) * args.scale_factor
hr = torch.randn(args.batch_size, 3, hr_image_size, hr_image_size, device=device)
with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    netD(hr)
print(" # Discriminator # ")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

# define loss function (adversarial_loss) and optimizer
generator_loss = GeneratorLoss().to(device)
mse_loss = nn.MSELoss().to(device)

optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr)

# Pre-train generator using raw MSE loss
print(f"Generator pre-training for {args.epochs // 10} epochs.")
print(f"Searching generator pretrained model weights.")

if os.path.exists(f"./weights/pretrained_X{args.scale_factor}.pth"):
    print("Found pretrained weights. Skip pre-train.")
    netG.load_state_dict(torch.load(f"./weights/pretrained_X{args.scale_factor}.pth", map_location=device))
else:
    print("Not found pretrained weights. start training.")
    for epoch in range(0, args.epochs // 10):
        for i, data in enumerate(train_dataloader):
            # Set generator gradients to zero
            netG.zero_grad()
            # Generate data
            lr_real_image = data[0].to(device)
            hr_real_image = data[1].to(device)

            # Generate real and fake inputs
            hr_fake_image = netG(lr_real_image)

            # Content loss
            g_loss = mse_loss(hr_fake_image, hr_real_image)

            # Calculate gradients for generator
            g_loss.backward()
            # Update generator weights
            optimizer_G.step()

            if i % args.print_freq == 0:
                print(f"[{epoch}/{args.epochs // 10}][{i}/{len(train_dataloader)}] "
                      f"Generator MSE loss: {g_loss.item():.4f}")

    torch.save(netG.state_dict(), f"./weights/pretrained_X{args.scale_factor}.pth")
    print(f"Saving pre-train weights to `./weights/pretrained_X{args.scale_factor}.pth`.")

optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr * 0.1)
optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr * 0.1)

g_losses = []
d_losses = []

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

for epoch in range(0, args.epochs):
    # Evaluate algorithm performance
    total_mse_value = 0.0
    total_rmse_value = 0.0
    total_psnr_value = 0.0
    total_ssim_value = 0.0
    total_ms_ssim_value = 0.0
    total_niqe_value = 0.0
    total_sam_value = 0.0
    total_vif_value = 0.0
    for i, data in enumerate(train_dataloader):
        lr_real_image = data[0].to(device)
        hr_real_image = data[1].to(device)

        ##############################################
        # (1) Update D network
        ##############################################
        hr_fake_image = netG(lr_real_image)

        # Set discriminator gradients to zero.
        netD.zero_grad()

        real_out = netD(lr_real_image).mean()
        fake_out = netD(hr_fake_image).mean()
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

        hr_fake_image = netG(lr_real_image)
        fake_out = netD(hr_fake_image).mean()

        g_loss = generator_loss(fake_out, hr_fake_image, hr_real_image)
        # Calculate gradients for generator.
        g_loss.backward()

        # Update generator weights.
        optimizer_G.step()

        if i % args.print_freq == 0:
            print(f"====================== [{epoch}/{args.epochs}][{i}/{len(train_dataloader)}] ========"
                  "Loss_D: {d_loss.item():.4f} loss_G: {g_loss.item():.4f}\n")
            print("============================== End ==============================")
            print("\n")

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

    # Start evaluate model performance
    with torch.no_grad():
        netG.eval()
        netD.eval()
        for i, data in enumerate(val_dataloader):
            lr_real_image = data[0].to(device)
            hr_restore_image = data[1].to(device)
            hr_real_image = data[2].to(device)

            hr_fake_image = netG(lr_real_image)

            vutils.save_image(hr_real_image, f"{args.outf}/hr_real_epoch_{epoch}.png", normalize=True)
            vutils.save_image(hr_restore_image, f"{args.outf}/hr_restore_epoch_{epoch}.png", normalize=True)
            vutils.save_image(hr_fake_image, f"{args.outf}/hr_fake_epoch_{epoch}.png", normalize=True)

            src_img = cv2.imread(f"{args.outf}/hr_fake_epoch_{epoch}.png")
            dst_img = cv2.imread(f"{args.outf}/hr_real_epoch_{epoch}.png")

            mse_value = cal_mse(src_img, dst_img)
            rmse_value = cal_rmse(src_img, dst_img)
            psnr_value = cal_psnr(src_img, dst_img)
            ssim_value = cal_ssim(src_img, dst_img)
            ms_ssim_value = msssim(src_img, dst_img)
            niqe_value = cal_niqe(f"{args.outf}/hr_fake_epoch_{epoch}.png")
            sam_value = sam(src_img, dst_img)
            vif_value = vifp(src_img, dst_img)

            total_mse_value += mse_value
            total_rmse_value += rmse_value
            total_psnr_value += psnr_value
            total_ssim_value += ssim_value
            total_ms_ssim_value += ms_ssim_value
            total_niqe_value += niqe_value
            total_sam_value += sam_value
            total_vif_value += vif_value

        # do checkpointing
        torch.save(netG.state_dict(), f"weights/srgan_G_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"weights/srgan_D_epoch_{epoch}.pth")

    avg_mse_value = total_mse_value / len(val_dataloader)
    avg_rmse_value = total_rmse_value / len(val_dataloader)
    avg_psnr_value = total_psnr_value / len(val_dataloader)
    avg_ssim_value = total_ssim_value / len(val_dataloader)
    avg_ms_ssim_value = total_ms_ssim_value / len(val_dataloader)
    avg_niqe_value = total_niqe_value / len(val_dataloader)
    avg_sam_value = total_sam_value / len(val_dataloader)
    avg_vif_value = total_vif_value / len(val_dataloader)

    print("\n")
    print("====================== Performance summary ======================")
    print(f"======================    Epoch {epoch}   =======================")
    print("=================================================================")
    print(f"Avg MSE: {avg_mse_value:.2f}\n"
          f"Avg RMSE: {avg_rmse_value:.2f}\n"
          f"Avg PSNR: {avg_psnr_value:.2f}\n"
          f"Avg SSIM: {avg_ssim_value:.4f}\n"
          f"Avg MS-SSIM: {avg_ms_ssim_value:.4f}\n"
          f"Avg NIQE: {avg_niqe_value:.2f}\n"
          f"Avg SAM: {avg_sam_value:.4f}\n"
          f"Avg VIF: {avg_vif_value:.4f}")
    print("============================== End ==============================")
    print("\n")

    # save best model
    if best_psnr_value < avg_psnr_value and best_ssim_value < avg_ssim_value:
        best_psnr_value = avg_psnr_value
        best_ssim_value = avg_ssim_value
        shutil.copyfile(f"weights/srgan_D_epoch_{epoch}.pth",
                        f"weights/srgan_D_X{args.scale_factor}.pth")
        shutil.copyfile(f"weights/srgan_G_epoch_{epoch}.pth",
                        f"weights/srgan_G_X{args.scale_factor}.pth")

    mse_list.append(total_mse_value / len(val_dataloader))
    rmse_list.append(total_rmse_value / len(val_dataloader))
    psnr_list.append(total_psnr_value / len(val_dataloader))
    rmse_list.append(total_ssim_value / len(val_dataloader))
    ms_ssim_list.append(total_ms_ssim_value / len(val_dataloader))
    niqe_list.append(total_niqe_value / len(val_dataloader))
    sam_list.append(total_sam_value / len(val_dataloader))
    vif_list.append(total_vif_value / len(val_dataloader))

plt.figure(figsize=(50, 2))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="G_Loss")
plt.plot(d_losses, label="D_Loss")
plt.xlabel("Iter")
plt.ylabel("Loss")
plt.legend()
plt.savefig("model_loss_result.png")

plt.figure(figsize=(50, 40))
plt.title("Model performance")
plt.plot(mse_list, label="MSE")
plt.plot(rmse_list, label="RMSE")
plt.plot(psnr_list, label="PSNR")
plt.plot(ssim_list, label="SSIM")
plt.plot(ms_ssim_list, label="MS-SSIM")
plt.plot(niqe_list, label="NIQE")
plt.plot(sam_list, label="SAM")
plt.plot(vif_list, label="VIF")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.savefig("train_performance_result.png")
