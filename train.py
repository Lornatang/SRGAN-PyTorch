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

import torch.autograd.profiler as profiler
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from matplotlib import pyplot as plt

from srgan_pytorch import Discriminator
from srgan_pytorch import FeatureExtractor
from srgan_pytorch import Generator
from srgan_pytorch import cal_psnr
from srgan_pytorch import cal_ssim
from srgan_pytorch import weights_init

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data/DIV2K",
                    help="Path to datasets. (default:`./data/DIV2K`)")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="Number of data loading workers. (default:0)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="Number of total epochs to run. (default:200)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Size of the data crop (squared assumed). (default:96)")
parser.add_argument("-b", "--batch-size", default=16, type=int,
                    metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (default:0.0001)")
parser.add_argument("--scale-factor", type=int, default=4,
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("-p", "--print-freq", default=50, type=int,
                    metavar="N", help="Print frequency. (default:100)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--netG", default="",
                    help="Path to netG (to continue training).")
parser.add_argument("--netD", default="",
                    help="Path to netD (to continue training).")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")
parser.add_argument("--ngpu", default=1, type=int,
                    help="GPU id to use. (default:1)")

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
    print("WARNING: You have a CUDA device, "
          "so you should probably run with --cuda")

dataset = datasets.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.RandomResizedCrop(
                                       args.image_size * args.scale_factor),
                                   transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, pin_memory=True,
                                         num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")
ngpu = int(args.ngpu)

netG = Generator(8, args.scale_factor).to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)

if args.netG != "":
    netG.load_state_dict(torch.load(args.netG))
if args.netD != "":
    netD.load_state_dict(torch.load(args.netD))

# Memory call of analysis model
lr = torch.randn(args.batch_size, 3, args.image_size, args.image_size,
                 device=device)
with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    netG(lr)
print(" # Generator # ")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

hr = torch.randn(args.batch_size, 3, args.image_size * args.scale_factor,
                 args.image_size * args.scale_factor, device=device)
with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    netD(hr)
print(" # Discriminator # ")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

# define loss function (adversarial_loss) and optimizer
feature_extractor = FeatureExtractor(torchvision.models.vgg19(True)).to(device)
content_loss = nn.MSELoss().to(device)
adv_loss = nn.BCELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr)
optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize(args.image_size),
                             transforms.ToTensor(),
                             normalize,
                             ])

# Pre-train generator using raw MSE loss
print(f"Generator pre-training for 8 epochs.")
print(f"Find generator pretrained model weights.")

if os.path.exists(f"./weights/pretrained_X{args.scale_factor}.pth"):
    print("Found pretrained weights. Skip pre-train.")
    netG.load_state_dict(
        torch.load(f"./weights/pretrained_X{args.scale_factor}.pth"))
else:
    print("Not found pretrained weights. start training.")
    for epoch in range(8):
        for i, data in enumerate(dataloader):
            # Set generator gradients to zero
            netG.zero_grad()
            # Generate data
            hr_real_image = data[0].to(device)
            batch_size = hr_real_image.size(0)

            lr_fake_image = torch.randn(batch_size, 3, args.image_size,
                                        args.image_size, device=device)

            # Down sample images to low resolution
            for batch_i in range(batch_size):
                lr_fake_image[batch_i] = resize(hr_real_image[batch_i].cpu())
                hr_real_image[batch_i] = normalize(hr_real_image[batch_i])

            # Generate real and fake inputs
            hr_fake_image = netG(lr_fake_image)

            # Content loss
            g_content_loss = content_loss(hr_fake_image, hr_real_image)

            # Calculate gradients for generator
            g_content_loss.backward()
            # Update generator weights
            optimizer_G.step()

            if i % args.print_freq == 0:
                print(f"[{epoch}/{8}][{i}/{len(dataloader)}] "
                      f"Generator mse loss: {g_content_loss.item():.4f}")

    torch.save(netG, f"./weights/pretrained_X{args.scale_factor}.pth")
    print(f"Saving generator weights to "
          f"`./weights/pretrained_X{args.scale_factor}.pth")

g_losses = []
d_losses = []

psnr_list = []
ssim_list = []

best_psnr_value = 0.0
best_ssim_value = 0.0

for epoch in range(0, args.epochs):
    for i, data in enumerate(dataloader):
        hr_real_image = data[0].to(device)
        batch_size = hr_real_image.size(0)

        lr_fake_image = torch.randn(batch_size, 3, args.image_size,
                                    args.image_size, device=device)

        # Down sample images to low resolution
        for batch_i in range(batch_size):
            lr_fake_image[batch_i] = resize(hr_real_image[batch_i].cpu())
            hr_real_image[batch_i] = normalize(hr_real_image[batch_i])

        # Generate real and fake inputs
        hr_fake_image = netG(lr_fake_image)
        real_image = torch.rand(batch_size, 1, device=device) * 0.5 + 0.7
        real_label = torch.ones(batch_size, 1, device=device)
        fake_image = torch.rand(batch_size, 1, device=device) * 0.3

        ##############################################
        # (1) Update D network
        ##############################################
        # Set discriminator gradients to zero.
        netD.zero_grad()

        # Real image loss.
        real_output = netD(hr_real_image)
        d_real_loss = adv_loss(real_output, real_image)

        # Fake image loss.
        fake_output = netD(hr_fake_image)
        d_fake_loss = adv_loss(fake_output.detach(), fake_image)
        # Combined real image loss and fake image loss.
        # At the same time calculate gradients.
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate gradients for discriminator.
        d_loss.backward()
        # Update discriminator weights.
        optimizer_D.step()

        ##############################################
        # (2) Update G network
        ##############################################
        # Set generator gradients to zero
        netG.zero_grad()

        # Extract picture features
        real_features = feature_extractor(hr_real_image)
        fake_features = feature_extractor(hr_fake_image)

        image_content_loss = content_loss(hr_fake_image,
                                          hr_real_image)
        feature_content_loss = content_loss(fake_features,
                                            real_features) * 0.006
        # Combined real image content loss and fake image content loss.
        # At the same time calculate gradients.
        g_content_loss = image_content_loss + feature_content_loss

        # Calculate the difference between the generated image and the real
        # image.
        g_adversarial_loss = adv_loss(fake_output.detach(), real_label) * 0.001
        # Combined real image content loss and fake image content loss.
        # At the same time calculate gradients.
        g_loss = g_content_loss + g_adversarial_loss

        # Calculate gradients for generator
        g_loss.backward()
        # Update generator weights
        optimizer_G.step()

        if i % args.print_freq == 0:
            print(f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] "
                  f"Loss_D: {d_loss.item():.4f} "
                  f"loss_G: {g_loss.item():.4f}")
            # Save Losses for plotting later
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            vutils.save_image(hr_real_image,
                              f"{args.outf}/real_samples.png",
                              normalize=True)
            vutils.save_image(hr_fake_image.detach(),
                              f"{args.outf}/fake_samples_epoch_{epoch}.png",
                              normalize=True)

    psnr_value = cal_psnr(f"{args.outf}/fake_samples_epoch_{epoch}.png",
                          f"{args.outf}/real_samples.png")
    ssim_value = cal_ssim(f"{args.outf}/fake_samples_epoch_{epoch}.png",
                          f"{args.outf}/real_samples.png")
    print("\n")
    print("============================ Summary ============================")
    print(f"Iter: {len(dataloader) * (epoch + 1)} "
          f"PSNR: {psnr_value:.4f} "
          f"SSIM: {ssim_value:.4f}")
    print("============================== End ==============================")
    print("\n")

    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)

    # do checkpointing
    torch.save(netG.state_dict(), f"weights/netG_epoch_{epoch}.pth")
    torch.save(netD.state_dict(), f"weights/netD_epoch_{epoch}.pth")

    # save best model
    if best_psnr_value < psnr_value and best_ssim_value < ssim_value:
        best_psnr_value = psnr_value
        best_ssim_value = ssim_value
        shutil.copyfile(f"weights/netG_epoch_{epoch}.pth",
                        f"weights/netG_X{args.scale_factor}.pth")
        shutil.copyfile(f"weights/netG_X{args.scale_factor}.pth",
                        f"weights/srgan_X{args.scale_factor}.pth")
        shutil.copyfile(f"weights/netD_epoch_{epoch}.pth",
                        f"weights/netD_X{args.scale_factor}.pth")

plt.figure(figsize=(50, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="G_Loss")
plt.plot(d_losses, label="D_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("model_loss_result.png")

plt.figure(figsize=(20, 5))
plt.title("Model performance")
plt.plot(psnr_list, label="PSNR")
plt.plot(ssim_list, label="SSIM")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.savefig("model_performance_result.png")
