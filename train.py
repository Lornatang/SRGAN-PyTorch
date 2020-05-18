# Copyright 2020 Lorna Authors. All Rights Reserved.
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

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from tqdm import tqdm

from srgan_pytorch import Discriminator
from srgan_pytorch import FeatureExtractor
from srgan_pytorch import Generator
from srgan_pytorch import evaluate_performance
from srgan_pytorch import weights_init

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                    help="Number of data loading workers. (default:8)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="Number of total epochs to run. (default:200)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Size of the data crop (squared assumed). (default:96)")
parser.add_argument("-b", "--batch-size", default=8, type=int,
                    metavar="N",
                    help="mini-batch size (default: 8), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (default:0.0001)")
parser.add_argument("--up-sampling", type=int, default=4,
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("-p", "--print-freq", default=50, type=int,
                    metavar="N", help="Print frequency. (default:100)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--netG", default="", help="Path to netG (to continue training).")
parser.add_argument("--netD", default="", help="Path to netD (to continue training).")
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
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = datasets.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.RandomResizedCrop(args.image_size * args.up_sampling),
                                   transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, pin_memory=True, num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")
ngpu = int(args.ngpu)

generator = Generator(n_residual_blocks=8, upsample_factor=args.up_sampling).to(device)
discriminator = Discriminator().to(device)

if args.cuda and ngpu > 1:
    generator = torch.nn.DataParallel(generator).to(device)
    discriminator = torch.nn.DataParallel(discriminator).to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

if args.netG != "":
    generator.load_state_dict(torch.load(args.netG))
if args.netD != "":
    discriminator.load_state_dict(torch.load(args.netD))

# define loss function (adversarial_loss) and optimizer
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(device)
content_loss = nn.MSELoss().to(device)
adversarial_loss = nn.BCELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize(args.image_size),
                             transforms.ToTensor(),
                             normalize,
                             ])

# Pre-train generator using raw MSE loss
pre_epochs = round(100 // args.batch_size)
print(f"Generator pre-training for {pre_epochs} epochs.")
for epoch in range(pre_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # Set generator gradients to zero
        generator.zero_grad()
        # Generate data
        high_resolution_real_image = data[0].to(device)
        batch_size = high_resolution_real_image.size(0)

        low_resolution_image = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)

        # Down sample images to low resolution
        for batch_index in range(batch_size):
            low_resolution_image[batch_index] = resize(high_resolution_real_image[batch_index].cpu())
            high_resolution_real_image[batch_index] = normalize(high_resolution_real_image[batch_index])

        # Generate real and fake inputs
        high_resolution_fake_image = generator(low_resolution_image)

        # Content loss
        generator_content_loss = content_loss(high_resolution_fake_image, high_resolution_real_image)

        # Calculate gradients for generator
        generator_content_loss.backward()
        # Update generator weights
        optimizer_G.step()

        progress_bar.set_description(f"[{epoch}/{pre_epochs}][{i}/{len(dataloader)}] "
                                     f"Generator_MSE_Loss: {generator_content_loss.item():.4f}")

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr * 0.1)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr * 0.1)

g_losses = []
d_losses = []

mse_list = []
psnr_list = []

best_mse_value = 0.0
best_psnr_value = 0.0

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        high_resolution_real_image = data[0].to(device)
        batch_size = high_resolution_real_image.size(0)

        low_resolution_image = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)

        # Down sample images to low resolution
        for batch_index in range(batch_size):
            low_resolution_image[batch_index] = resize(high_resolution_real_image[batch_index].cpu())
            high_resolution_real_image[batch_index] = normalize(high_resolution_real_image[batch_index])

        # Generate real and fake inputs
        high_resolution_fake_image = generator(low_resolution_image)
        real_image = torch.rand(batch_size, 1, device=device) * 0.5 + 0.7
        real_label = torch.ones(batch_size, 1, device=device)
        fake_image = torch.rand(batch_size, 1, device=device) * 0.3

        ##############################################
        # (1) Update D network
        ##############################################
        # Set discriminator gradients to zero.
        discriminator.zero_grad()

        # Real image loss.
        real_output = discriminator(high_resolution_real_image)
        discriminator_real_loss = adversarial_loss(real_output, real_image)
        # Fake image loss.
        fake_output = discriminator(high_resolution_fake_image)
        discriminator_fake_loss = adversarial_loss(fake_output.detach(), fake_image)
        # Combined real image loss and fake image loss. At the same time calculate gradients.
        discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2

        # Calculate gradients for discriminator.
        discriminator_loss.backward()
        # Update discriminator weights.
        optimizer_D.step()

        ##############################################
        # (2) Update G network
        ##############################################
        # Set generator gradients to zero
        generator.zero_grad()

        # Extract picture features
        real_features = feature_extractor(high_resolution_real_image)
        fake_features = feature_extractor(high_resolution_fake_image)

        image_content_loss = content_loss(high_resolution_fake_image, high_resolution_real_image)
        feature_content_loss = content_loss(fake_features, real_features) * 0.006
        # Combined real image content loss and fake image content loss. At the same time calculate gradients.
        generator_content_loss = image_content_loss + feature_content_loss

        # Calculate the difference between the generated image and the real image.
        generator_adversarial_loss = adversarial_loss(fake_output.detach(), real_label) * 0.001
        # Combined real image content loss and fake image content loss. At the same time calculate gradients.
        generator_loss = generator_content_loss + generator_adversarial_loss

        # Calculate gradients for generator
        generator_loss.backward()
        # Update generator weights
        optimizer_G.step()

        progress_bar.set_description(f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] "
                                     f"Loss_D: {discriminator_loss.item():.4f} "
                                     f"loss_G: {generator_loss.item():.4f}")

        if i % args.print_freq == 0:
            # Save Losses for plotting later
            d_losses.append(discriminator_loss.item())
            g_losses.append(generator_loss.item())

            vutils.save_image(high_resolution_real_image,
                              f"{args.outf}/real_samples.png",
                              normalize=True)
            vutils.save_image(high_resolution_fake_image.detach(),
                              f"{args.outf}/fake_samples_epoch_{epoch}.png",
                              normalize=True)

    mse_value, psnr_value = evaluate_performance(f"{args.outf}/real_samples.png",
                                                 f"{args.outf}/fake_samples_epoch_{epoch}.png")
    print("\n")
    print("================================== Summary ==================================")
    print(f"Iter: {len(dataloader) * (epoch + 1)} MSE: {mse_value:.4f}, PSNR: {psnr_value:.4f}")
    print("==================================== End ====================================")
    print("\n")

    mse_list.append(mse_value)
    psnr_list.append(psnr_value)

    # do checkpointing
    if ngpu > 1:
        torch.save(generator.module.state_dict(), f"weights/netG_epoch_{epoch}.pth")
        torch.save(discriminator.module.state_dict(), f"weights/netD_epoch_{epoch}.pth")
    else:
        torch.save(generator.state_dict(), f"weights/netG_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"weights/netD_epoch_{epoch}.pth")

    # save best model
    if best_mse_value < mse_value and best_psnr_value < psnr_value:
        best_mse_value = mse_value
        best_psnr_value = psnr_value
        shutil.copyfile(f"weights/netG_epoch_{epoch}.pth", "weights/netG.pth")
        shutil.copyfile(f"weights/netD_epoch_{epoch}.pth", "weights/netD.pth")

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
plt.plot(mse_list, label="MSE")
plt.plot(psnr_list, label="PSNR")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.savefig("model_performance_result.png")
