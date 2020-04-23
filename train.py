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

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from srgan_pytorch import Discriminator
from srgan_pytorch import FeatureExtractor
from srgan_pytorch import Generator
from srgan_pytorch import weights_init

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                    help="Number of data loading workers. (default:8)")
parser.add_argument("--epochs", default=100, type=int, metavar="N",
                    help="Number of total epochs to run. (default:100)")
parser.add_argument("--image-size", type=int, default=64,
                    help="Size of the data crop (squared assumed). (default:64)")
parser.add_argument("-b", "--batch-size", default=16, type=int,
                    metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (default:0.0001)")
parser.add_argument("--up-sampling", type=int, default=4,
                    help="Low to high resolution scaling factor. (default:4) Option: [2, 4, 8].")
parser.add_argument("-p", "--print-freq", default=100, type=int,
                    metavar="N", help="Print frequency. (default:100)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--netG", default="", help="Path to netG (to continue training).")
parser.add_argument("--netD", default="", help="Path to netD (to continue training).")
parser.add_argument("--outf", default="./outputs",
                    help="Folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use. (default:None)")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training")

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
                                   transforms.RandomCrop(args.image_size * args.up_sampling),
                                   transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, pin_memory=True, num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")
ngpu = int(args.ngpu)

generator = Generator(args.batch_size, args.up_sampling).to(device)
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
print(generator)
print(discriminator)

# define loss function (adversarial_loss) and optimizer
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(device)
content_loss = nn.MSELoss().to(device)
adversarial_loss = nn.BCELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize(args.image_size),
                             transforms.ToTensor(),
                             normalize,
                             ])

low_resolution_image = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

real_label = torch.ones(args.batch_size, 1, device=device)

# Pre-train generator using raw MSE loss
print("Generator pre-training for 2 epoch.")
for epoch in range(2):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # Set generator gradients to zero
        generator.zero_grad()
        # Generate data
        high_resolution_real_image = data[0].to(device)
        batch_size = high_resolution_real_image.size(0)

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

        progress_bar.set_description(f"[{epoch}/2][{i}/{len(dataloader) - 1}] "
                                     f"Generator_MSE_Loss: {generator_content_loss.item():.4f}")

    # Do checkpointing
    torch.save(generator.state_dict(), f"weights/generator_pretrained.pth")

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr * 0.1)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr * 0.1)

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        high_resolution_real_image = data[0].to(device)
        batch_size = high_resolution_real_image.size(0)

        # Down sample images to low resolution
        for batch_index in range(batch_size):
            low_resolution_image[batch_index] = resize(high_resolution_real_image[batch_index].cpu())
            high_resolution_real_image[batch_index] = normalize(high_resolution_real_image[batch_index])

        # Generate real and fake inputs
        high_resolution_fake_image = generator(low_resolution_image)
        real_image = torch.rand(batch_size, 1, device=device) * 0.5 + 0.7
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
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

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

        # Content based image loss value.
        image_content_loss = content_loss(high_resolution_fake_image, high_resolution_real_image)
        # Content based feature loss value.
        feature_content_loss = content_loss(fake_features, real_features) * 0.006
        # Combined real image content loss and fake image content loss. At the same time calculate gradients.
        generator_content_loss = image_content_loss + feature_content_loss
        # Calculate the difference between the generated image and the real image.
        generator_adversarial_loss = adversarial_loss(fake_output.detach(), real_label) * 0.001
        # Combined real image content loss and fake image content loss. At the same time calculate gradients.
        generator_total_loss = generator_content_loss + generator_adversarial_loss

        # Calculate gradients for generator
        generator_total_loss.backward()
        # Update generator weights
        optimizer_G.step()

        progress_bar.set_description(f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] "
                                     f"Loss_D: {discriminator_loss.item():.4f} "
                                     f"Loss_G_content: {generator_content_loss.item():.4f} "
                                     f"Loss_G_adversarial: {generator_adversarial_loss.item():.4f} "
                                     f"loss_G_total: {generator_total_loss.item():.4f}")

        if i % args.print_freq == 0:
            vutils.save_image(high_resolution_real_image,
                              f"{args.outf}/real_samples.png",
                              normalize=True)
            vutils.save_image(high_resolution_fake_image.detach(),
                              f"{args.outf}/fake_samples_epoch_{epoch}.png",
                              normalize=True)

    # do checkpointing
    torch.save(generator.state_dict(), f"weights/netG_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"weights/netD_epoch_{epoch}.pth")
