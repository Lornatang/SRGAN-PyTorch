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
import argparse
import logging
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tensorboardX import SummaryWriter

from srgan_pytorch.dataset import BaseDataset
from srgan_pytorch.loss import ContentLoss
from srgan_pytorch.models import discriminator_for_vgg
from srgan_pytorch.models import srgan
from srgan_pytorch.utils.common import AverageMeter
from srgan_pytorch.utils.common import ProgressMeter
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.estimate import test

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

best_psnr = 0.0
best_ssim = 0.0
# Load base low-resolution image.
fixed_lr = transforms.ToTensor()(Image.open(os.path.join("assets", "butterfly.png"))).unsqueeze(0)


def main(args):
    global best_psnr, best_ssim, fixed_lr

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for training.")

    if args.seed is not None:
        # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set
        # convolution algorithm.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.warning("You have chosen to seed training. "
                       "This will turn on the CUDNN deterministic setting, "
                       "which can slow down your training considerably! "
                       "You may see unexpected behavior when restarting "
                       "from checkpoints.")
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # If set `True`, ensure that every time the same input returns the same result. Default set `False`
        cudnn.deterministic = False

    if args.gpu is not None:
        logger.warning("You have chosen a specific GPU. This will completely disable data parallelism.")

    # create model
    generator = srgan(pretrained=args.pretrained)
    discriminator = discriminator_for_vgg()

    # Loss = pixel_loss + 0.006 * content loss + 0.001 * adversarial loss
    pixel_criterion = nn.MSELoss()
    content_criterion = ContentLoss()
    adversarial_criterion = nn.BCELoss()

    if args.gpu is not None:
        generator = generator.cuda(args.gpu)
        pixel_criterion = nn.MSELoss().cuda(args.gpu)
        content_criterion = ContentLoss().cuda(args.gpu)
        adversarial_criterion = nn.BCELoss().cuda(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        fixed_lr = fixed_lr.cuda(args.gpu)

    # All optimizer function and scheduler function.
    psnr_optimizer = optim.Adam(generator.parameters(), 0.0001, (0.9, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), 0.0001, (0.9, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), 0.0001, (0.9, 0.999))
    discriminator_scheduler = optim.lr_scheduler.StepLR(discriminator_optimizer, 64, 0.1)
    generator_scheduler = optim.lr_scheduler.StepLR(generator_optimizer, 64, 0.1)

    # Selection of appropriate treatment equipment.
    train_dataset = BaseDataset(os.path.join(args.data, "train"), 384, 4)
    test_dataset = BaseDataset(os.path.join(args.data, "test"), 384, 4)

    train_dataloader = data.DataLoader(train_dataset, 16, True, pin_memory=True)
    test_dataloader = data.DataLoader(test_dataset, 16, False, pin_memory=True)

    # Load pre training model.
    if args.netD != "":
        discriminator.load_state_dict(torch.load(args.netD))
    if args.netG != "":
        generator.load_state_dict(torch.load(args.netG))

    # Create a SummaryWriter at the beginning of training.
    psnr_writer = SummaryWriter(f"runs/psnr_logs")
    gan_writer = SummaryWriter(f"runs/gan_logs")

    for epoch in range(args.start_psnr_epoch, 512):
        # Train for one epoch for PSNR-oral.
        train_psnr(train_dataloader, generator, pixel_criterion, psnr_optimizer, epoch, psnr_writer, args)

        # Evaluate on test dataset.
        psnr, ssim = test(test_dataloader, generator, args.gpu)
        psnr_writer.add_scalar("PSNR_Test/PSNR", psnr, epoch)
        psnr_writer.add_scalar("PSNR_Test/SSIM", ssim, epoch)

        # Check whether the evaluation index of the current model is the highest.
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        # Save model weights for every epoch.
        torch.save(generator.state_dict(), os.path.join("weights", f"PSNR_epoch{epoch}.pth"))
        if is_best:
            torch.save(generator.state_dict(), os.path.join("weights", f"PSNR-best.pth"))

        # Save the last training model parameters.
    torch.save(generator.state_dict(), os.path.join("weights", f"PSNR-last.pth"))

    for epoch in range(args.start_gan_epoch, 128):
        # Train for one epoch for GAN-oral.
        train_gan(train_dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer,
                  pixel_criterion, content_criterion, adversarial_criterion, epoch, gan_writer, args)
        # Update GAN-oral optimizer learning rate.
        discriminator_scheduler.step()
        generator_scheduler.step()

        # Evaluate on test dataset.
        psnr, ssim = test(test_dataloader, generator, args.gpu)
        gan_writer.add_scalar("GAN_Test/PSNR", psnr, epoch)
        gan_writer.add_scalar("GAN_Test/SSIM", ssim, epoch)

        # Check whether the evaluation index of the current model is the highest.
        is_best = ssim > best_ssim
        best_ssim = max(ssim, best_ssim)
        # Save model weights for every epoch.
        torch.save(discriminator.state_dict(), os.path.join("weights", f"Discriminator_epoch{epoch}.pth"))
        torch.save(generator.state_dict(), os.path.join("weights", f"Generator_epoch{epoch}.pth"))
        if is_best:
            torch.save(generator.state_dict(), os.path.join("weights", f"GAN-best.pth"))

    # Save the last training model parameters.
    torch.save(generator.state_dict(), os.path.join("weights", f"GAN-last.pth"))


def train_psnr(dataloader, model, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter("Time", ":6.6f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(num_batches=len(dataloader),
                             meters=[batch_time, losses],
                             prefix=f"Epoch: [{epoch}]")

    # Switch to train mode.
    model.train()

    end = time.time()
    for i, (lr, hr) in enumerate(dataloader):
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)

        sr = model(lr)
        loss = criterion(sr, hr)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        losses.update(loss.item(), lr.size(0))

        # Add scalar data to summary.
        writer.add_scalar("PSNR_Train/Loss", loss.item(), i + epoch * len(dataloader) + 1)

        # Output results every 10 batches.
        if i % 10 == 0:
            progress.display(i)

    # Each one epoch create a sr image.
    with torch.no_grad():
        sr = model(fixed_lr)
        vutils.save_image(sr.detach(), os.path.join("runs", f"PSNR_epoch_{epoch}.png"), normalize=True)


def train_gan(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer,
              pixel_criterion, content_criterion, adversarial_criterion, epoch, writer, args):
    batch_time = AverageMeter("Time", ":6.4f")
    d_losses = AverageMeter("D Loss", ":6.6f")
    g_losses = AverageMeter("G Loss", ":6.6f")
    content_losses = AverageMeter("Content Loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial Loss", ":6.6f")
    progress = ProgressMeter(num_batches=len(dataloader),
                             meters=[batch_time, d_losses, g_losses, content_losses, adversarial_losses],
                             prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    discriminator.train()
    generator.train()

    end = time.time()
    for i, (lr, hr) in enumerate(dataloader):
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)
        batch_size = lr.size(0)

        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype).cuda(args.gpu, non_blocking=True)
        fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype).cuda(args.gpu, non_blocking=True)

        ##############################################
        # (1) Update D network: E(hr)[log(D(hr))] + E(sr)[log(1- D(G(sr))]
        ##############################################
        # Sets gradients of discriminator model parameters to zero.
        discriminator.zero_grad()

        # Generating fake high resolution images from real low resolution images.
        sr = generator(lr)

        # The discriminator marks the real sample as 1.
        d_loss_real = adversarial_criterion(discriminator(hr), real_label)
        d_loss_real.backward()
        # The discriminator marks the fake sample as 0.
        d_loss_fake = adversarial_criterion(discriminator(sr.detach()), fake_label)
        d_loss_fake.backward()

        # Count all discriminator losses.
        d_loss = d_loss_real + d_loss_fake

        # Update discriminator model parameters.
        discriminator_optimizer.step()

        ##############################################
        # (2) Update G network: pixel loss + 0.006 * content loss + 0.001 * adversarial loss
        ##############################################
        # Sets gradients of generator model parameters to zero.
        generator.zero_grad()
        
        # The calculate two image pixel value.
        pixel_loss = pixel_criterion(sr, hr.detach())
        # The 36th layer in VGG19 is used as the feature extractor by default.
        content_loss = content_criterion(sr, hr.detach())
        # The discriminator marks the fake sample as 1.
        adversarial_loss = adversarial_criterion(discriminator(sr), real_label)
        # Count all generator losses.
        g_loss = pixel_loss +  0.006 * content_loss + 0.001 * adversarial_loss
        g_loss.backward()

        # Update generator model parameters.
        generator_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        d_losses.update(d_loss.item(), lr.size(0))
        g_losses.update(g_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))

        iters = i + epoch * len(dataloader) + 1
        writer.add_scalar("GAN_Train/D_Loss", d_loss.item(), iters)
        writer.add_scalar("GAN_Train/G_Loss", g_loss.item(), iters)
        writer.add_scalar("GAN_Train/Content_Loss", content_loss.item(), iters)
        writer.add_scalar("GAN_Train/Adversarial_Loss", adversarial_loss.item(), iters)

        # Output results every 10 batches.
        if i % 10 == 0:
            progress.display(i)

    # Each one epoch create a sr image.
    with torch.no_grad():
        sr = generator(fixed_lr)
        vutils.save_image(sr.detach(), os.path.join("runs", f"GAN_epoch_{epoch}.png"), normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", metavar="DIR",
                        help="Path to dataset.")
    parser.add_argument("--start-psnr-epoch", default=0, type=int,
                        help="Manual psnr epoch number (useful on restarts). (Default: 0)")
    parser.add_argument("--start-gan-epoch", default=0, type=int,
                        help="Manual gan epoch number (useful on restarts). (Default: 0)")
    parser.add_argument("--netD", default="", type=str,
                        help="Path to Discriminator checkpoint.")
    parser.add_argument("--netG", default="", type=str,
                        help="Path to Generator checkpoint.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for initializing training.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    create_folder("runs")
    create_folder("weights")

    logger.info("TrainEngine:")
    logger.info("\tAPI version .......... 0.3.1")
    logger.info("\tBuild ................ 2021.07.06")

    main(args)

    logger.info("All training has been completed successfully.\n")
