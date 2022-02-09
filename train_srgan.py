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
"""File description: Initialize the SRResNet model."""
import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import ImageDataset
from model import Discriminator, Generator, ContentLoss


def main():
    print("Load train dataset and valid dataset...")
    train_dataloader, valid_dataloader = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    print("Build SRGAN model...")
    discriminator, generator = build_model()
    print("Build SRGAN model successfully.")

    print("Define all loss functions...")
    psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")

    print("Define all optimizer functions...")
    d_optimizer, g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    print("Define all optimizer scheduler functions...")
    d_scheduler, g_scheduler = define_optimizer(discriminator, generator)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether the training weight is restored...")
    resume_checkpoint(discriminator, generator)
    print("Check whether the training weight is restored successfully.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler.
    scaler = amp.GradScaler()

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    print("Start train SRGAN model.")
    for epoch in range(config.start_epoch, config.epochs):
        train(discriminator,
              generator,
              train_dataloader,
              psnr_criterion,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler,
              writer)

        psnr = validate(generator, valid_dataloader, psnr_criterion, epoch, writer)
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save(discriminator.state_dict(), os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth"))
        if is_best:
            torch.save(discriminator.state_dict(), os.path.join(results_dir, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(results_dir, f"g-best.pth"))

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

    # Save the generator weight under the last Epoch in this stage
    torch.save(discriminator.state_dict(), os.path.join(results_dir, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(results_dir, "g-last.pth"))
    print("End train SRGAN model.")


def load_dataset() -> [DataLoader, DataLoader]:
    """Load super-resolution data set

     Returns:
         training data set iterator, validation data set iterator

    """
    # Initialize the LMDB data set class and write the contents of the LMDB database file into memory
    train_datasets = ImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "train")
    valid_datasets = ImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "valid")
    # Make it into a data set type supported by PyTorch
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=False)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=False)

    return train_dataloader, valid_dataloader


def build_model() -> nn.Module:
    """Building discriminator and generators model

    Returns:
        SRGAN model

    """
    discriminator = Discriminator().to(config.device)
    generator = Generator().to(config.device)

    return discriminator, generator


def define_loss() -> [nn.MSELoss, nn.MSELoss, ContentLoss, nn.BCEWithLogitsLoss]:
    """Defines all loss functions

    Returns:
        PSNR loss, pixel loss, content loss, adversarial loss

    """
    psnr_criterion = nn.MSELoss().to(config.device)
    pixel_criterion = nn.MSELoss().to(config.device)
    content_criterion = ContentLoss().to(config.device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(config.device)

    return psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(discriminator: nn.Module, generator: nn.Module) -> [optim.Adam, optim.Adam]:
    """Define all optimizer functions

    Args:
        discriminator (nn.Module): Discriminator model
        generator (nn.Module): Generator model

    Returns:
        SRGAN optimizer

    """
    d_optimizer = optim.Adam(discriminator.parameters(), config.d_model_lr, config.d_model_betas)
    g_optimizer = optim.Adam(generator.parameters(), config.g_model_lr, config.g_model_betas)

    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> [lr_scheduler.StepLR, lr_scheduler.StepLR]:
    """Define learning rate scheduler

    Args:
        d_optimizer (optim.Adam): Discriminator optimizer
        g_optimizer (optim.Adam): Generator optimizer

    Returns:
        SRGAN model scheduler

    """
    d_scheduler = lr_scheduler.StepLR(d_optimizer, config.d_optimizer_step_size, config.d_optimizer_gamma)
    g_scheduler = lr_scheduler.StepLR(g_optimizer, config.g_optimizer_step_size, config.g_optimizer_gamma)

    return d_scheduler, g_scheduler


def resume_checkpoint(discriminator: nn.Module, generator: nn.Module) -> None:
    """Transfer training or recovery training

    Args:
        discriminator (nn.Module): Discriminator model
        generator (nn.Module): Generator model

    """
    if config.resume:
        if config.resume_d_weight != "":
            discriminator.load_state_dict(torch.load(config.resume_d_weight), strict=config.strict)
        if config.resume_g_weight != "":
            generator.load_state_dict(torch.load(config.resume_g_weight), strict=config.strict)


def train(discriminator,
          generator,
          train_dataloader,
          psnr_criterion,
          pixel_criterion,
          content_criterion,
          adversarial_criterion,
          d_optimizer,
          g_optimizer,
          epoch,
          scaler,
          writer) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_dataloader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses,
                              d_hr_probabilities, d_sr_probabilities,
                              psnres],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put all model in train mode.
    discriminator.train()
    generator.train()

    end = time.time()
    for index, (lr, hr) in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Send data to designated device
        lr = lr.to(config.device, non_blocking=True)
        hr = hr.to(config.device, non_blocking=True)

        # Set the real sample label to 1, and the false sample label to 0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)

        # Use generators to create super-resolution images
        sr = generator(lr)

        # Start training discriminator
        # At this stage, the discriminator needs to require a derivative gradient
        for p in discriminator.parameters():
            p.requires_grad = True

        # Initialize the discriminator optimizer gradient
        d_optimizer.zero_grad()

        # Calculate the loss of the discriminator on the high-resolution image
        with amp.autocast():
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_criterion(hr_output, real_label)
        # Gradient zoom
        scaler.scale(d_loss_hr).backward()

        # Calculate the loss of the discriminator on the super-resolution image.
        with amp.autocast():
            sr_output = discriminator(sr.detach())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
        # Gradient zoom
        scaler.scale(d_loss_sr).backward()
        # Update discriminator parameters
        scaler.step(d_optimizer)
        scaler.update()

        # Count discriminator total loss
        d_loss = d_loss_hr + d_loss_sr
        # End training discriminator

        # Start training generator
        # At this stage, the discriminator no needs to require a derivative gradient
        for p in discriminator.parameters():
            p.requires_grad = False

        # Initialize the generator optimizer gradient
        g_optimizer.zero_grad()

        # Calculate the loss of the generator on the super-resolution image
        with amp.autocast():
            output = discriminator(sr)
            pixel_loss = config.pixel_weight * pixel_criterion(sr, hr.detach())
            content_loss = config.content_weight * content_criterion(sr, hr.detach())
            adversarial_loss = config.adversarial_weight * adversarial_criterion(output, real_label)
        # Count discriminator total loss
        g_loss = pixel_loss + content_loss + adversarial_loss
        # Gradient zoom
        scaler.scale(g_loss).backward()
        # Update generator parameters
        scaler.step(g_optimizer)
        scaler.update()

        # End training generator

        # Calculate the scores of the two images on the discriminator
        d_hr_probability = torch.sigmoid(torch.mean(hr_output))
        d_sr_probability = torch.sigmoid(torch.mean(sr_output))

        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        iters = index + epoch * batches + 1
        writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
        writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
        writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
        writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
        writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
        writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
        writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
        if index % config.print_frequency == 0 and index != 0:
            progress.display(index)


def validate(model, valid_dataloader, psnr_criterion, epoch, writer) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(len(valid_dataloader), [batch_time, psnres], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (lr, hr) in enumerate(valid_dataloader):
            lr = lr.to(config.device, non_blocking=True)
            hr = hr.to(config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
            psnres.update(psnr.item(), hr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % config.print_frequency == 0:
                progress.display(index)

        writer.add_scalar("Valid/PSNR", psnres.avg, epoch + 1)
        # Print evaluation indicators.
        print(f"* PSNR: {psnres.avg:4.2f}.\n")

    return psnres.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
