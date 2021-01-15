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
import csv
import logging
import math
import os

import lpips
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

import srgan_pytorch.models as models
from srgan_pytorch.dataset import BaseTestDataset
from srgan_pytorch.dataset import BaseTrainDataset
from srgan_pytorch.loss import VGGLoss
from srgan_pytorch.models.discriminator import discriminator
from srgan_pytorch.utils.common import init_torch_seeds
from srgan_pytorch.utils.common import save_checkpoint
from srgan_pytorch.utils.common import weights_init
from srgan_pytorch.utils.device import select_device
from srgan_pytorch.utils.estimate import test_lpips
from srgan_pytorch.utils.estimate import test_psnr

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def train_psnr(epoch: int,
               total_epoch: int,
               total_iters: int,
               dataloader: torch.utils.data.DataLoader,
               model: nn.Module,
               content_criterion: nn.MSELoss,
               optimizer: torch.optim.Adam,
               device: torch.device):
    # switch train mode.
    model.train()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # Move data to special device.
        lr = data[0].to(device)
        hr = data[1].to(device)

        # Generating fake high resolution images from real low resolution images.
        sr = model(lr)
        # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
        loss = content_criterion(sr, hr)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_description(f"[{epoch + 1}/{total_epoch}]"
                                     f"[{i + 1}/{len(dataloader)}] "
                                     f"Loss: {loss.item():.6f}")

        iters = i + epoch * len(dataloader) + 1
        # The image is saved every 1000 epoch.
        if iters % 1000 == 0:
            vutils.save_image(hr, os.path.join("output", "hr", f"ResNet_{iters}.bmp"))
            hr = model(lr)
            vutils.save_image(hr.detach(), os.path.join("output", "sr", f"ResNet_{iters}.bmp"))

        if iters == int(total_iters):  # If the iteration is reached, exit.
            break


def train_gan(epoch: int,
              total_epoch: int,
              total_iters: int,
              dataloader: torch.utils.data.DataLoader,
              discriminator: nn.Module,
              generator: nn.Module,
              perceptual_criterion: VGGLoss,
              adversarial_criterion: nn.BCELoss,
              discriminator_optimizer: torch.optim.Adam,
              generator_optimizer: torch.optim.Adam,
              discriminator_scheduler: torch.optim.lr_scheduler,
              generator_scheduler: torch.optim.lr_scheduler,
              device: torch.device):
    # switch train mode.
    generator.train()
    discriminator.train()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        lr = data[0].to(device)
        hr = data[1].to(device)
        batch_size = lr.size(0)

        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype, device=device)
        fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype, device=device)

        ##############################################
        # (1) Update D network: maximize - E(hr)[1- log(D(hr, sr))] - E(sr)[log(D(sr, hr))]
        ##############################################
        # Set discriminator gradients to zero.
        discriminator.zero_grad()

        # Train with real high resolution image.
        output = discriminator(hr)  # Train lr image.
        errD_real = adversarial_criterion(output, real_label)
        D_x = output.mean().item()
        errD_real.backward()

        # Generating fake high resolution images from real low resolution images.
        sr = generator(lr)

        # Train with fake image resolution image.
        output = discriminator(sr.detach())  # No train sr image.
        errD_fake = adversarial_criterion(output, fake_label)
        D_G_z1 = output.mean().item()
        errD_fake.backward()

        errD = errD_real + errD_fake
        discriminator_optimizer.step()

        ##############################################
        # (2) Update G network: -logD[G(LR)]
        ##############################################
        # Set generator gradients to zero
        generator.zero_grad()

        # According to the feature map, the root mean square error is regarded as the content loss.
        perceptual_loss = perceptual_criterion(sr, hr)
        # Train with fake high resolution image.
        output = discriminator(sr)  # Train fake image.
        D_G_z2 = output.mean().item()
        # Adversarial loss.
        adversarial_loss = adversarial_criterion(output, real_label)
        errG = perceptual_loss + 0.001 * adversarial_loss
        errG.backward()
        generator_optimizer.step()

        progress_bar.set_description(f"[{epoch + 1}/{total_epoch}][{i + 1}/{len(dataloader)}] "
                                     f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                     f"D(HR): {D_x:.6f} D(G(LR)): {D_G_z1:.6f}/{D_G_z2:.6f}")

        iters = i + epoch * len(dataloader) + 1
        # The image is saved every 1000 epoch.
        if iters % 1000 == 0:
            vutils.save_image(hr, os.path.join("output", "hr", f"GAN_{iters}.bmp"))
            hr = generator(lr)
            vutils.save_image(hr.detach(), os.path.join("output", "sr", f"GAN_{iters}.bmp"))

        if iters == int(total_iters):  # If the iteration is reached, exit.
            break

    # Dynamic adjustment of learning rate
    discriminator_scheduler.step()
    generator_scheduler.step()


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)

        logger.info("Load training dataset")
        # Selection of appropriate treatment equipment.
        train_dataset = BaseTrainDataset(root=f"{args.data}/train",
                                         image_size=args.image_size,
                                         upscale_factor=args.upscale_factor)
        test_dataset = BaseTestDataset(root=f"{args.data}/test",
                                       image_size=args.image_size,
                                       upscale_factor=args.upscale_factor)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=args.batch_size,
                                                            pin_memory=True,
                                                            num_workers=int(args.workers))
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=args.batch_size,
                                                           pin_memory=True,
                                                           num_workers=int(args.workers))

        logger.info(f"Train Dataset information:\n"
                    f"\tTrain Dataset dir is `{os.getcwd()}/{args.data}/train`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")
        logger.info(f"Test Dataset information:\n"
                    f"\tTest Dataset dir is `{os.getcwd()}/{args.data}/test`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")

        # Construct network architecture model of generator and discriminator.
        self.device = select_device(args.device, batch_size=1)
        if args.pretrained:
            logger.info(f"Using pre-trained model `{args.arch}`")
            self.generator = models.__dict__[args.arch](pretrained=True).to(self.device)
        else:
            logger.info(f"Creating model `{args.arch}`")
            self.generator = models.__dict__[args.arch]().to(self.device)
        logger.info(f"Creating discriminator model")
        self.discriminator = discriminator().to(self.device)

        self.generator = self.generator.apply(weights_init)
        self.discriminator = self.discriminator.apply(weights_init)

        # Parameters of pre training model.
        self.start_psnr_epoch = math.floor(args.start_psnr_iter / len(self.train_dataloader))
        self.psnr_epochs = math.ceil(args.psnr_iters / len(self.train_dataloader))
        self.psnr_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.9, 0.999))

        logger.info(f"Pre-training model training parameters:\n"
                    f"\tIters is {args.psnr_iters}\n"
                    f"\tEpoch is {self.psnr_epochs}\n"
                    f"\tOptimizer Adam\n"
                    f"\tLearning rate {args.lr}\n"
                    f"\tBetas (0.9, 0.999)")

        # Parameters of GAN training model.
        self.start_epoch = math.floor(args.start_iter / len(self.train_dataloader))
        self.epochs = math.ceil(args.iters / len(self.train_dataloader))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(self.discriminator_optimizer,
                                                                       step_size=self.epochs // 2,
                                                                       gamma=0.1)
        self.generator_scheduler = torch.optim.lr_scheduler.StepLR(self.generator_optimizer,
                                                                   step_size=self.epochs // 2,
                                                                   gamma=0.1)
        logger.info(f"All model training parameters:\n"
                    f"\tIters is {args.iters}\n"
                    f"\tEpoch is {self.epochs}\n"
                    f"\tOptimizer is Adam\n"
                    f"\tLearning rate is {args.lr}\n"
                    f"\tBetas is (0.9, 0.999)\n"
                    f"\tScheduler is StepLR")

        # We use VGG5.4 as our feature extraction method by default.
        self.perceptual_criterion = VGGLoss().to(self.device)
        # Loss = perceptual loss + 0.001 * adversarial loss
        self.content_criterion = nn.MSELoss().to(self.device)
        self.adversarial_criterion = nn.BCELoss().to(self.device)
        # LPIPS Evaluating.
        self.lpips_criterion = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        logger.info(f"Loss function:\n"
                    f"\tPerceptual loss is VGGLoss\n"
                    f"\tContent loss is MSELoss\n"
                    f"\tAdversarial loss is BCELoss")

    def run(self):
        args = self.args
        best_psnr = 0.
        best_lpips = 1.

        # Loading PSNR pre training model.
        if args.netP != "":
            checkpoint = torch.load(args.netP)
            self.args.start_psnr_iter = checkpoint["iter"]
            best_psnr = checkpoint["best_psnr"]
            self.generator.load_state_dict(checkpoint["state_dict"])

        # Start train PSNR model.
        logger.info("Staring training PSNR model")
        logger.info(f"Training for {args.psnr_iters} iters")

        # Writer train PSNR model log.
        if self.args.start_psnr_iter == 0:
            with open(f"ResNet_{args.arch}.csv", "w+") as f:
                writer = csv.writer(f)
                writer.writerow(["Iter", "PSNR"])

        for psnr_epoch in range(self.start_psnr_epoch, self.psnr_epochs):
            # Train epoch.
            train_psnr(epoch=psnr_epoch,
                       total_epoch=self.psnr_epochs,
                       total_iters=args.psnr_iters,
                       dataloader=self.train_dataloader,
                       model=self.generator,
                       content_criterion=self.content_criterion,
                       optimizer=self.psnr_optimizer,
                       device=self.device)

            # every 10 epoch test.
            if (psnr_epoch + 1) % 10 == 0:
                # Test for every epoch.
                psnr = test_psnr(self.generator, self.content_criterion, self.test_dataloader, self.device)
                iters = (psnr_epoch + 1) * len(self.train_dataloader)

                # remember best psnr and save checkpoint
                is_best = psnr > best_psnr
                best_psnr = max(psnr, best_psnr)

                # The model is saved every 1 epoch.
                save_checkpoint(
                    {"iter": iters,
                     "state_dict": self.generator.state_dict(),
                     "best_psnr": best_psnr,
                     "optimizer": self.psnr_optimizer.state_dict()
                     }, is_best,
                    os.path.join("weights", f"ResNet_{args.arch}_iter_{iters}.pth"),
                    os.path.join("weights", f"ResNet_{args.arch}.pth"))

                # Writer training log
                with open(f"ResNet_{args.arch}.csv", "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow([iters, psnr])

        # Load best generator model weight.
        self.generator.load_state_dict(torch.load(os.path.join("weights", f"ResNet_{args.arch}.pth"), self.device))

        # Loading SRGAN training model.
        if args.netG != "":
            checkpoint = torch.load(args.netG)
            self.args.start_psnr_iter = checkpoint["iter"]
            best_lpips = checkpoint["best_lpips"]
            self.generator.load_state_dict(checkpoint["state_dict"])

        # Writer train GAN model log.
        if args.start_iter == 0:
            with open(f"GAN_{args.arch}.csv", "w+") as f:
                writer = csv.writer(f)
                writer.writerow(["Iter", "LPIPS"])

        for epoch in range(self.start_epoch, self.epochs):
            # Train epoch.
            train_gan(epoch=epoch,
                      total_epoch=self.epochs,
                      total_iters=args.iters,
                      dataloader=self.train_dataloader,
                      discriminator=self.discriminator,
                      generator=self.generator,
                      perceptual_criterion=self.perceptual_criterion,
                      adversarial_criterion=self.adversarial_criterion,
                      discriminator_optimizer=self.discriminator_optimizer,
                      generator_optimizer=self.generator_optimizer,
                      discriminator_scheduler=self.discriminator_scheduler,
                      generator_scheduler=self.generator_scheduler,
                      device=self.device)
            # Test for every epoch.
            lpips = test_lpips(self.generator, self.lpips_criterion, self.test_dataloader, self.device)
            iters = (epoch + 1) * len(self.train_dataloader)

            # remember best psnr and save checkpoint
            is_best = lpips < best_lpips
            best_lpips = min(lpips, best_lpips)

            # The model is saved every 1 epoch.
            save_checkpoint(
                {"iter": iters,
                 "state_dict": self.generator.state_dict(),
                 "best_lpips": best_lpips,
                 "optimizer": self.generator_optimizer.state_dict()
                 }, is_best,
                os.path.join("weights", f"GAN_{args.arch}_iter_{iters}.pth"),
                os.path.join("weights", f"GAN_{args.arch}.pth"))

            # Writer training log
            with open(f"GAN_{args.arch}.csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerow([iters, lpips])
