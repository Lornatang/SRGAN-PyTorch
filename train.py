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
import random
from argparse import ArgumentParser
from os.path import join

import torch
from torch.backends import cudnn
from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from srgan_pytorch.dataset import BaseDataset
from srgan_pytorch.loss import ContentLoss
from srgan_pytorch.model import discriminator
from srgan_pytorch.model import generator
from srgan_pytorch.utils import create_folder
from test import iqa
from test import sr

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--dataroot", default="data/DIV2K/train",
                    help="Path to dataset.")
parser.add_argument("--p-epochs", default=512, type=int,
                    help="Number of total p-oral epochs to run. (Default: 512)")
parser.add_argument("--g-epochs", default=128, type=int,
                    help="Number of total g-oral epochs to run. (Default: 128)")
parser.add_argument("--batch-size", default=16, type=int,
                    help="The batch size of the dataset. (Default: 16)")
parser.add_argument("--p-lr", default=0.0001, type=float,
                    help="Learning rate for psnr-oral. (Default: 0.0001)")
parser.add_argument("--g-lr", default=0.0001, type=float,
                    help="Learning rate for gan-oral. (Default: 0.0001)")
parser.add_argument("--image-size", default=96, type=int,
                    help="Image size of high resolution image. (Default: 96)")
parser.add_argument("--scale", default=4, type=int, choices=[4],
                    help="Low to high resolution scaling factor. "
                         "Optional: [4]. (Default: 4)")
parser.add_argument("--netD", default="", type=str,
                    help="Path to Discriminator checkpoint.")
parser.add_argument("--netG", default="", type=str,
                    help="Path to Generator checkpoint.")
parser.add_argument("--seed", default=None, type=int,
                    help="Seed for initializing training.")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--cuda", dest="cuda", action="store_true",
                    help="Enables cuda.")
args = parser.parse_args()

# Random seed can ensure that the results of each training are inconsistent.
if args.seed is None:
    args.seed = random.randint(1, 10000)
logger.info(f"Random Seed: {args.seed}")
random.seed(args.seed)
torch.manual_seed(args.seed)

# Because the resolution of each input image is fixed, setting it to `True`
# will make CUDNN automatically find the optimal convolution method.
# If the input image resolution is not fixed, it needs to be set to `False`.
cudnn.benchmark = True

# Set whether to use CUDA.
if torch.cuda.is_available() and not args.cuda:
    logger.warning("You have a CUDA device, so you should probably "
                   "run with --cuda")
device = torch.device("cuda:0" if args.cuda else "cpu")

# Load dataset.
dataset = BaseDataset(dataroot=args.dataroot,
                      image_size=args.image_size,
                      scale=args.scale)
dataloader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        pin_memory=True)

# Load model.
netD = discriminator().to(device)
netG = generator(args.pretrained).to(device)

# Optional: Resume training.
start_p_epoch = 0
start_g_epoch = 0
if args.netD != "" and args.netG != "":
    netD.load_state_dict(torch.load(args.netD))
    start_g_epoch = "".join(list(filter(str.isdigit, args.netD)))
    logger.info(f"You loaded {args.netD} for discriminator."
                f"G-Oral resume epoch from {start_g_epoch}.")
if args.netG != "" and args.netD == "":
    netG.load_state_dict(torch.load(args.netG))
    start_p_epoch = "".join(list(filter(str.isdigit, args.netG)))
    logger.info(f"You loaded {args.netG} for generator."
                f"P-Oral resume epoch from {start_p_epoch}.")

# Define loss function.
pixel_criterion = MSELoss().to(device)
content_criterion = ContentLoss().to(device)
adv_criterion = BCELoss().to(device)

# Define optimizer function.
p_optim = Adam(netG.parameters(), args.p_lr, (0.9, 0.999))
d_optim = Adam(netD.parameters(), args.g_lr, (0.9, 0.999))
g_optim = Adam(netG.parameters(), args.g_lr, (0.9, 0.999))

# Define scheduler function.
d_scheduler = StepLR(d_optim, args.g_epochs // 2, 0.1)
g_scheduler = StepLR(g_optim, args.g_epochs // 2, 0.1)

# Visualization. Use Tensorboard to record the Loss curve during training.
p_writer = SummaryWriter("samples/psnr_logs")
g_writer = SummaryWriter("samples/gan_logs")


def main():
    # Use PSNR value as the image evaluation index in the process of training PSNR.
    # Use SSIM value as the image evaluation index in the process of training GAN.
    # If an Epoch is higher than the current index, save the model weight under
    # the current Epoch as `XXX-best.pth` and save it to the `weights` folder.
    best_psnr = 0.0
    best_ssim = 0.0

    # Train the PSNR stage of the generative model, and save the model weight
    # after reaching a certain index.
    for epoch in range(int(start_p_epoch), args.p_epochs):
        # Training.
        train_psnr(epoch)
        # Test.
        sr(netG, join("assets", "lr.png"), join("assets", "sr.png"))
        psnr, ssim = iqa(join("assets", "sr.png"), join("assets", "hr.png"))
        logger.info(f"P-Oral epoch {epoch} PSNR: {psnr:.2f}dB SSIM: {ssim:.4f}.")
        # Write result to TensorBoard.
        p_writer.add_scalar("P_Test/PSNR", psnr, epoch)
        p_writer.add_scalar("P_Test/SSIM", ssim, epoch)

        # Check whether the PSNR value of the current epoch is the highest value
        # ever in the training PSNR phase.
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        # Save the model once after each epoch. If the current PSNR value is the
        # highest, save another model ending with `best`.
        torch.save(netG.state_dict(), join("weights", f"P_epoch{epoch}.pth"))
        if is_best:
            torch.save(netG.state_dict(), join("weights", "P-best.pth"))

    # Save the model weights of the last iteration of the PSNR stage.
    torch.save(netG.state_dict(), join("weights", "P-last.pth"))
    
    # Load the model weights with the best results from the previous training.
    netG.load_state_dict(torch.load(join("weights", "P-best.pth")))

    # Train the generative model in the GAN stage and save the model weight after
    # reaching a certain index.
    for epoch in range(int(start_g_epoch), args.g_epochs):
        # Training.
        train_gan(epoch)
        # Test.
        sr(netG, join("assets", "lr.png"), join("assets", "sr.png"))
        psnr, ssim = iqa(join("assets", "sr.png"), join("assets", "hr.png"))
        logger.info(f"G-Oral epoch {epoch} PSNR: {psnr:.2f}dB SSIM: {ssim:.4f}.")
        # Write result to TensorBoard.
        p_writer.add_scalar("G_Test/PSNR", psnr, epoch)
        p_writer.add_scalar("G_Test/SSIM", ssim, epoch)

        # Check whether the PSNR value of the current epoch is the highest value
        # in the history of the training GAN stage.
        is_best = ssim > best_ssim
        best_ssim = max(ssim, best_ssim)
        # Save the model once after each epoch, if the current PSNR value is the
        # highest, save another model ending with `best`.
        torch.save(netD.state_dict(), join("weights", f"D_epoch{epoch}.pth"))
        torch.save(netG.state_dict(), join("weights", f"G_epoch{epoch}.pth"))
        if is_best:
            torch.save(netD.state_dict(), join("weights", "D-best.pth"))
            torch.save(netG.state_dict(), join("weights", "G-best.pth"))

        # Call the scheduler function to adjust the learning rate of the
        # generator model and the discrimination model.
        d_scheduler.step()
        g_scheduler.step()

    # Save the model weights of the last iteration of the GAN stage.
    torch.save(netG.state_dict(), join("weights", "G-last.pth"))


def train_psnr(epoch):
    num_batches = len(dataloader)
    for index, data in enumerate(dataloader, 1):
        # 将数据拷贝至指定设备当中.
        inputs, target = data[0].to(device), data[1].to(device)

        ##############################################
        # (0) Update G network: min MSE(output, target)
        ##############################################
        netG.zero_grad()
        output = netG(inputs)
        loss = pixel_criterion(output, target)
        loss.backward()
        p_optim.step()

        logger.info(f"Epoch[{epoch}/{args.p_epochs}]"
                    f"({index}/{num_batches}) P Loss: {loss.item():.4f}.")

        # Write the loss value during PSNR training into Tensorboard.
        batches = index + epoch * num_batches + 1
        p_writer.add_scalar("Train/P_Loss", loss.item(), batches)


def train_gan(epoch):
    num_batches = len(dataloader)
    for index, data in enumerate(dataloader, 1):
        # Copy the data to the designated device.
        inputs, target = data[0].to(device), data[1].to(device)
        batch_size = inputs.size(0)

        # Set the real sample label to 1, and the false sample label to 0.
        real_label = torch.full((batch_size, 1), 1, dtype=inputs.dtype).to(
            device)
        fake_label = torch.full((batch_size, 1), 0, dtype=inputs.dtype).to(
            device)

        ##############################################
        # (1) Update D network: E(real)[log(D(real))] + E(fake)[log(1 - D(G(fake))]
        ##############################################
        netD.zero_grad()
        fake = netG(inputs)
        d_loss_real = adv_criterion(netD(target), real_label)
        d_loss_fake = adv_criterion(netD(fake.detach()), fake_label)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optim.step()

        ##############################################
        # (2) Update G network: E(fake)[log(1 - D(G(fake))]
        ##############################################
        netG.zero_grad()
        fake = netG(inputs)
        pixel_loss = 1e+1 * pixel_criterion(fake, target.detach())
        content_loss = 2e-6 * content_criterion(fake, target.detach())
        adv_loss = 1e-3 * adv_criterion(netD(fake), real_label)
        g_loss = pixel_loss + content_loss + adv_loss
        g_loss.backward()
        g_optim.step()

        logger.info(f"Epoch[{epoch}/{args.g_epochs}]"
                    f"({index}/{num_batches}) "
                    f"D Loss: {d_loss.item():.4f} "
                    f"G Loss: {g_loss.item():.4f}.")

        # Write the loss value during GAN training into Tensorboard.
        batches = index + epoch * num_batches + 1
        g_writer.add_scalar("Train/D_Loss", d_loss.item(), batches)
        g_writer.add_scalar("Train/G_Loss", g_loss.item(), batches)


if __name__ == "__main__":
    create_folder("weights")
    create_folder("samples")

    logger.info("TrainEngine:")
    logger.info("\tAPI version .......... 0.4.0")
    logger.info("\tBuild ................ 2021.07.09")

    main()

    logger.info("All training has been completed successfully.\n")
