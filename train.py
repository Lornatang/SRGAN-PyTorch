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
# ============================================================================

# ============================================================================
# File description: Realize the model training function.
# ============================================================================
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
from model import Generator, Discriminator, ContentLoss

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
# ============================================================================
"""文件说明: 实现模型训练功能."""
import os

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import ImageDataset
from model import Discriminator, Generator, ContentLoss
from basicsr.archs.rrdbnet_arch import RRDBNet


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
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=True)

    return train_dataloader, valid_dataloader


def build_model() -> [nn.Module, nn.Module]:
    """Building generators and discriminators

    Returns:
        Discriminator model, generator model

    """
    discriminator = Discriminator().to(config.device)
    generator = Generator().to(config.device)

    return discriminator, generator


def define_loss() -> [nn.MSELoss, ContentLoss, nn.BCEWithLogitsLoss]:
    """Defines all loss functions

    Returns:
        Pixel loss, content loss, adversarial loss

    """
    pixel_criterion = nn.MSELoss().to(config.device)
    content_criterion = ContentLoss().to(config.device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(config.device)

    return pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(discriminator, generator) -> [optim.Adam, optim.Adam, optim.Adam]:
    """Define all optimizer functions

    Args:
        discriminator (nn.Module): discriminator model
        generator (nn.Module): generator model

    Returns:
        SRResNet and SRGAN optimizer

    """
    srresnet_g_optimizer = optim.Adam(generator.parameters(), config.srresnet_g_lr, config.srresnet_g_betas)
    srgan_d_optimizer = optim.Adam(discriminator.parameters(), config.srgan_d_lr, config.srgan_d_betas)
    srgan_g_optimizer = optim.Adam(generator.parameters(), config.srgan_d_lr, config.srgan_d_betas)

    return srresnet_g_optimizer, srgan_d_optimizer, srgan_g_optimizer


def define_scheduler(srgan_d_optimizer, srgan_g_optimizer) -> [optim.lr_scheduler, optim.lr_scheduler]:
    """Define the learning rate adjustment method

    Args:
        srgan_d_optimizer (optim.Adam): Discriminator Optimizer in Adversarial Networks
        srgan_g_optimizer (optim.Adam): Generator Optimizer in Adversarial Networks

    """
    srgan_d_scheduler = lr_scheduler.MultiStepLR(srgan_d_optimizer, milestones=config.srgan_d_milestones, gamma=config.srgan_d_gamma)
    srgan_g_scheduler = lr_scheduler.MultiStepLR(srgan_g_optimizer, milestones=config.srgan_g_milestones, gamma=config.srgan_g_gamma)

    return srgan_d_scheduler, srgan_g_scheduler


def resume_checkpoint(discriminator, generator) -> None:
    if config.resume:
        if config.srresnet_g_resume_weight != "":
            generator.load_state_dict(torch.load(config.srresnet_g_resume_weight), strict=config.strict)
        else:
            discriminator.load_state_dict(torch.load(config.srgan_d_resume_weight), strict=config.strict)
            generator.load_state_dict(torch.load(config.srgan_g_resume_weight), strict=config.strict)


def train_pminet(generator,
                 train_dataloader,
                 pixel_criterion,
                 srresnet_g_optimizer,
                 epoch,
                 scaler,
                 pminet_writer) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_dataloader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in training mode
    model.train()

    # 计算Epoch下有多少迭代数
    batches = len(train_dataloader)
    # 使生成器处于训练模式
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        lr = lr.to(config.device, non_blocking=True)
        hr = hr.to(config.device, non_blocking=True)

        # 初始化生成器梯度
        generator.zero_grad()

        # 混合精度训练+梯度裁剪
        with amp.autocast():
            sr = generator(lr)
            pixel_loss = pixel_criterion(sr, hr)
        # 梯度缩放
        scaler.scale(pixel_loss).backward()
        # 梯度裁剪
        scaler.unscale_(pminet_g_optimizer)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        # 更新生成器权重
        scaler.step(pminet_g_optimizer)
        scaler.update()

        # 在本次Epoch中每一百次迭代和最后一次迭代打印一次损失函数同时写入Tensorboard
        if (index + 1) % 100 == 0 or (index + 1) == batches:
            iters = index + epoch * batches + 1
            pminet_writer.add_scalar("Train_PMINet/MAE_Loss", pixel_loss.item(), iters)
            print(f"Epoch[{epoch + 1:05d}/{config.pminet_epochs:05d}]({index + 1:05d}/{batches:05d}) MAE loss: {pixel_loss.item():.6f}.")


def train_pmigan(discriminator,
                 generator,
                 train_dataloader,
                 pixel_criterion,
                 content_criterion,
                 adversarial_criterion,
                 pmigan_d_optimizer,
                 pmigan_g_optimizer,
                 epoch,
                 scaler,
                 pmigan_writer) -> None:
    """训练PMIGAN网络

    Args:
        discriminator (nn.Module): 鉴别器模型
        generator (nn.Module): 生成器模型
        train_dataloader (DataLoader): 训练数据集的加载器
        pixel_criterion (nn.L1Loss): 像素损失函数
        content_criterion (ContentLoss): 内容损失函数
        adversarial_criterion (nn.BCEWithLogitsLoss): 对抗损失函数
        pmigan_d_optimizer (optim.Adam): 关于PMIGAN网络的鉴别器优化器参数
        pmigan_g_optimizer (optim.Adam): 关于PMIGAN网络的生成器优化器参数
        epoch (int): 当前训练周期数
        scaler (amp.GradScaler): 梯度缩放器
        pmigan_writer (SummaryWriter): TensorBoard 日志写入

    Returns:

    """
    # 计算Epoch下有多少迭代数
    batches = len(train_dataloader)
    # 使两个模型均处于训练模式
    discriminator.train()
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        lr = lr.to(config.device, non_blocking=True)
        hr = hr.to(config.device, non_blocking=True)

        # 将真实样本标签置为1，虚假样本标签置为0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)

        # 将鉴别器设置为训练模式，表示这个阶段需要更新鉴别器参数
        for p in discriminator.parameters():
            p.requires_grad = True

        # 生成超分辨图像.
        sr = generator(lr)

        # 开始训练鉴别器
        # 初始化鉴别器梯度
        discriminator.zero_grad()

        # 计算鉴别器在高分辨图像上的损失
        with amp.autocast():
            output = discriminator(hr)
            d_loss_hr = adversarial_criterion(output, real_label)
        # 梯度缩放
        scaler.scale(d_loss_hr).backward()
        # 鉴别器判断真实高分辨图像的得分
        d_hr = output.mean().sigmoid().item()

        # 计算鉴别器在超分辨图像上的损失.
        with amp.autocast():
            output = discriminator(sr.detach())
            d_loss_sr = adversarial_criterion(output, fake_label)
        # 梯度缩放
        scaler.scale(d_loss_sr).backward()
        # 鉴别器第一次判断超分辨图像的得分
        d_sr1 = output.mean().sigmoid().item()

        # 梯度裁剪
        scaler.unscale_(pmigan_d_optimizer)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        # 更新鉴别器权重
        scaler.step(pmigan_d_optimizer)
        scaler.update()

        # 鉴别器总损失
        d_loss = d_loss_hr + d_loss_sr
        # 结束训练鉴别器

        # 将鉴别器设置为验证模式，表示这个阶段需要不更新鉴别器参数
        for p in discriminator.parameters():
            p.requires_grad = False

        # 开始训练生成器
        # 初始化生成器梯度.
        generator.zero_grad()

        # 计算鉴别器在超分辨图像上的损失.
        with amp.autocast():
            output = discriminator(sr)
            pixel_loss = config.pixel_weight * pixel_criterion(sr, hr.detach())
            content_loss = config.content_weight * content_criterion(sr, hr.detach())
            adversarial_loss = config.adversarial_weight * adversarial_criterion(output, real_label)
            g_loss = pixel_loss + content_loss + adversarial_loss
        # 梯度缩放
        scaler.scale(g_loss).backward()
        # 梯度裁剪
        scaler.unscale_(pmigan_g_optimizer)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        # 鉴别器第二次判断超分辨图像的得分
        d_sr2 = output.mean().sigmoid().item()

        # 更新鉴别器权重.
        scaler.step(pmigan_g_optimizer)
        scaler.update()

        # 在本次Epoch中每一百次迭代和最后一次迭代打印一次损失函数同时写入Tensorboard
        if (index + 1) % 100 == 0 or (index + 1) == batches:
            iters = index + epoch * batches + 1

            pmigan_writer.add_scalar("Train_PMIGAN/D_Loss", d_loss.item(), iters)
            pmigan_writer.add_scalar("Train_PMIGAN/G_Loss", g_loss.item(), iters)
            pmigan_writer.add_scalar("Train_PMIGAN/Pixel_Loss", pixel_loss.item(), iters)
            pmigan_writer.add_scalar("Train_PMIGAN/Content_Loss", content_loss.item(), iters)
            pmigan_writer.add_scalar("Train_PMIGAN/Adversarial_Loss", adversarial_loss.item(), iters)
            pmigan_writer.add_scalar("Train_PMIGAN/D_HR", d_hr, iters)
            pmigan_writer.add_scalar("Train_PMIGAN/D_SR1", d_sr1, iters)
            pmigan_writer.add_scalar("Train_PMIGAN/D_SR2", d_sr2, iters)

            print(f"Epoch[{epoch + 1:05d}/{config.pmigan_epochs:05d}]({index + 1:05d}/{batches:05d}) "
                  f"D loss: {d_loss.item():.6f} "
                  f"G loss: {g_loss.item():.6f} "
                  f"pixel loss: {pixel_loss.item():.6f} "
                  f"content loss: {content_loss.item():.6f} "
                  f"adversarial loss: {adversarial_loss.item():.6f} "
                  f"D(HR): {d_hr:.6f} "
                  f"D(SR1)/D(SR2): {d_sr1:.6f}/{d_sr2:.6f}.")


def validate(generator,
             valid_dataloader,
             mse_criterion,
             epoch,
             stage,
             pminet_writer,
             pmigan_writer) -> float:
    """通过PSNR指标来验证模型训练程度

    Args:
        generator (nn.Module): 生成器模型
        valid_dataloader (torch.utils.data.DataLoader): 验证数据集的加载器
        mse_criterion (nn.MSELoss): MSE损失函数
        epoch (int): 验证周期数
        stage (str): 在哪个阶段下进行验证，一种是`pminet`, 另外一种是`pmigan`
        pminet_writer (SummaryWriter): PMINet的Tensorboard
        pmigan_writer (SummaryWriter): PMIGAN的Tensorboard

    Returns:
        PSNR value(float)
    """
    # 计算Epoch下有多少迭代数.
    batches = len(valid_dataloader)
    # 使生成器处于验证模式.
    generator.eval()
    # 初始化评价指标.
    total_psnr = 0.0

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            lr = lr.to(config.device, non_blocking=True)
            hr = hr.to(config.device, non_blocking=True)
            # 计算PSNR评价指标.
            sr = generator(lr)
            psnr = 10 * torch.log10(1 / mse_criterion(sr, hr)).item()
            total_psnr += psnr

        avg_psnr = total_psnr / batches
        # 将每轮的验证指标数值写入Tensorboard中.
        if stage == "pminet":
            pminet_writer.add_scalar("Valid_PMINet/PSNR", avg_psnr, epoch + 1)
        elif stage == "pmigan":
            pmigan_writer.add_scalar("Valid_PMIGAN/PSNR", avg_psnr, epoch + 1)
        # 打印评价指标.
        print(f"Valid stage: {stage}. Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr:.2f}.\n")

    return avg_psnr


def main() -> None:
    # 创建超分辨实验结果文件夹
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 创建训练过程日志文件
    pminet_writer = SummaryWriter(os.path.join("samples", "logs", f"pminet_{config.exp_name}"))
    pmigan_writer = SummaryWriter(os.path.join("samples", "logs", f"pmigan_{config.exp_name}"))

    # 初始化训练和验证数据集
    print("Load train dataset and valid dataset...")
    train_dataloader, valid_dataloader = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    # 初始化超分辨模型
    print("Build SR model...")
    discriminator, generator = build_model()
    print("Build SR model successfully.")

    # 初始化损失函数
    print("Define all loss functions...")
    mse_criterion, pixel_criterion, content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")

    # 初始化优化器
    print("Define all optimizer functions...")
    pminet_g_optimizer, pmigan_d_optimizer, pmigan_g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    # 检查是否恢复上一次异常结束的训练进度
    print("Check whether the training weight is restored...")
    resume_checkpoint(discriminator, generator)
    print("Check whether the training weight is restored successfully.")

    # 初始化梯度缩放器.
    scaler = amp.GradScaler()

    # 初始化训练生成网络评价指标.
    best_pminet_psnr = 0.0
    best_pmigan_psnr = 0.0

    print("Start train PMINet.")
    # 训练生成网络阶段.
    for epoch in range(config.pminet_start_epoch, config.pminet_epochs):
        train_pminet(generator,
                     train_dataloader,
                     pixel_criterion,
                     pminet_g_optimizer,
                     epoch,
                     scaler,
                     pminet_writer)

        psnr = validate(generator, valid_dataloader, mse_criterion, epoch, "pminet", pminet_writer, pmigan_writer)

        # 自动保存指标最高的那个模型
        is_best = psnr > best_pminet_psnr
        best_pminet_psnr = max(psnr, best_pminet_psnr)
        torch.save(generator.state_dict(), os.path.join(samples_dir, f"pminet_g_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(generator.state_dict(), os.path.join(results_dir, "pminet_g_best.pth"))

    # 保存本阶段最后一次Epoch下生成器权重.
    torch.save(generator.state_dict(), os.path.join(results_dir, "pminet_g_last.pth"))
    print("End train PMINet.")

    print("Start train PMIGAN.")
    # 训练对抗网络阶段.
    for epoch in range(config.pmigan_start_epoch, config.pmigan_epochs):
        train_pmigan(discriminator,
                     generator,
                     train_dataloader,
                     pixel_criterion,
                     content_criterion,
                     adversarial_criterion,
                     pmigan_d_optimizer,
                     pmigan_g_optimizer,
                     epoch,
                     scaler,
                     pmigan_writer)

        psnr = validate(generator, valid_dataloader, mse_criterion, epoch, "pmigan", pminet_writer, pmigan_writer)

        # 自动保存指标最高的那个模型
        is_best = psnr > best_pmigan_psnr
        best_pmigan_psnr = max(psnr, best_pmigan_psnr)
        torch.save(discriminator.state_dict(), os.path.join(samples_dir, f"pmigan_d_epoch{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(samples_dir, f"pmigan_g_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(discriminator.state_dict(), os.path.join(results_dir, "pmigan_d_best.pth"))
            torch.save(generator.state_dict(), os.path.join(results_dir, "pmigan_g_best.pth"))

    # 保存本阶段最后一次Epoch下对抗网络权重.
    torch.save(discriminator.state_dict(), os.path.join(results_dir, "pmigan_d_last.pth"))
    torch.save(generator.state_dict(), os.path.join(results_dir, "pmigan_g_last.pth"))
    print("End train PMIGAN.")


def main():
    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    srresnet_writer = SummaryWriter(os.path.join("samples", "logs", f"srresnet_{config.exp_name}"))
    srgan_writer = SummaryWriter(os.path.join("samples", "logs", f"srgan_{config.exp_name}"))

    print("Load train dataset and valid dataset...")
    train_dataloader, valid_dataloader = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    print("Build SR model...")
    discriminator, generator = build_model()
    print("Build SR model successfully.")

    print("Define all loss functions...")
    pixel_criterion, content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")

    print("Define all optimizer functions...")
    srresnet_g_optimizer, srgan_d_optimizer, srgan_g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    print("Define all scheduler functions...")
    srresnet_g_scheduler, srgan_d_scheduler, srgan_g_scheduler = define_scheduler(srresnet_g_optimizer, srgan_d_optimizer, srgan_g_optimizer)
    print("Define all optimizer scheduler successfully.")

    print("Check whether the training weight is restored...")
    resume_checkpoint(discriminator, generator)
    print("Check whether the training weight is restored successfully.")

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    print("Start train model.")
    for epoch in range(config.start_epoch, config.epochs):
        train(model, train_dataloader, criterion, optimizer, epoch, scaler, writer)

        psnr = validate(model, valid_dataloader, criterion, epoch, writer)
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save(model.state_dict(), os.path.join(samples_dir, f"epoch_{epoch + 1}.pth"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(results_dir, "best.pth"))

        # Update lr
        scheduler.step()

    # Save the generator weight under the last Epoch in this stage
    torch.save(model.state_dict(), os.path.join(results_dir, "last.pth"))
    print("End train model.")


def train_generator(train_dataloader, epoch) -> None:
    """Training the generator network.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The loader of the training dataset.
        epoch (int): number of training cycles.
    """
    # Calculate how many iterations there are under epoch.
    batches = len(train_dataloader)
    # Set generator network in training mode.
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # Copy the data to the specified device.
        lr = lr.to(device)
        hr = hr.to(device)
        # Initialize the gradient of the generator model.
        generator.zero_grad()
        # Generate super-resolution images.
        sr = generator(lr)
        # Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
        pixel_loss = pixel_criterion(sr, hr)
        # Update the weights of the generator model.
        pixel_loss.backward()
        p_optimizer.step()
        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Generator/Loss", pixel_loss.item(), iters)
        # Print the loss function every ten iterations and the last iteration in this epoch.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Train Epoch[{epoch + 1:04d}/{p_epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"Loss: {pixel_loss.item():.6f}.")


def train_adversarial(train_dataloader, epoch) -> None:
    """Training the adversarial network.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The loader of the training dataset.
        epoch (int): number of training cycles.
    """
    # Calculate how many iterations there are under Epoch.
    batches = len(train_dataloader)
    # Set adversarial network in training mode.
    discriminator.train()
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # Copy the data to the specified device.
        lr = lr.to(device)
        hr = hr.to(device)
        label_size = lr.size(0)
        # Create label. Set the real sample label to 1, and the false sample label to 0.
        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)

        # Initialize the gradient of the discriminator model.
        discriminator.zero_grad()
        # Calculate the loss of the discriminator model on the high-resolution image.
        output = discriminator(hr)
        d_loss_hr = adversarial_criterion(output, real_label)
        d_loss_hr.backward()
        d_hr = output.mean().item()
        # Generate super-resolution images.
        sr = generator(lr)
        # Calculate the loss of the discriminator model on the super-resolution image.
        output = discriminator(sr.detach())
        d_loss_sr = adversarial_criterion(output, fake_label)
        d_loss_sr.backward()
        d_sr1 = output.mean().item()
        # Update the weights of the discriminator model.
        d_loss = d_loss_hr + d_loss_sr
        d_optimizer.step()

        # Initialize the gradient of the generator model.
        generator.zero_grad()
        # Calculate the loss of the discriminator model on the super-resolution image.
        output = discriminator(sr)
        # Perceptual loss=0.01 * pixel loss + 1.0 * content loss + 0.001 * adversarial loss.
        pixel_loss = pixel_weight * pixel_criterion(sr, hr.detach())
        perceptual_loss = content_weight * content_criterion(sr, hr.detach())
        adversarial_loss = adversarial_weight * adversarial_criterion(output, real_label)
        # Update the weights of the generator model.
        g_loss = pixel_loss + perceptual_loss + adversarial_loss
        g_loss.backward()
        g_optimizer.step()
        d_sr2 = output.mean().item()

        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Adversarial/D_Loss", d_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/G_Loss", g_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/D_HR", d_hr, iters)
        writer.add_scalar("Train_Adversarial/D_SR1", d_sr1, iters)
        writer.add_scalar("Train_Adversarial/D_SR2", d_sr2, iters)
        # Print the loss function every ten iterations and the last iteration in this epoch.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Train stage: adversarial "
                  f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"Pixel Loss: {pixel_loss.item() * 100} "
                  f"Perceptual Loss: {perceptual_loss.item()} "
                  f"Adversarial Loss: {adversarial_loss.item() * 1000} "
                  f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.item():.6f} "
                  f"D(HR): {d_hr:.6f} D(SR1)/D(SR2): {d_sr1:.6f}/{d_sr2:.6f}.")


def validate(valid_dataloader, epoch, stage) -> float:
    """Verify the generator model.

    Args:
        valid_dataloader (torch.utils.data.DataLoader): loader for validating dataset.
        epoch (int): number of training cycles.
        stage (str): In which stage to verify, one is `generator`, the other is `adversarial`.

    Returns:
        PSNR value(float).
    """
    # Calculate how many iterations there are under epoch.
    batches = len(valid_dataloader)
    # Set generator model in verification mode.
    generator.eval()
    # Initialize the evaluation index.
    total_psnr_value = 0.0

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            # Copy the data to the specified device.
            lr = lr.to(device)
            hr = hr.to(device)
            # Generate super-resolution images.
            sr = generator(lr)
            # Calculate the PSNR indicator.
            mse_loss = psnr_criterion(sr, hr)
            psnr_value = 10 * torch.log10(1 / mse_loss).item()
            total_psnr_value += psnr_value

        avg_psnr_value = total_psnr_value / batches
        # Write the value of each round of verification indicators into Tensorboard.
        if stage == "generator":
            writer.add_scalar("Val_Generator/PSNR", avg_psnr_value, epoch + 1)
        elif stage == "adversarial":
            writer.add_scalar("Val_Adversarial/PSNR", avg_psnr_value, epoch + 1)
        # Print evaluation indicators.
        print(f"Valid stage: {stage} Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr_value:.2f}.\n")

    return avg_psnr_value


def main() -> None:
    # Create a super-resolution experiment result folder.
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)

    # Load the dataset.
    train_dataset = BaseDataset(train_dir, image_size, upscale_factor, "train")
    valid_dataset = BaseDataset(valid_dir, image_size, upscale_factor, "valid")
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)
    # Check whether the training progress of the last abnormal end is restored, for example, the power is
    # cut off in the middle of the training.
    if resume:
        print("Resuming...")
        if resume_p_weight != "":
            generator.load_state_dict(torch.load(resume_p_weight))
        else:
            discriminator.load_state_dict(torch.load(resume_d_weight))
            generator.load_state_dict(torch.load(resume_g_weight))

    # Initialize the evaluation indicators for the training stage of the generator model.
    best_psnr_value = 0.0
    # Train the generative network stage.
    for epoch in range(start_p_epoch, p_epochs):
        # Train each epoch for generator network.
        train_generator(train_dataloader, epoch)
        # Verify each epoch for generator network.
        psnr_value = validate(valid_dataloader, epoch, "generator")
        # Determine whether the performance of the generator network under epoch is the best.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the weight of the generator network under epoch. If the performance of the generator network under epoch
        # is best, save a file ending with `-best.pth` in the `results` directory.
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"p_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-best.pth"))

    # Save the weight of the last generator network under epoch in this stage.
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-last.pth"))

    # Initialize the evaluation index of the adversarial network training phase.
    best_psnr_value = 0.0
    # Load the model weights with the best indicators in the previous round of training.
    generator.load_state_dict(torch.load(os.path.join(exp_dir2, "p-best.pth")))
    # Training the adversarial network stage.
    for epoch in range(start_epoch, epochs):
        # Train each epoch for adversarial network.
        train_adversarial(train_dataloader, epoch)
        # Verify each epoch for adversarial network.
        psnr_value = validate(valid_dataloader, epoch, "adversarial")
        # Determine whether the performance of the adversarial network under epoch is the best.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the weight of the adversarial network under epoch. If the performance of the adversarial network
        # under epoch is the best, it will save two additional files ending with `-best.pth` in the `results` directory.
        torch.save(discriminator.state_dict(), os.path.join(exp_dir1, f"d_epoch{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"g_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best.pth"))
        # Adjust the learning rate of the adversarial model.
        d_scheduler.step()
        g_scheduler.step()

    # Save the weight of the adversarial network under the last epoch in this stage.
    torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-last.pth"))


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
