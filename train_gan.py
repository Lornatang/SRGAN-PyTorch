# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
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
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
from dataset import CUDAPrefetcher, BaseImageDataset, PairedImageDataset
from imgproc import random_crop_torch, random_rotate_torch, random_vertically_flip_torch, random_horizontally_flip_torch
from test import test
from utils import build_iqa_model, load_resume_state_dict, load_pretrained_state_dict, make_directory, save_checkpoint, \
    Summary, AverageMeter, ProgressMeter


def main():
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/train/SRGAN_x4-SRGAN_ImageNet-Set5.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    # Fixed random number seed
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Default to start training from scratch
    start_epoch = 0

    # Initialize the image clarity evaluation index
    best_psnr = 0.0
    best_ssim = 0.0

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    # Define the basic functions needed to start training
    train_data_prefetcher, paired_test_data_prefetcher = load_dataset(config, device)
    g_model, ema_g_model, d_model = build_model(config, device)
    pixel_criterion, content_criterion, adversarial_criterion = define_loss(config, device)
    g_optimizer, d_optimizer = define_optimizer(g_model, d_model, config)
    g_scheduler, d_scheduler = define_scheduler(g_optimizer, d_optimizer, config)

    # Load the pretrained model
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"]:
        g_model = load_pretrained_state_dict(g_model,
                                             config["MODEL"]["G"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_G_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained g model weights not found.")
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_MODEL"]:
        d_model = load_pretrained_state_dict(d_model,
                                             config["MODEL"]["D"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_D_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained dd model weights not found.")

    # Load the last training interruption model node
    if config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"]:
        g_model, ema_g_model, start_epoch, best_psnr, best_ssim, g_optimizer, g_scheduler = load_resume_state_dict(
            g_model,
            ema_g_model,
            g_optimizer,
            g_scheduler,
            config["MODEL"]["G"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"],
        )
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['RESUMED_G_MODEL']}` resume model weights successfully.")
    else:
        print("Resume training g model not found. Start training from scratch.")
    if config["TRAIN"]["CHECKPOINT"]["RESUMED_D_MODEL"]:
        d_model, _, start_epoch, best_psnr, best_ssim, d_optimizer, d_scheduler = load_resume_state_dict(
            d_model,
            None,
            d_optimizer,
            d_scheduler,
            config["MODEL"]["D"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_D_MODEL"],
        )
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['RESUMED_D_MODEL']}` resume model weights successfully.")
    else:
        print("Resume training d model not found. Start training from scratch.")

    # Initialize the image clarity evaluation method
    psnr_model, ssim_model = build_iqa_model(
        config["SCALE"],
        config["TEST"]["ONLY_TEST_Y_CHANNEL"],
        device,
    )

    # Create the folder where the model weights are saved
    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", config["EXP_NAME"]))

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(g_model,
              ema_g_model,
              d_model,
              train_data_prefetcher,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              g_optimizer,
              d_optimizer,
              epoch,
              scaler,
              writer,
              device,
              config)

        # Update LR
        g_scheduler.step()
        d_scheduler.step()

        psnr, ssim = test(g_model,
                          paired_test_data_prefetcher,
                          psnr_model,
                          ssim_model,
                          device,
                          config)
        print("\n")

        # Write the evaluation indicators of each round of Epoch to the log
        writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)

        # Automatically save model weights
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "psnr": psnr,
                         "ssim": ssim,
                         "state_dict": g_model.state_dict(),
                         "ema_state_dict": ema_g_model.state_dict() if ema_g_model is not None else None,
                         "optimizer": g_optimizer.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "psnr": psnr,
                         "ssim": ssim,
                         "state_dict": d_model.state_dict(),
                         "optimizer": d_optimizer.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_best.pth.tar",
                        "d_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset(
        config: Any,
        device: torch.device,
) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load the train dataset
    degenerated_train_datasets = BaseImageDataset(
        config["TRAIN"]["DATASET"]["TRAIN_GT_IMAGES_DIR"],
        None,
        config["SCALE"],
    )

    # Load the registration test dataset
    paired_test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                              config["TEST"]["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"])

    # generate dataset iterator
    degenerated_train_dataloader = DataLoader(degenerated_train_datasets,
                                              batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                              shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                              num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                              pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                              drop_last=True,
                                              persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])
    paired_test_dataloader = DataLoader(paired_test_datasets,
                                        batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                        shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                        num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                        pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                        drop_last=False,
                                        persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])

    # Replace the data set iterator with CUDA to speed up
    train_data_prefetcher = CUDAPrefetcher(degenerated_train_dataloader, device)
    paired_test_data_prefetcher = CUDAPrefetcher(paired_test_dataloader, device)

    return train_data_prefetcher, paired_test_data_prefetcher


def build_model(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module or Any, nn.Module]:
    g_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           num_rcb=config["MODEL"]["G"]["NUM_RCB"])
    d_model = model.__dict__[config["MODEL"]["D"]["NAME"]](in_channels=config["MODEL"]["D"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["D"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["D"]["CHANNELS"])

    g_model = g_model.to(device)
    d_model = d_model.to(device)

    if config["MODEL"]["EMA"]["ENABLE"]:
        # Generate an exponential average model based on a generator to stabilize model training
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_g_model = AveragedModel(g_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_g_model = None

    # compile model
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model)
    if config["MODEL"]["D"]["COMPILED"]:
        d_model = torch.compile(d_model)
    if config["MODEL"]["EMA"]["COMPILED"] and ema_g_model is not None:
        ema_g_model = torch.compile(ema_g_model)

    return g_model, ema_g_model, d_model


def define_loss(config: Any, device: torch.device) -> [nn.MSELoss, model.ContentLoss, nn.BCEWithLogitsLoss]:
    if config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["NAME"] == "MSELoss":
        pixel_criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['PIXEL_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NAME"] == "ContentLoss":
        content_criterion = model.ContentLoss(
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NET_CFG_NAME"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["BATCH_NORM"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NUM_CLASSES"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["MODEL_WEIGHTS_PATH"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NODES"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NORMALIZE_MEAN"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NORMALIZE_STD"],
        )
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['CONTENT_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["NAME"] == "vanilla":
        adversarial_criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['ADVERSARIAL_LOSS']['NAME']} is not implemented.")

    pixel_criterion = pixel_criterion.to(device)
    content_criterion = content_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)

    return pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(g_model: nn.Module, d_model: nn.Module, config: Any) -> [optim.Adam, optim.Adam]:
    if config["TRAIN"]["OPTIM"]["NAME"] == "Adam":
        g_optimizer = optim.Adam(g_model.parameters(),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
        d_optimizer = optim.Adam(d_model.parameters(),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])

    else:
        raise NotImplementedError(f"Optimizer {config['TRAIN']['OPTIM']['NAME']} is not implemented.")

    return g_optimizer, d_optimizer


def define_scheduler(g_optimizer: optim.Adam, d_optimizer: optim.Adam, config: Any) -> [lr_scheduler.MultiStepLR, lr_scheduler.MultiStepLR]:
    if config["TRAIN"]["LR_SCHEDULER"]["NAME"] == "MultiStepLR":
        g_scheduler = lr_scheduler.MultiStepLR(g_optimizer,
                                               config["TRAIN"]["LR_SCHEDULER"]["MILESTONES"],
                                               config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])
        d_scheduler = lr_scheduler.MultiStepLR(d_optimizer,
                                               config["TRAIN"]["LR_SCHEDULER"]["MILESTONES"],
                                               config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])

    else:
        raise NotImplementedError(f"LR Scheduler {config['TRAIN']['LR_SCHEDULER']['NAME']} is not implemented.")

    return g_scheduler, d_scheduler


def train(
        g_model: nn.Module,
        ema_g_model: nn.Module,
        d_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        content_criterion: model.ContentLoss,
        adversarial_criterion: nn.BCEWithLogitsLoss,
        g_optimizer: optim.Adam,
        d_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    # Calculate how many batches of data there are under a dataset iterator
    batches = len(train_data_prefetcher)

    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    g_losses = AverageMeter("G Loss", ":6.6f", Summary.NONE)
    d_losses = AverageMeter("D Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, g_losses, d_losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Set the model to training mode
    g_model.train()
    d_model.train()

    # Define loss function weights
    pixel_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["WEIGHT"]).to(device)
    feature_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["WEIGHT"]).to(device)
    adversarial_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["WEIGHT"]).to(device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

    # Used for discriminator binary classification output, the input sample comes from the data set (real sample) is marked as 1, and the input sample comes from the generator (generated sample) is marked as 0
    batch_size = batch_data["gt"].shape[0]
    if config["MODEL"]["D"]["NAME"] == "discriminator_for_vgg":
        real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=device)
        fake_label = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=device)
    elif config["MODEL"]["D"]["NAME"] == "discriminator_for_unet":
        image_height = config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"]
        image_width = config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"]
        real_label = torch.full([batch_size, 1, image_height, image_width], 1.0, dtype=torch.float, device=device)
        fake_label = torch.full([batch_size, 1, image_height, image_width], 0.0, dtype=torch.float, device=device)
    else:
        raise ValueError(f"The `{config['MODEL']['D']['NAME']}` is not supported.")

    while batch_data is not None:
        # Load batches of data
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # image data augmentation
        gt, lr = random_crop_torch(gt,
                                   lr,
                                   config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"],
                                   config["SCALE"])
        gt, lr = random_rotate_torch(gt, lr, config["SCALE"], [0, 90, 180, 270])
        gt, lr = random_vertically_flip_torch(gt, lr)
        gt, lr = random_horizontally_flip_torch(gt, lr)

        # Record the time to load a batch of data
        data_time.update(time.time() - end)

        # start training the generator model
        # Disable discriminator backpropagation during generator training
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize the generator model gradient
        g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and confrontation loss
        with amp.autocast():
            sr = g_model(lr)
            pixel_loss = pixel_criterion(sr, gt)
            feature_loss = content_criterion(sr, gt)
            adversarial_loss = adversarial_criterion(d_model(sr), real_label)
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))
            feature_loss = torch.sum(torch.mul(feature_weight, feature_loss))
            adversarial_loss = torch.sum(torch.mul(adversarial_weight, adversarial_loss))
            # Compute generator total loss
            g_loss = pixel_loss + feature_loss + adversarial_loss
        # Backpropagation generator loss on generated samples
        scaler.scale(g_loss).backward()
        # update generator model weights
        scaler.step(g_optimizer)
        scaler.update()
        # end training generator model

        # start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradient
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model on real samples
        with amp.autocast():
            gt_output = d_model(gt)
            d_loss_gt = adversarial_criterion(gt_output, real_label)

        # backpropagate discriminator's loss on real samples
        scaler.scale(d_loss_gt).backward()

        # Calculate the classification score of the generated samples by the discriminator model
        with amp.autocast():
            sr_output = d_model(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
        # backpropagate discriminator loss on generated samples
        scaler.scale(d_loss_sr).backward()

        # Compute the discriminator total loss value
        d_loss = d_loss_gt + d_loss_sr
        # Update discriminator model weights
        scaler.step(d_optimizer)
        scaler.update()
        # end training discriminator model

        if config["MODEL"]["EMA"]["ENABLE"]:
            # update exponentially averaged model weights
            ema_g_model.update_parameters(g_model)

        # record the loss value
        d_losses.update(d_loss.item(), batch_size)
        g_losses.update(g_loss.item(), batch_size)

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output training log information once
        if batch_index % config["TRAIN"]["PRINT_FREQ"] == 0:
            # write training log
            iters = batch_index + epoch * batches
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Loss", d_loss_gt.item(), iters)
            writer.add_scalar("Train/D(SR)_Loss", d_loss_sr.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Feature_Loss", feature_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", torch.sigmoid_(torch.mean(gt_output.detach())).item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", torch.sigmoid_(torch.mean(sr_output.detach())).item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


if __name__ == "__main__":
    main()
