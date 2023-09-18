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
                        default="./configs/train/SRResNet_x4-SRGAN_ImageNet-Set5.yaml",
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
    g_model, ema_g_model = build_model(config, device)
    pixel_criterion = define_loss(config, device)
    optimizer = define_optimizer(g_model, config)

    # Load the pretrained model
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"]:
        g_model = load_pretrained_state_dict(g_model,
                                             config["MODEL"]["G"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_G_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # Load the last training interruption model node
    if config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"]:
        g_model, ema_g_model, start_epoch, best_psnr, best_ssim, optimizer = load_resume_state_dict(
            g_model,
            ema_g_model,
            optimizer,
            None,
            config["MODEL"]["G"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"],
        )
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['RESUMED_G_MODEL']}` resume model weights successfully.")
    else:
        print("Resume training model not found. Start training from scratch.")

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
              train_data_prefetcher,
              pixel_criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              device,
              config)
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
                         "optimizer": optimizer.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
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
) -> [nn.Module, nn.Module or Any]:
    g_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           num_rcb=config["MODEL"]["G"]["NUM_RCB"])
    g_model = g_model.to(device)

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
    if config["MODEL"]["EMA"]["COMPILED"] and ema_g_model is not None:
        ema_g_model = torch.compile(ema_g_model)

    return g_model, ema_g_model


def define_loss(config: Any, device: torch.device) -> nn.MSELoss:
    if config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["NAME"] == "MSELoss":
        pixel_criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['PIXEL_LOSS']['NAME']} is not implemented.")
    pixel_criterion = pixel_criterion.to(device)

    return pixel_criterion


def define_optimizer(g_model: nn.Module, config: Any) -> optim.Adam:
    if config["TRAIN"]["OPTIM"]["NAME"] == "Adam":
        optimizer = optim.Adam(g_model.parameters(),
                               config["TRAIN"]["OPTIM"]["LR"],
                               config["TRAIN"]["OPTIM"]["BETAS"],
                               config["TRAIN"]["OPTIM"]["EPS"],
                               config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
    else:
        raise NotImplementedError(f"Optimizer {config['TRAIN']['OPTIM']['NAME']} is not implemented.")

    return optimizer


def train(
        g_model: nn.Module,
        ema_g_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.MSELoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_data_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    g_model.train()

    # Define loss function weights
    loss_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["WEIGHT"]).to(device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

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

        # Initialize the generator model gradient
        g_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = g_model(lr)
            pixel_loss = pixel_criterion(sr, gt)
            pixel_loss = torch.sum(torch.mul(loss_weight, pixel_loss))

        # Backpropagation
        scaler.scale(pixel_loss).backward()
        # update model weights
        scaler.step(optimizer)
        scaler.update()

        if config["MODEL"]["EMA"]["ENABLE"]:
            # update exponentially averaged model weights
            ema_g_model.update_parameters(g_model)

        # record the loss value
        losses.update(pixel_loss.item(), lr.size(0))

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output training log information once
        if batch_index % config["TRAIN"]["PRINT_FREQ"] == 0:
            # write training log
            iters = batch_index + epoch * batches
            writer.add_scalar("Train/Loss", pixel_loss.item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # Add 1 to the number of data batches
        batch_index += 1


if __name__ == "__main__":
    main()
