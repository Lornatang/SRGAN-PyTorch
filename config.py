# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "train_srresnet"
# Experiment name, easy to save weights and log files
exp_name = "SRResNet_baseline"

if mode == "train_srresnet":
    # Dataset address
    train_image_dir = "data/ImageNet/SRGAN/train"
    valid_image_dir = "data/ImageNet/SRGAN/valid"
    test_lr_image_dir = f"data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = f"data/Set5/GTmod12"

    image_size = 96
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total num epochs
    epochs = 44

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    print_frequency = 100

if mode == "train_srgan":
    # Dataset address
    train_image_dir = "data/ImageNet/SRGAN/train"
    valid_image_dir = "data/ImageNet/SRGAN/valid"
    test_lr_image_dir = f"data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = f"data/Set5/GTmod12"

    image_size = 96
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = "results/SRResNet_baseline/g_last.pth.tar"
    resume_d = ""
    resume_g = ""

    # Total num epochs
    epochs = 9

    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.35"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # Loss function weight
    content_weight = 1.0
    adversarial_weight = 0.001

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # LR scheduler parameter
    lr_scheduler_step_size = epochs // 2
    lr_scheduler_gamma = 0.1

    print_frequency = 100

if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/g_last.pth.tar"
