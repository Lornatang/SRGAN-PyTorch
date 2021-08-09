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

# ==============================================================================
# File description: Realize the parameter configuration function of data set, model, training and verification code.
# ==============================================================================

import os
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from model import PerceptualLoss
from model import Discriminator
from model import Generator

# ==============================================================================
#                              Common configure
# ==============================================================================
torch.manual_seed(0)
cudnn.benchmark = True
cudnn.deterministic = False
device = torch.device("cuda:0")
# Runing mode.
mode = "train"
scale_factor = 4

# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":
    # 1. Dataset path.
    dataroot              = "data/DIV2K/train"
    image_size            = 96
    batch_size            = 16

    # 2. Define model.
    discriminator         = Discriminator().to(device)
    generator             = Generator().to(device)

    # 3. Reume training.
    start_p_epoch         = 0                                              
    start_g_epoch         = 0                                              
    resume                = False                                          
    resume_p_weight       = ""                                             
    resume_d_weight       = ""                                             
    resume_g_weight       = ""                                             

    # 4. Number of epochs.
    p_epochs              = 20000                                         
    g_epochs              = 4000
                                             
    # 5. Loss function.
    pixel_criterion       = nn.MSELoss().to(device)                        
    perceptual_criterion  = PerceptualLoss().to(device)                   
    adversarial_criterion = nn.BCELoss().to(device)           
    # Loss function weight.
    pixel_weight          = 1e-00
    perceptual_weight     = 2e-06
    adversarial_weight    = 1e-03

    # 6. Optimizer.
    p_lr                  = 1e-4
    d_lr                  = 1e-4
    g_lr                  = 1e-4
    p_optimizer           = optim.Adam(generator.parameters(),     p_lr) 
    d_optimizer           = optim.Adam(discriminator.parameters(), d_lr) 
    g_optimizer           = optim.Adam(generator.parameters(),     g_lr) 

    # 7. Leaning scheduler.
    d_scheduler           = StepLR(d_optimizer, g_epochs // 2)           
    g_scheduler           = StepLR(g_optimizer, g_epochs // 2)           

    # 8. Training log.
    times                 = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    writer                = SummaryWriter(os.path.join("samples", "logs", times))

# ==============================================================================
#                              Validate configure
# ==============================================================================
if mode == "validate":
    dataset = "Set5"

    net = Generator().to(device)
    net.load_state_dict(torch.load("", map_location=device))
    
    if dataset == "Set5":
        # Set5 dataset.
        lr_dir = "data/Set5/LRbicx4"
        sr_dir = "results/test/Set5"
        hr_dir = "data/Set5/GTmod12"
    elif dataset == "Set14":
        # Set14 dataset.
        lr_dir = "data/Set14/LRbicx4"
        sr_dir = "results/test/Set14"
        hr_dir = "data/Set14/GTmod12"
    elif dataset == "BSD100":
        # BSD100 dataset.
        lr_dir = "data/BSD100/LRbicx4"
        sr_dir = "results/test/BSD100"
        hr_dir = "data/BSD100/GTmod12"
    elif dataset == "Custom":
        # Custom dataset.
        lr_dir = "data/Custom/LRbicx4"
        sr_dir = "results/test/Custom"
        hr_dir = "data/Custom/HR"
    else:
        # Set5 dataset.
        lr_dir = "data/Set5/LRbicx4"
        sr_dir = "results/test/Set5"
        hr_dir = "data/Set5/GTmod12"
