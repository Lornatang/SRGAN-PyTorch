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

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator
from model import Generator
from model import PerceptualLoss

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
# 实验名称.
exp_name = "exp001"

# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":
    # 1. Dataset path.
    dataroot              = "data/ImageNet/train"
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
    p_epochs              = 165                                         
    g_epochs              = 33
                                             
    # 5. Loss function.
    pixel_criterion       = nn.MSELoss().to(device)                        
    perceptual_criterion  = PerceptualLoss().to(device)                   
    adversarial_criterion = nn.BCELoss().to(device)           
    # Loss function weight.
    perceptual_weight     = 6e-03
    adversarial_weight    = 1e-03

    # 6. Optimizer.
    p_optimizer           = optim.Adam(generator.parameters(),     1e-4) 
    d_optimizer           = optim.Adam(discriminator.parameters(), 1e-4) 
    g_optimizer           = optim.Adam(generator.parameters(),     1e-4) 

    # 7. Leaning scheduler.
    d_scheduler           = StepLR(d_optimizer, g_epochs // 2)           
    g_scheduler           = StepLR(g_optimizer, g_epochs // 2)           

    # 8. Training log.
    writer                = SummaryWriter(os.path.join("sample", "logs", exp_name))

    # Exp model name.
    p_filename            = f"P-{exp_name}.pth"
    d_filename            = f"D-{exp_name}.pth"
    g_filename            = f"G-{exp_name}.pth"

# ==============================================================================
#                              Validate configure
# ==============================================================================
if mode == "validate":
    net        = Generator().to(device)
    # Model weight path.
    model_path = f"results/G-{exp_name}.pth"
    # Test dataset path.
    lr_dir     = f"data/Set5/LRbicx4"
    sr_dir     = f"results/{exp_name}"
    hr_dir     = f"data/Set5/GTmod12"                        
