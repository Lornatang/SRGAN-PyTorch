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
torch.manual_seed(0)                       # Set random seed.
upscale_factor   = 4                       # How many times the size of the high-resolution image in the data set is than the low-resolution image.
device           = torch.device("cuda:0")  # Use the first GPU for processing by default.
cudnn.benchmark  = True                    # If the dimension or type of the input data of the network does not change much, turn it on, otherwise turn it off.
mode             = "train"                 # Run mode. Specific mode loads specific variables.
exp_name         = "exp000"                # Experiment name.

# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":
    # 1. Configure dataset.
    train_dir             = "data/ImageNet/train"                 # The address of the training data set.
    valid_dir             = "data/ImageNet/valid"                 # Verify the address of the data set.
    image_size            = 96                                    # High-resolution image size in the training data set.
    batch_size            = 16                                    # Training data batch size.

    # 2. Configure model.
    discriminator         = Discriminator().to(device)            # Load the discriminator model.
    generator             = Generator().to(device)                # Load the generative model.

    # 3. Resume training.
    start_p_epoch         = 0                                     # The number of initial iterations of the generator training phase. When set to 0, it means incremental training.
    start_epoch           = 0                                     # The number of initial iterations of the adversarial network training. When set to 0, it means incremental training.
    resume                = False                                 # Set to `True` to continue training from the previous training progress.
    resume_p_weight       = ""                                    # Restore the weight of the generative model during generator training.
    resume_d_weight       = ""                                    # Restore the weight of the generative model during the training of the adversarial network.
    resume_g_weight       = ""                                    # Restore the weight of the discriminator model during the training of the adversarial network.

    # 4. Train epochs.
    p_epochs              = 128                                   # The total number of cycles of the generator training phase.
    epochs                = 32                                    # The total number of cycles in the training phase of the adversarial network.

    # 5. Loss function.
    pixel_criterion       = nn.MSELoss().to(device)               # Pixel loss.
    perceptual_criterion  = PerceptualLoss().to(device)           # Perceived loss.
    adversarial_criterion = nn.BCELoss().to(device)               # Fight against loss.
    # Loss function weight.
    pixel_weight          = 1.0
    perceptual_weight     = 0.06
    adversarial_weight    = 0.001

    # 6. Optimizer.
    p_optimizer           = optim.Adam(generator.parameters(),     0.0001, (0.9, 0.999))  # Generate model learning rate during generator training.
    d_optimizer           = optim.Adam(discriminator.parameters(), 0.0001, (0.9, 0.999))  # Discriminator learning rate during adversarial network training.
    g_optimizer           = optim.Adam(generator.parameters(),     0.0001, (0.9, 0.999))  # The learning rate of the generator during network training.

    # 7. Scheduler.
    d_scheduler           = StepLR(d_optimizer, epochs // 2)  # Identify the model scheduler during adversarial training.
    g_scheduler           = StepLR(g_optimizer, epochs // 2)  # Generate model scheduler during adversarial training.

    # 8. Training log.
    writer                = SummaryWriter(os.path.join("samples",  "logs", exp_name))

    # Additional variables.
    exp_dir1 = os.path.join("samples", exp_name)
    exp_dir2 = os.path.join("results", exp_name)

# ==============================================================================
#                              Validate configure
# ==============================================================================
if mode == "validate":
    exp_dir    = os.path.join("results", "test", exp_name)  # Additional variables.

    model      = Generator().to(device)                     # Load the super-resolution model.
    model_path = f"results/{exp_name}/g-best.pth"           # Model weight address.
    lr_dir     = f"data/Set5/LRbicx4"                       # Low resolution image address.
    sr_dir     = f"results/test/{exp_name}"                 # Super-resolution image address.
    hr_dir     = f"data/Set5/GTmod12"                       # High resolution image address.

