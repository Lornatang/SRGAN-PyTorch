# Copyright 2020 Lorna Authors. All Rights Reserved.
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
import random
import time

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

from srgan_pytorch import Generator

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--file", type=str, default="assets/test_1.jpg",
                    help="Test low resolution image name. (default:`assets/test_1.jpg`)")
parser.add_argument("--model-name", type=str, default="weights/netG.pth",
                    help="Generator model name.  (default:`weights/netG.pth`)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--image-size", type=int, default=88,
                    help="size of the data crop (squared assumed). (default:88)")
parser.add_argument("--upscale-factor", default=4, type=int, help="Super resolution upscale factor")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = Generator(scale_factor=args.upscale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args.model_name, map_location=device))

# Set model mode
model.eval()

# Load image
image = Image.open(args.file)
pre_process = transforms.Compose([transforms.Resize(args.image_size),
                                  transforms.ToTensor()])
image = pre_process(image).unsqueeze(0)
image = image.to(device)

start = time.clock()
high_resolution_fake_image = model(image)
elapsed = (time.clock() - start)
print(f"cost {elapsed:.4f}s")
vutils.save_image(high_resolution_fake_image.detach(), "result.jpg", normalize=True)
