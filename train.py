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
import argparse
import logging

import srgan_pytorch.models as models
from srgan_pytorch.utils.common import create_folder
from trainer import Trainer

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo-Realistic Single Image Super-Resolution Using a "
                                                 "Generative Adversarial Network.")
    parser.add_argument("data", metavar="DIR",
                        help="path to dataset")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan_4x4_16",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             " (default: srgan_4x4_16)")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="Number of data loading workers. (default:4)")
    parser.add_argument("--start-psnr-iter", default=0, type=int, metavar="N",
                        help="manual iter number (useful on restarts)")
    parser.add_argument("--psnr-iters", default=1000000, type=int, metavar="N",
                        help="The number of iterations is needed in the training of PSNR model. (default:1000000)")
    parser.add_argument("--start-iter", default=0, type=int, metavar="N",
                        help="manual iter number (useful on restarts)")
    parser.add_argument("--iters", default=200000, type=int, metavar="N",
                        help="The training of srgan model requires the number of iterations. (default:200000)")
    parser.add_argument("-b", "--batch-size", default=16, type=int, metavar="N",
                        help="mini-batch size (default: 16), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate. (default:0.0001)")
    parser.add_argument("--image-size", type=int, default=96,
                        help="Image size of real sample. (default:96).")
    parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                        help="Low to high resolution scaling factor. (default:4).")
    parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                        help="Path to latest checkpoint for model. (default: ````).")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--netP", default="", type=str, metavar="PATH",
                        help="Path to latest psnr checkpoint. (default: ````).")
    parser.add_argument("--netD", default="", type=str, metavar="PATH",
                        help="Path to latest discriminator checkpoint. (default: ````).")
    parser.add_argument("--netG", default="", type=str, metavar="PATH",
                        help="Path to latest generator checkpoint. (default: ````).")
    parser.add_argument("--manualSeed", type=int, default=1111,
                        help="Seed for initializing training. (default:1111)")
    parser.add_argument("--device", default="",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ````).")
    args = parser.parse_args()

    print("##################################################\n")
    print("Run Training Engine.\n")
    print(args)

    create_folder("output")
    create_folder("output/hr")
    create_folder("output/sr")
    create_folder("weights")

    logger.info("TrainingEngine:")
    print("\tAPI version .......... 0.1.1")
    print("\tBuild ................ 2020.11.30-1116-0c5adc7e")

    logger.info("Creating Training Engine")
    trainer = Trainer(args)

    logger.info("Staring training model")
    trainer.run()
    print("##################################################\n")

    logger.info("All training has been completed successfully.\n")
