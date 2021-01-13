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
from tester import Video

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo-Realistic Single Image Super-Resolution Using a "
                                                 "Generative Adversarial Network.")

    # basic parameters
    parser.add_argument("--file", type=str, required=True,
                        help="Test low resolution video name.")
    parser.add_argument("--outf", default="video", type=str, metavar="PATH",
                        help="The location of the image in the evaluation process. (default: ``video``).")
    parser.add_argument("--device", default="0",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``0``).")
    parser.add_argument("--view", action="store_true",
                        help="Super resolution real time to show.")

    # model parameters
    parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan_4x4_16",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             " (default: srgan_4x4_16)")
    parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                        help="Low to high resolution scaling factor. (default:4).")
    parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                        help="Path to latest checkpoint for model. (default: ````).")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")

    args = parser.parse_args()

    print("##################################################\n")
    print("Run Testing Engine.\n")
    print(args)

    create_folder(args.outf)

    logger.info("TestEngine:")
    print("\tAPI version .......... 0.1.1")
    print("\tBuild ................ 2020.11.30-1116-0c5adc7e")

    logger.info("Creating SR Engine")
    video = Video(args)

    logger.info("Start video super-resolution")
    video.run()
    print("##################################################\n")

    logger.info("Super-resolution video completed successfully.\n")
