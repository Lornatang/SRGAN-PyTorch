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
import argparse
import logging

import torch

import srgan_pytorch.models as models

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         ". (Default: srgan)")
parser.add_argument("--model-path", type=str, metavar="PATH", required=True,
                    help="Path to latest checkpoint for model.")
args = parser.parse_args()

model = models.__dict__[args.arch]()
model.load_state_dict(torch.load(args.model_path)["state_dict"])
torch.save(model.state_dict(), "Generator.pth")
logger.info("Model convert done.")
