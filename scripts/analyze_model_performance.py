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
import time

import srgan_pytorch.models as models
import torch

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         ". (Default: `srgan`)")
parser.add_argument("-i", "--image-size", type=int, default=24,
                    help="Image size of low-resolution. (Default: 24)")
parser.add_argument("--mode", default="cpu", type=bool,
                    help="Use GPU analyze model performance.")


def main():
    args = parser.parse_args()

    model = models.__dict__[args.arch]()
    model.eval()
    data = torch.ones([1, 3, args.image_size, args.image_size])

    if args.mode == "gpu":
        model = model.cuda("0")
        data = data.cuda("0")

    for _ in range(5):
        start = time.time()
        _ = model(data)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Time:{(end - start) * 1000:.2f}ms")

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        _ = model(data)
    print(prof.table())
    prof.export_chrome_trace("profile.json")


if __name__ == "__main__":
    main()
