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

import prettytable as pt
import torch
from thop import profile

import srgan_pytorch.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.")
parser.add_argument("-i", "--image-size", type=int, default=24,
                    help="Image size of low-resolution. (Default: 24)")
parser.add_argument("-b", "--batch-size", default=128, type=int,
                    metavar="N",
                    help="Mini-batch size (default: 128), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")


def inference(arch, cpu_data, cuda_data, args):
    cpu_model = models.__dict__[arch]()

    params = sum(x.numel() for x in cpu_model.parameters()) / 1E6

    # Cal flops and parameters.
    flops = profile(model=cpu_model, inputs=(cpu_data,), verbose=False)[0] / 1E9 * 2

    with torch.no_grad():
        start_time = time.time()
        for _ in range(args.batch_size):
            _ = cpu_model(cpu_data)
        cpu_speed = (time.time() - start_time) / args.batch_size * 1E3
        cuda_speed = 0.

        if args.gpu is not None:
            cuda_model = cpu_model.cuda(args.gpu)

            start_time = time.time()
            for _ in range(args.batch_size):
                _ = cuda_model(cuda_data)
            cuda_speed = (time.time() - start_time) / args.batch_size * 1E3

    return params, flops, cpu_speed, cuda_speed


def main():
    args = parser.parse_args()
    tb = pt.PrettyTable()

    cpu_data = torch.randn([1, 3, args.image_size, args.image_size])
    if args.gpu is not None:
        cuda_data = cpu_data.cuda(args.gpu)
    else:
        cuda_data = None

    tb.field_names = ["Model", "Params", "FLOPs", "CPU Speed", "GPU Speed"]

    for i in range(len(model_names)):
        value = inference(model_names[i], cpu_data, cuda_data, args)
        tb.add_row([f"{model_names[i].center(15)}",
                    f"{value[0]:4.2f}M",
                    f"{value[1]:4.1f}G",
                    f"{int(value[2]):4d}ms",
                    f"{int(value[3]):4d}ms"])

    print(tb)


if __name__ == "__main__":
    main()
