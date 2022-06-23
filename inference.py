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
import argparse

import cv2
import numpy as np
import torch

import config
import imgproc
from model import Generator


def main(args):
    # Initialize the model
    model = Generator()
    model = model.to(memory_format=torch.channels_last, device=config.device)
    print("Build SRGAN model successfully.")

    # Load the SRGAN model weights
    checkpoint = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load SRGAN model weights `{args.weights_path}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    # Read LR image and HR image
    lr_image = cv2.imread(args.inputs_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

    # Convert BGR channel image format data to RGB channel image format data
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

    # Convert RGB channel image format data to Tensor channel image format data
    lr_tensor = imgproc.image_to_tensor(lr_image, False, False).unsqueeze_(0)

    # Transfer Tensor channel image format data to CUDA device
    lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, sr_image)

    print(f"SR image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the SRGAN model generator super-resolution images.")
    parser.add_argument("--inputs_path", type=str, help="Low-resolution image path.")
    parser.add_argument("--output_path", type=str, help="Super-resolution image path.")
    parser.add_argument("--weights_path", type=str, help="Model weights file path.")
    args = parser.parse_args()

    main(args)
