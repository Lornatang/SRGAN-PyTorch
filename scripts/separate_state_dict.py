# Copyright 2022 Lorna Author. All Rights Reserved.
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
from collections import OrderedDict

import torch

import argparse


def main(args):
    # Process parameter dictionary
    state_dict = torch.load(args.inputs_model_path, map_location=torch.device("cpu"))["state_dict"]
    new_state_dict = OrderedDict()

    # Delete _orig_mod. in the parameter name
    for k, v in state_dict.items():
        name = k[10:]
        new_state_dict[name] = v

    torch.save({"state_dict": new_state_dict}, args.output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_model_path", type=str, help="Path to the model to be converted")
    parser.add_argument("--output_model_path", type=str, help="Path to the converted model")
    args = parser.parse_args()
    main(args)
