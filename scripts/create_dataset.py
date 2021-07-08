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
import logging
import os
import random
import shutil

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main():
    r""" Make training data set and validation data set."""
    total_image_lists = os.listdir(os.path.join("train", "input"))
    # The original data set is divided into 9:1 (train:test)
    test_image_lists = random.sample(total_image_lists,
                                     int(len(total_image_lists) / 10))
    # Move the validation set to the specified location.
    for test_image_name in test_image_lists:
        filename = os.path.join("train", "input", test_image_name)
        logger.info(f"Process: `{filename}`.")
        # Move the test image into the test folder.
        shutil.move(os.path.join("train", "input", test_image_name),
                    os.path.join("test", "input", test_image_name))
        shutil.move(os.path.join("train", "target", test_image_name),
                    os.path.join("test", "target", test_image_name))


if __name__ == "__main__":
    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.07.02")

    main()
