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
# ============================================================================
import logging
import os

__all__ = ["create_folder", "check_image_file"]

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def create_folder(dir_name):
    r""" Create folder, this is a relative path.
    
    Args:
        dir_name (str): The name of the folder to be created.
    """
    dir_path = os.path.join(os.getcwd(), dir_name)
    try:
        os.makedirs(dir_name)
        logger.info(f"Create `{dir_path}` directory successful.")
    except OSError:
        logger.warning(f"Directory `{dir_path}` already exists!")
        pass


def check_image_file(filename):
    r""" Filter non image files in directory.

    Note:
        The performance of the model trained using `png, bmp` 
        compression format images is generally better.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.
    """
    support_format = [
        "jpg", "jpeg", "png", "bmp", "tiff", "JPG", "JPEG", "PNG", "BMP", "TIFF"
    ]
    return any(filename.endswith(extension) for extension in support_format)
