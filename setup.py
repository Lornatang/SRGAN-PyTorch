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
import io
import os
import sys
from shutil import rmtree

from setuptools import Command
from setuptools import find_packages
from setuptools import setup

# Configure library params.
NAME = "srgan_pytorch"
DESCRIPTION = "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network."
URL = "https://github.com/Lornatang/SRGAN-PyTorch"
EMAIL = "liu_changyu@dakewe.com"
AUTHOR = "Liu Goodfellow"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "1.0.0"

# Libraries that must be installed.
REQUIRED = ["torch"]

# The following libraries directory need to be installed if you need to run all scripts.
EXTRAS = {}

# Find the current running location.
here = os.path.abspath(os.path.dirname(__file__))

# About README file description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Set Current Library Version.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(name=NAME,
      version=about["__version__"],
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author=AUTHOR,
      author_email=EMAIL,
      python_requires=REQUIRES_PYTHON,
      url=URL,
      packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
      install_requires=REQUIRED,
      extras_require=EXTRAS,
      include_package_data=True,
      license="Apache",
      classifiers=[
          # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
          "License :: OSI Approved :: Apache Software License",
          "Programming Language :: Python :: 3 :: Only"
      ],
      cmdclass={
          "upload": UploadCommand,
      },
      )
