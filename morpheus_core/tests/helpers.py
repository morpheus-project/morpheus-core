# MIT License
# Copyright 2020 Ryan Hausen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================
"""Helpers for testing"""

import os
import shutil

import numpy as np
from astropy.io import fits

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(LOCAL_DIR, "../../tmp")


def safe_make_dir():
    """ """
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)


def setup() -> None:
    """ """
    safe_make_dir()


def tear_down() -> None:
    """ """
    shutil.rmtree(TMP_DIR)


def make_sample_file() -> str:
    """ """
    fits.PrimaryHDU(data=np.arange(100).reshape([10, 10])).writeto(
        os.path.join(TMP_DIR, "sample.fits")
    )

    return os.path.join(TMP_DIR, "sample.fits")


def make_sample_file2() -> str:
    """ """
    fits.PrimaryHDU(data=np.arange(100).reshape([10, 10])).writeto(
        os.path.join(TMP_DIR, "sample2.fits")
    )

    return os.path.join(TMP_DIR, "sample2.fits")


def make_mock_input() -> str:
    fits.PrimaryHDU(data=np.arange(int(100 * 100)).reshape([100, 100, 1])).writeto(
        os.path.join(TMP_DIR, "input.fits")
    )

    return os.path.join(TMP_DIR, "input.fits")
