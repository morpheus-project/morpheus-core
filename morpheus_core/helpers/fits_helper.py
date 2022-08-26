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

from functools import partial
from itertools import count
from morpheus_core.helpers import misc_helper
from typing import List, Tuple

import numpy as np
from astropy.io import fits


def open_file(
    file_name: str, mode: str = "readonly"
) -> Tuple[fits.HDUList, np.ndarray]:
    """Gets the HDULS and data handles for all the files in file_names.

       This is a convience function to opening a singls FITS file using memmap.

        Args:
            file_name (str): filename to open
            mode (str): the mode to pass to fits.open
        Returns:
            Tuple containing the HDUL and the corresponding numpy array
    """
    hdul = fits.open(file_name, mode=mode, memmap=True)
    return hdul, hdul[0].data


def open_files(
    file_names: List[str], mode: str = "readonly"
) -> Tuple[List[fits.HDUList], List[np.ndarray]]:
    """Gets the HDULS and data handles for all the files in file_names.

       This is a convience function to opening multiple FITS files using
       memmap.

        Args:
            file_names (List[str]): a list of file names including paths to FITS
                                    files
            mode (str): the mode to pass to fits.open
        Returns:
            Tuple of a list numpy arrays that are the mmapped data handles for
            each of the FITS files and the HDULs that go along with them
    """
    return zip(*map(partial(open_file, mode=mode), file_names))


def dtype_to_bytes_per_value(dtype: np.dtype) -> int:
    """Gets the number of bytes as an int for each numpy datatype.

        Args:
            dtype (np.dtype): the numpy datatype to get the bytes for
        Returns:
            The number of bytes, as an int, for the given numpy datatype
        Raises:
            ValueError for a value that is not one of: np.uint8, np.int16,
            np.int32, np.float32, np.float64
    """

    if dtype == np.uint8:
        bytes_per_value = 1
    elif dtype == np.int16:
        bytes_per_value = 2
    elif dtype == np.int32:
        bytes_per_value = 4
    elif dtype == np.float32:
        bytes_per_value = 4
    elif dtype == np.float64:
        bytes_per_value = 8
    else:
        err_msg = " ".join(
            [
                "Invalid dtype. Please use one of the following: np.uint8,",
                "np.int16, np.int32, np.float32, np.float64",
            ]
        )
        raise ValueError(err_msg)

    return bytes_per_value


def create_file(file_name: str, shape: Tuple[int], dtype: np.dtype) -> None:
    """Creates a fits file without loading it into memory.

    This is a helper method to create large FITS files without loading an
    array into memory. The method follows the direction given at:
    http://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html

    Args:
        file_name (str): the complete path to the file to be created.
        data_shape (tuple): a tuple describe the shape of the file to be
                            created, the shape should be one of the following
                            shapes: (n, w, h) or (n, w, h, 2)
        dtype (numpy.dtype): the numpy datatype used in the array
    Returns:
        None

    TODO: Figure out why this throws warning about size occasionally
            when files that are created by it are opened
    """

    bytes_per_value = dtype_to_bytes_per_value(dtype)
    # if len(shape) < 3:
    #     raise ValueError("Invalid shape, should be (w,h,n) or (w,h,n,2), even for n=1")

    stub_size = [50, 50]
    if len(shape) == 3:
        stub_size = [shape[0]] + stub_size
    if len(shape) == 4:
        stub_size = [shape[0]] + stub_size + [2]

    stub = np.zeros(stub_size, dtype)

    hdu = fits.PrimaryHDU(data=stub)
    header = hdu.header
    while len(header) < (36 * 4 - 1):
        header.append()

    shape = list(reversed(shape))  # for some reason the dims are backwards in fits

    def add_dim(shp, i):
        header[f"NAXIS{i}"] = shp

    misc_helper.apply(add_dim, zip(shape, count(start=1)))

    header.tofile(file_name)

    with open(file_name, "rb+") as f:
        header_size = len(header.tostring())
        data_size = np.prod(shape) * bytes_per_value - 1

        f.seek(header_size + data_size)
        f.write(b"\0")
