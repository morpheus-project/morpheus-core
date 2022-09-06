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
import os

import numpy as np
import pytest
from astropy.io import fits

import morpheus_core.helpers.fits_helper as fh
import morpheus_core.tests.helpers as helper


@pytest.mark.unit
def test_open_file():
    """Tests morpheus_core.helpers.fits_helper.open_file"""
    helper.setup()
    sample_location = helper.make_sample_file()

    expected_array = np.arange(100).reshape([10, 10])

    hdul, actual_array = fh.open_file(sample_location)

    np.testing.assert_array_equal(expected_array, actual_array)

    helper.tear_down()


@pytest.mark.unit
def test_open_files():
    """Tests morpheus_core.helpers.fits_helper.open_file"""
    helper.setup()
    sample_location = helper.make_sample_file()
    sample2_location = helper.make_sample_file2()

    expected_array = np.arange(100).reshape([10, 10])

    _, actual_arrays = fh.open_files([sample_location, sample2_location])

    np.testing.assert_array_equal(expected_array, actual_arrays[0])
    np.testing.assert_array_equal(expected_array, actual_arrays[1])

    helper.tear_down()


@pytest.mark.unit
def test_dtype_to_bytes_per_value():
    """Tests morpheus_core.helpers.fits_helper.dtype_to_bytes_per_value"""
    types = [np.uint8, np.int16, np.int32, np.float32, np.float64]
    expected_bytes_per_value = [1, 2, 4, 4, 8]

    actual_bytes_per_value = list(map(fh.dtype_to_bytes_per_value, types))

    assert actual_bytes_per_value == expected_bytes_per_value


@pytest.mark.unit
def test_dtype_to_bytes_per_value_fails():
    """Tests morpheus_core.helpers.fits_helper.dtype_to_bytes_per_value"""
    with pytest.raises(ValueError):
        fh.dtype_to_bytes_per_value(bool)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_create_file():
    """Tests morpheus_core.helpers.fits_helper.create_file"""
    helper.setup()

    shape = (100, 100)

    tmp_out = os.path.join(helper.TMP_DIR, "test.fits")
    fh.create_file(tmp_out, shape, np.float32)

    actual = fits.getdata(tmp_out)
    assert actual.shape == shape

    helper.tear_down()
