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
"""Tests for morpheus_framework.helpers.misc_helper"""

import numpy as np
import pytest

import morpheus_framework.helpers.misc_helper as mh


@pytest.mark.unit
def test_vaidate_input_types_is_str_passes_str():
    """Test morpheus_framework.helpers.misc_helpers.validate_input_types_is_str"""

    valid_inputs = ["test"]

    assert mh.vaidate_input_types_is_str(valid_inputs)


@pytest.mark.unit
def test_vaidate_input_types_is_str_passes_np():
    """Test morpheus_framework.helpers.misc_helpers.validate_input_types_is_str"""

    valid_inputs = [np.ones([2, 2])]

    assert not mh.vaidate_input_types_is_str(valid_inputs)


@pytest.mark.unit
def test_vaidate_input_types_is_str_raises_val_for_two_types():
    """Test morpheus_framework.helpers.misc_helpers.validate_input_types_is_str fails"""

    invalid_inputs = [np.ones([2, 2]), "test"]

    with pytest.raises(ValueError):
        mh.vaidate_input_types_is_str(invalid_inputs)


@pytest.mark.unit
def test_vaidate_input_types_is_str_raises_val_for_wrong_type():
    """Test morpheus_framework.helpers.misc_helpers.validate_input_types_is_str fails"""

    invalid_inputs = [1]

    with pytest.raises(ValueError):
        mh.vaidate_input_types_is_str(invalid_inputs)


@pytest.mark.unit
def test_validate_parallel_params_fails_for_both_given():
    """Test morpheus_framework.helper.misc_helper.validate_parallel_params"""

    gpus = [0, 1]
    cpus = [3]
    out_dir = None

    with pytest.raises(ValueError):
        mh.validate_parallel_params(gpus, cpus, out_dir)


@pytest.mark.unit
def test_validate_parallel_params_passes_both_none():
    """Test morpheus_framework.helper.misc_helper.validate_parallel_params"""

    gpus = None
    cpus = None
    out_dir = None

    expected_workers = [0]
    expected_is_gpu = False

    actual_workers, actual_is_gpu = mh.validate_parallel_params(gpus, cpus, out_dir)

    assert actual_workers == expected_workers
    assert actual_is_gpu == expected_is_gpu


@pytest.mark.unit
def test_validate_parallel_params_fails_single_gpu():
    """Test morpheus_framework.helper.misc_helper.validate_parallel_params"""

    gpus = [0]
    cpus = None
    out_dir = "."

    with pytest.raises(ValueError):
        mh.validate_parallel_params(gpus, cpus, out_dir)


@pytest.mark.unit
def test_validate_parallel_params_fails_no_out_dir():
    """Test morpheus_framework.helper.misc_helper.validate_parallel_params"""

    gpus = [0, 1]
    cpus = None
    out_dir = None

    with pytest.raises(ValueError):
        mh.validate_parallel_params(gpus, cpus, out_dir)


@pytest.mark.unit
def test_validate_parallel_params_passes_multi_gpu():
    """Test morpheus_framework.helper.misc_helper.validate_parallel_params"""

    gpus = [0, 1]
    cpus = None
    out_dir = "."

    expected_workers = gpus
    expected_is_gpu = True

    actual_workers, actual_is_gpu = mh.validate_parallel_params(gpus, cpus, out_dir)

    assert actual_workers == expected_workers
    assert actual_is_gpu == expected_is_gpu


@pytest.mark.unit
def test_validate_parallel_params_fails_cpus_lt_2():
    """Test morpheus_framework.helper.misc_helper.validate_parallel_params"""

    gpus = None
    cpus = 1
    out_dir = "."

    with pytest.raises(ValueError):
        mh.validate_parallel_params(gpus, cpus, out_dir)


@pytest.mark.unit
def test_validate_parallel_params_passes_multi_cpu():
    """Test morpheus_framework.helper.misc_helper.validate_parallel_params"""

    gpus = None
    cpus = 3
    out_dir = "."

    expected_workers = np.arange(cpus)
    expected_is_gpu = False

    actual_workers, actual_is_gpu = mh.validate_parallel_params(gpus, cpus, out_dir)

    np.testing.assert_array_equal(actual_workers, expected_workers)
    assert actual_is_gpu == expected_is_gpu


@pytest.mark.unit
def test_arrays_not_same_size_false():
    """Test morpheus_framework.helper.misc_helper.arrays_not_same_size"""
    arrays = [np.ones([2, 2]), np.ones([2, 2])]

    assert not mh.arrays_not_same_size(arrays)


@pytest.mark.unit
def test_arrays_not_same_size_true():
    """Test morpheus_framework.helper.misc_helper.arrays_not_same_size"""
    arrays = [np.ones([2, 2]), np.ones([2, 3])]

    assert mh.arrays_not_same_size(arrays)


from functools import partial


@pytest.mark.unit
def test_apply_tuple_args():
    """Test morpheus_framework.helper.misc_helper.apply"""

    f = lambda a, b, c: None
    vals = iter([(1, 2, 3), (4, 5, 6)])

    mh.apply(f, vals)


from functools import partial


@pytest.mark.unit
def test_apply_single_arg():
    """Test morpheus_framework.helper.misc_helper.apply"""

    f = lambda a: None
    vals = iter([1, 2])

    mh.apply(f, vals)
