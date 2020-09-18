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
from itertools import product, repeat

import numpy as np
import pytest
from astropy.io import fits

import morpheus_framework.helpers.label_helper as lh
from morpheus_framework.helpers.label_helper import finalize_rank_vote
import morpheus_framework.tests.helpers as helper


@pytest.mark.unit
def test_get_mean_var_array_in_mem():
    """tests morpheus_framework.helpers.label_helper.get_mean_var_array"""

    shape = [100, 100, 1]

    hdul, array = lh.get_mean_var_array(shape)

    assert hdul is None
    assert array.shape == (100, 100, 1, 2)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_get_mean_var_array_on_disk():
    """tests morpheus_framework.helpers.label_helper.get_mean_var_array"""

    helper.setup()

    shape = [100, 100, 1]

    out_file = os.path.join(helper.TMP_DIR, "test.fits")
    hdul, in_mem_arrays = lh.get_mean_var_array(shape, write_to=out_file)
    on_disk_array = fits.getdata(out_file)

    # validate the returned array
    assert hdul is not None
    assert in_mem_arrays.shape == (100, 100, 1, 2)
    assert on_disk_array.shape == (100, 100, 1, 2)

    helper.tear_down()


@pytest.mark.unit
def test_get_rank_vote_array_in_mem():
    """tests morpheus_framework.helpers.label_helper.get_mean_var_array"""

    shape = [100, 100, 1]

    hdul, array = lh.get_rank_vote_array(shape)

    assert hdul is None
    assert array.shape == (100, 100, 1)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_get_rank_vote_array_on_disk():
    """tests morpheus_framework.helpers.label_helper.get_mean_var_array"""

    helper.setup()

    shape = [100, 100, 1]

    out_file = os.path.join(helper.TMP_DIR, "test.fits")
    hdul, in_mem_array = lh.get_rank_vote_array(shape, write_to=out_file)
    on_disk_array = fits.getdata(out_file)

    # validate the returned array
    assert hdul is not None
    assert in_mem_array.shape == (100, 100, 1)
    assert on_disk_array.shape == (100, 100, 1)

    helper.tear_down()


@pytest.mark.unit
def test_get_n_array_in_mem():
    """tests morpheus_framework.helpers.label_helper.get_mean_var_array"""

    shape = [100, 100]

    hdul, array = lh.get_n_array(shape)

    assert hdul is None
    assert array.shape == (100, 100)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_get_n_array_on_disk():
    """tests morpheus_framework.helpers.label_helper.get_mean_var_array"""

    helper.setup()

    shape = [100, 100]

    out_file = os.path.join(helper.TMP_DIR, "test.fits")
    hdul, array = lh.get_n_array(shape, write_to=out_file)
    on_disk_array = fits.getdata(out_file)

    # validate the returned array
    assert hdul is not None
    assert array.shape == (100, 100)
    assert on_disk_array.shape == (100, 100)

    helper.tear_down()


@pytest.mark.unit
def test_get_windowed_index_generator_fails():
    """test morpheus_framework.helpers.label_helper.get_windowed_index_generator"""
    total_wh = (100, 100)
    window_shape = (10, 10)
    stride = (1,)

    with pytest.raises(ValueError):
        lh.get_windowed_index_generator(total_wh, window_shape, stride)


@pytest.mark.unit
def test_get_windowed_index_generator_stride_one():
    """test morpheus_framework.helpers.label_helper.get_windowed_index_generator"""
    total_wh = (100, 100)
    window_shape = (10, 10)
    stride = (1, 1)

    test = np.zeros(total_wh)

    ys, xs = zip(*lh.get_windowed_index_generator(total_wh, window_shape, stride))

    test[ys, xs] = 1
    assert test[: total_wh[0] - window_shape[0], : total_wh[1] - window_shape[1]].all()


@pytest.mark.unit
def test_get_windowed_index_generator_stride_one():
    """test morpheus_framework.helpers.label_helper.get_windowed_index_generator"""
    total_wh = (100, 100)
    window_shape = (10, 10)
    stride = (2, 2)

    test = np.zeros(total_wh)

    ys, xs = zip(*lh.get_windowed_index_generator(total_wh, window_shape, stride))

    test[ys, xs] = 1

    # assert that all of the desired indicies where hit
    assert test[
        : total_wh[0] - window_shape[0] : stride[0],
        : total_wh[1] - window_shape[1] : stride[1],
    ].all()

    # assert that all of the skipped indicies where missed
    assert not test[
        1 : total_wh[0] - window_shape[0] + 1 : stride[0],
        1 : total_wh[1] - window_shape[1] + 1 : stride[1],
    ].any()


@pytest.mark.unit
def test_get_final_map_final():
    """Test the get_final_map method on complete array."""

    total_shape = (100, 100)
    update_mask_shape = (10, 10)
    stride = (1, 1)
    output_idx = (
        total_shape[0] - update_mask_shape[0],
        total_shape[1] - update_mask_shape[1],
    )

    expected_idx = product(range(update_mask_shape[0]), range(update_mask_shape[1]))

    actual_idx = lh.get_final_map(total_shape, update_mask_shape, stride, output_idx)

    assert all([a == b for a, b in zip(expected_idx, actual_idx)])


@pytest.mark.unit
def test_get_final_map_end_y():
    """Test the get_final_map method on final row."""

    total_shape = (100, 100)
    update_mask_shape = (10, 10)
    stride = (1, 1)
    output_idx = (total_shape[0] - update_mask_shape[0], 0)

    expected_idx = zip(range(update_mask_shape[0]), repeat(0, update_mask_shape[1]))

    actual_idx = lh.get_final_map(total_shape, update_mask_shape, stride, output_idx)

    assert all([a == b for a, b in zip(expected_idx, actual_idx)])


@pytest.mark.unit
def test_get_final_map_end_x():
    """Test the get_final_map method on final col."""

    total_shape = (100, 100)
    update_mask_shape = (10, 10)
    stride = (1, 1)
    output_idx = (0, total_shape[1] - update_mask_shape[1])

    expected_idx = zip(repeat(0, update_mask_shape[0]), range(update_mask_shape[1]))

    actual_idx = lh.get_final_map(total_shape, update_mask_shape, stride, output_idx)

    assert all([a == b for a, b in zip(expected_idx, actual_idx)])


@pytest.mark.unit
def test_get_final_map_first():
    """Test the get_final_map method on any non final position."""

    total_shape = (100, 100)
    update_mask_shape = (10, 10)
    stride = (1, 1)
    output_idx = (0, 0)
    expected_idx = [(0, 0)]

    actual_idx = lh.get_final_map(total_shape, update_mask_shape, stride, output_idx)

    assert list(actual_idx) == expected_idx


@pytest.mark.unit
def test_update_n():
    """Test the get_final_map method on any non final position."""

    update_mask = np.ones([10, 10])
    n = np.zeros([100, 100])
    output_idx = (0, 0)

    n_new = lh.update_n(update_mask, n, output_idx)

    assert n_new[:10, :10].all()
    assert n_new.sum() == update_mask.sum()


@pytest.mark.unit
def test_iterative_mean():
    """Tests morpheus_framework.helpers.label_helper.iterative_mean."""

    shape = (10, 10)
    n = np.ones(shape) * 2
    curr_mean = np.ones(shape)
    update_mask = curr_mean.copy()
    x_n = curr_mean * 2

    actual_mean = lh.iterative_mean(n, curr_mean, x_n, update_mask)
    np.testing.assert_array_equal(np.ones(shape) * 1.5, actual_mean)


@pytest.mark.unit
def test_iterative_variance():
    """Tests morpheus_framework.helpers.label_helper.iterative_variance"""
    shape = (10, 10)
    terms = [np.ones(shape) * i for i in range(9)]
    s_n = np.zeros(shape)
    update_mask = np.ones((shape))

    for i in range(9):
        curr_mean = np.mean(terms[: i + 1], axis=0)
        if i > 0:
            prev_mean = np.mean(terms[:i], axis=0)
        else:
            prev_mean = curr_mean.copy()

        s_n = lh.iterative_variance(s_n, terms[i], prev_mean, curr_mean, update_mask)

    n = np.ones(shape) * 9
    expected_sn = np.var(terms, axis=0) * n

    all_same = np.equal(expected_sn, s_n)

    assert all_same.all()


@pytest.mark.unit
def test_finalize_variance():
    """Tests morpheus_framework.helpers.label_helper.finalize_variance"""

    shape = (10, 10)
    terms = [np.ones(shape) * i for i in range(9)]

    expected_var = np.var(terms, axis=0)

    n = np.ones(shape) * 9
    sn = expected_var * n

    final_map = list(product(range(shape[0]), range(shape[1])))

    var = lh.finalize_variance(n, final_map, sn)

    all_same = np.equal(expected_var, var)

    assert all_same.all()


@pytest.mark.unit
def test_update_single_class_mean_var():
    """Tests morpheus_framework.helpers.label_helper.update_single_class_mean_var"""
    shape = (10, 10)
    terms = np.dstack([np.ones(shape) * i for i in range(9)])
    update_mask = np.ones((shape))
    n = np.ones(shape) * 9

    mean_var = np.zeros((10, 10, 2))

    mean_var[:, :, 0] = np.mean(terms[:, :, :-1], axis=-1)
    mean_var[:, :, 1] = np.var(terms[:, :, :-1], axis=-1) * 8

    actual_mean, acutal_var = lh.update_single_class_mean_var(
        update_mask, n, mean_var, terms[:, :, -1]
    )

    expected_mean = np.mean(terms, axis=-1)
    expected_var = np.var(terms, axis=-1) * n

    np.testing.assert_array_equal(actual_mean, expected_mean)
    np.testing.assert_array_equal(acutal_var, expected_var)


@pytest.mark.unit
def test_update_mean_var():
    """Tests morpheus_framework.helpers.label_helper.update_mean_var"""

    window_shape = (10, 10)
    total_shape = (100, 100)
    stride = (1, 1)

    update_mask = np.ones(window_shape)
    n = np.zeros(total_shape)
    output_array = np.zeros((100, 100, 1, 2))
    batch_output = np.ones((10, 10, 1))
    output_idx = (0, 0)

    lh.update_mean_var(update_mask, stride, n, output_array, batch_output, output_idx)

    assert output_array[:, :, :, 0].sum() == 100
    assert output_array[:, :, :, 1].sum() == 0


@pytest.mark.unit
def test_finalize_rank_vote():
    """Tests morpheus_framework.helpers.label_helper.finalize_rank_vote"""
    window_size = (10, 10)

    n = np.ones(window_size)
    final_map = [(0, 0)]
    output = np.dstack(
        [np.zeros(window_size), np.zeros(window_size), np.ones(window_size)]
    )

    return output.sum() == finalize_rank_vote(n, final_map, output).sum()


@pytest.mark.unit
def test_update_rank_vote():
    """Tests morpheus_framework.helpers.label_helper.update_rank_vote"""

    window_shape = (10, 10)
    total_shape = (100, 100)
    stride = (1, 1)

    update_mask = np.ones(window_shape)
    n = np.zeros(total_shape)
    output = np.zeros((100, 100, 3))
    single_output = np.dstack(
        [
            np.ones(window_shape) * 0,
            np.ones(window_shape) * 0.4,
            np.ones(window_shape) * 0.6,
        ]
    )
    output_idx = (0, 0)

    lh.update_rank_vote(update_mask, stride, n, output, single_output, output_idx)

    expected_output = np.dstack(
        [np.ones(window_shape) * 0, np.ones(window_shape) * 0, np.ones(window_shape)]
    )

    assert output.sum() == expected_output.sum()


if __name__ == "__main__":
    test_update_mean_var()
