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
"""Tests for morpheus_core.morpheus_core module."""
import os

import numpy as np
import pytest
from astropy.io import fits

import morpheus_core.tests.helpers as helpers
import morpheus_core.helpers.misc_helper as misc
from morpheus_core import morpheus_core


@pytest.mark.unit
def test_build_batch():
    """Tests morpheus_core.build_batch"""

    arr = np.arange(int(100 * 100)).reshape([100, 100, 1])
    window_size = (10, 10)
    batch_idxs = [(0, 0), (0, 1), (0, 2)]

    expected_sums = [45450, 45550, 45650]
    expected_idxs = [(0, 0), (0, 1), (0, 2)]

    arrs, actual_idxs = morpheus_core.build_batch([arr], window_size, batch_idxs)
    actual_sums = [a.sum() for a in arrs[0]]

    assert actual_sums == expected_sums
    assert actual_idxs == expected_idxs


@pytest.mark.unit
def test_predict_batch():
    """Tests morpheus_core.predict_batch"""

    model_f = lambda x: x.sum(axis=1).sum(axis=1)
    batch = np.ones([3, 10, 10])
    batch_idxs = [(0, 0), (0, 1), (0, 2)]

    expected_result = np.array([100, 100, 100])

    actual_result, actual_idxs = morpheus_core.predict_batch(model_f, batch, batch_idxs)

    np.testing.assert_array_equal(expected_result, actual_result)
    assert batch_idxs == actual_idxs


@pytest.mark.unit
def test_update_output_fails_invalid_choice():
    """Tests morpheus_core.update_output"""

    invalid_aggregate_method = "invalid"

    with pytest.raises(ValueError):
        morpheus_core.update_output(
            invalid_aggregate_method, None, (1, 1), 1, None, None, None, (1, 1)
        )


@pytest.mark.integration
def test_update_output_mean_var():
    """Tests morpheus_core.update_ouput"""

    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR
    update_map = np.ones([10, 10])
    stride = (1, 1)
    dilation = float(1)
    n = np.zeros([100, 100])
    outputs = np.zeros([100, 100, 1, 2])
    batch_out = np.ones([10, 10, 1])
    batch_idxs = (0, 0)

    morpheus_core.update_output(
        aggregate_method,
        update_map,
        stride,
        dilation,
        n,
        outputs,
        batch_out,
        batch_idxs,
    )

    assert n[0:10, 0:10].all()
    assert outputs[0:10, 0:10, 0, 0].sum() == 100
    assert outputs[0:10, 0:10, 0, 1].sum() == 0


@pytest.mark.integration
def test_update_output_mean_var():
    """Tests morpheus_core.update_ouput with mean_var aggregation"""

    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR
    update_map = np.ones([10, 10])
    stride = (1, 1)
    dilation = 1
    n = np.zeros([100, 100])
    outputs = np.zeros([100, 100, 1, 2])
    batch_out = np.ones([10, 10, 1])
    batch_idxs = (0, 0)

    morpheus_core.update_output(
        aggregate_method,
        update_map,
        stride,
        dilation,
        n,
        outputs,
        batch_out,
        batch_idxs,
    )

    assert n[0:10, 0:10].all()
    assert outputs[0:10, 0:10, 0, 0].sum() == 100
    assert outputs[0:10, 0:10, 0, 1].sum() == 0


@pytest.mark.integration
def test_update_output_rank_vote():
    """Tests morpheus_core.update_output with rank_vote aggregation"""

    aggregate_method = morpheus_core.AGGREGATION_METHODS.RANK_VOTE
    update_map = np.ones([10, 10])
    stride = (1, 1)
    dilation = 1
    n = np.zeros([100, 100])
    outputs = np.zeros([100, 100, 1])
    batch_out = np.ones([10, 10, 1])
    batch_idxs = (0, 0)

    morpheus_core.update_output(
        aggregate_method,
        update_map,
        stride,
        dilation,
        n,
        outputs,
        batch_out,
        batch_idxs,
    )

    assert n[0:10, 0:10].all()
    assert outputs[0:10, 0:10, 0].sum() == 100


@pytest.mark.integration
def test_predict_arrays_mean_var():
    """Tests morpheus_core.predict_arrays with mean_var aggergation"""
    total_shape = (100, 100, 1)

    model = lambda x: np.ones_like(x[0])
    model_inputs = [np.ones(total_shape)]
    n_classes = 1
    batch_size = 10
    window_size = (10, 10)
    stride = (1, 1)
    dilation = 1
    update_map = np.ones(window_size)
    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR
    out_dir = None

    _, outputs = morpheus_core.predict_arrays(
        model,
        model_inputs,
        n_classes,
        batch_size,
        window_size,
        dilation,
        stride,
        update_map,
        aggregate_method,
        out_dir,
    )
    out_lbl, out_n = outputs

    assert out_lbl.sum() == 100 * 100
    assert out_lbl.shape == (100, 100, 1, 2)
    np.testing.assert_array_equal(out_n[0, :10], np.arange(10) + 1)


@pytest.mark.integration
def test_predict_arrays_median():
    """Tests morpheus_core.predict_arrays with mean_var aggergation"""
    total_shape = (100, 100, 1)

    model = lambda x: np.ones_like(x[0])
    model_inputs = [np.ones(total_shape)]
    n_classes = 1
    batch_size = 10
    window_size = (10, 10)
    stride = (1, 1)
    dilation = 1
    update_map = np.ones(window_size)
    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEDIAN
    out_dir = None

    _, outputs = morpheus_core.predict_arrays(
        model,
        model_inputs,
        n_classes,
        batch_size,
        window_size,
        dilation,
        stride,
        update_map,
        aggregate_method,
        out_dir,
    )
    out_lbl, out_n = outputs

    assert out_lbl.sum() == 100 * 100
    assert out_lbl.shape == (100, 100, 1, 2)
    np.testing.assert_array_equal(out_n[0, :10], np.arange(10) + 1)



@pytest.mark.integration
def test_predict_arrays_mean_var_super_resolution():
    """Tests morpheus_core.predict_arrays with mean_var aggergation"""
    total_shape = (100, 100, 1)
    dilation = 3

    model = lambda x: np.ones(
        list(map(lambda x: x * dilation, x[0].shape[:-1])) + [x[0].shape[-1]]
    )
    model_inputs = [np.ones(total_shape)]
    n_classes = 1
    batch_size = 10
    window_size = (10, 10)
    stride = (1, 1)
    update_map = np.ones((30, 30))
    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR
    out_dir = None

    _, outputs = morpheus_core.predict_arrays(
        model,
        model_inputs,
        n_classes,
        batch_size,
        window_size,
        dilation,
        stride,
        update_map,
        aggregate_method,
        out_dir,
    )
    out_lbl, out_n = outputs

    assert out_lbl.sum() == 300 * 300
    assert out_lbl.shape == (300, 300, 1, 2)
    np.testing.assert_array_equal(
        out_n[0, :10], np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
    )


@pytest.mark.integration
def test_predict_arrays_rank_vote():
    """Tests morpheus_core.predict_arrays with rank_vote aggergation"""
    total_shape = (100, 100, 1)

    model = lambda x: np.ones_like(x[0])
    model_inputs = [np.ones(total_shape)]
    n_classes = 1
    batch_size = 10
    window_size = (10, 10)
    dilation = 1
    stride = (1, 1)
    update_map = np.ones(window_size)
    aggregate_method = morpheus_core.AGGREGATION_METHODS.RANK_VOTE
    out_dir = None

    _, outputs = morpheus_core.predict_arrays(
        model,
        model_inputs,
        n_classes,
        batch_size,
        window_size,
        dilation,
        stride,
        update_map,
        aggregate_method,
        out_dir,
    )
    out_lbl, out_n = outputs

    assert out_lbl.sum() == 100 * 100
    assert out_lbl.shape == (100, 100, 1)
    np.testing.assert_array_equal(out_n[0, :10], np.arange(10) + 1)


@pytest.mark.unit
def test_predict_arrays_invalid_aggregation_method():
    """Tests morpheus_core.predict_arrays with invalid choice"""
    total_shape = (100, 100, 1)

    model = lambda x: np.ones_like(x[0])
    model_inputs = [np.ones(total_shape)]
    n_classes = 1
    batch_size = 10
    window_size = (10, 10)
    dilation = 1
    stride = (1, 1)
    update_map = np.ones(window_size)
    aggregate_method = "invalid"
    out_dir = None

    with pytest.raises(ValueError):
        morpheus_core.predict_arrays(
            model,
            model_inputs,
            n_classes,
            batch_size,
            window_size,
            dilation,
            stride,
            update_map,
            aggregate_method,
            out_dir,
        )


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_predict_mean_var_on_disk():
    """Tests morpheus_core.predict with mean_var aggergation on disk"""
    helpers.setup()

    model = lambda x: np.ones_like(x[0])
    model_inputs = [helpers.make_mock_input()]
    n_classes = 1
    batch_size = 10
    window_size = (10, 10)
    dilation = 1
    stride = (1, 1)
    update_map = np.ones(window_size)
    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR
    out_dir = helpers.TMP_DIR
    gpus = None
    cpus = None
    parallel_check_interval = 1

    hduls, outputs = morpheus_core.predict(
        model,
        model_inputs,
        n_classes,
        batch_size,
        window_size,
        dilation,
        stride,
        update_map,
        aggregate_method,
        out_dir,
        gpus,
        cpus,
        parallel_check_interval,
    )
    out_lbl, out_n = outputs

    disk_lbl = fits.getdata(os.path.join(helpers.TMP_DIR, "output.fits"))
    disk_n = fits.getdata(os.path.join(helpers.TMP_DIR, "n.fits"))

    assert out_lbl.sum() == 100 * 100
    assert out_lbl.shape == (100, 100, 1, 2)
    np.testing.assert_array_equal(disk_lbl, out_lbl)
    np.testing.assert_array_equal(disk_n, out_n)
    np.testing.assert_array_equal(out_n[0, :10], np.arange(10) + 1)

    misc.apply(lambda x: x.close(), hduls)

    helpers.tear_down()


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_predict_mean_var_in_mem():
    """Tests morpheus_core.predict with mean_var aggergation in mem"""
    total_shape = (100, 100, 1)

    model = lambda x: np.ones_like(x[0])
    model_inputs = [np.ones(total_shape)]
    n_classes = 1
    batch_size = 10
    window_size = (10, 10)
    dilation = 1
    stride = (1, 1)
    update_map = np.ones(window_size)
    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR
    out_dir = None
    gpus = None
    cpus = None
    parallel_check_interval = 1

    _, outputs = morpheus_core.predict(
        model,
        model_inputs,
        n_classes,
        batch_size,
        window_size,
        dilation,
        stride,
        update_map,
        aggregate_method,
        out_dir,
        gpus,
        cpus,
        parallel_check_interval,
    )
    out_lbl, out_n = outputs

    assert out_lbl.sum() == 100 * 100
    assert out_lbl.shape == (100, 100, 1, 2)
    np.testing.assert_array_equal(out_n[0, :10], np.arange(10) + 1)


if __name__ == "__main__":
    test_predict_arrays_mean_var_super_resolution()
