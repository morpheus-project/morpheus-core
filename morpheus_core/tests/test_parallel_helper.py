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
import time
from astropy.io.fits.convenience import writeto
from astropy.io.fits.hdu.image import PrimaryHDU

import numpy as np
from numpy.lib.stride_tricks import as_strided
import pytest
from astropy.io import fits

import morpheus_core.helpers.parallel_helper as ph
import morpheus_core.morpheus_core as morpheus_core
import morpheus_core.tests.helpers as helper
from morpheus_core.tests.helpers import TMP_DIR


@pytest.mark.unit
def test_get_split_length():
    """Tests morpheus_core.helpers.parallel_helper.get_split_length"""

    shape = (1200, 1200)
    num_workers = 4
    window_shape = (40, 40)

    expected_length = 330
    actual_length = ph.get_split_length(shape, num_workers, window_shape)

    assert expected_length == actual_length


@pytest.mark.unit
def test_get_split_slice_generator():
    """Tests morpheus_core.helpers.parallel_helper.get_spit_slice_generator"""

    shape = (1200, 1200)
    window_shape = (40, 40)
    num_workers = 4
    split_length = 330

    expected_slices = [
        slice(0, 330),
        slice(291, 621),
        slice(582, 912),
        slice(873, 1200),
    ]

    actual_slices = list(
        ph.get_split_slice_generator(shape, window_shape, num_workers, split_length)
    )

    print(actual_slices)

    assert expected_slices == actual_slices


@pytest.mark.unit
def test_make_runnable_file():
    """Tests morpheus_core.helpers.parallel_helper.make_runnable_file"""

    helper.setup()

    path = helper.TMP_DIR
    input_file_names = [helper.make_sample_file()]
    n_classes = 2
    batch_size = 10
    window_size = (10, 10)
    stride = (1, 1)
    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR

    local = os.path.dirname(os.path.dirname(__file__))
    expected_text = [
        "import sys",
        f"sys.path.append('{local}')",
        "import os",
        "import dill",
        "import numpy as np",
        "from tqdm import tqdm",
        "from morpheus_core import morpheus_core",
        "def main():",
        "    output_dir = './output'",
        "    if 'output' not in os.listdir():",
        "        os.mkdir('./output')",
        "",
        "    with open('model.pkl', 'rb') as f:",
        "        model = dill.load(f)",
        "",
        "    model_inputs = [",
        "        " + ",".join(["'" + i + "'" for i in input_file_names]),
        "    ]",
        "",
        "    update_map = np.load('update_map.npy', allow_pickle=True)",
        "",
        "    morpheus_core.predict(",
        "        model,",
        "        model_inputs,",
        f"       {n_classes},",
        f"       {batch_size},",
        f"       {window_size},",
        f"       stride={stride},",
        "        update_map=update_map,",
        f"       aggregate_method='{aggregate_method}',",
        "        out_dir=output_dir,",
        "    )",
        "    sys.exit(0)",
        "if __name__=='__main__':",
        "    main()",
    ]

    ph.make_runnable_file(
        path,
        input_file_names,
        n_classes,
        batch_size,
        window_size,
        stride,
        aggregate_method,
    )

    with open(os.path.join(path, "main.py"), "r") as f:
        actual_text = [l.rstrip() for l in f.readlines()]

    assert expected_text == actual_text

    helper.tear_down()


def pickle_rick(x):
    """A pickleable function to use a mock model"""
    return None


@pytest.mark.integration
def test_build_parallel_classification_structure():
    """Tests morpheus_core.helpers.parallel_helper.build_parallel_classification_structure"""
    helper.setup()

    model = pickle_rick
    arr_fnames = [helper.make_mock_input()]
    arrs = [fits.getdata(arr_fnames[0])]
    n_classes = 2
    batch_size = 10
    window_size = (10, 10)
    stride = (1, 1)
    update_map = np.ones(window_size)
    aggregate_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR
    out_dir = helper.TMP_DIR
    workers = [0, 1]

    ph.build_parallel_classification_structure(
        model,
        arrs,
        arr_fnames,
        n_classes,
        batch_size,
        window_size,
        stride,
        update_map,
        aggregate_method,
        out_dir,
        workers,
    )

    expected_parent_tree = set(["input.fits", "0", "1"])
    expected_sub_tree = set(["input.fits", "main.py", "model.pkl", "update_map.npy"])

    assert expected_parent_tree == set(os.listdir(out_dir))
    assert expected_sub_tree == set(os.listdir(os.path.join(out_dir, "0")))
    assert expected_sub_tree == set(os.listdir(os.path.join(out_dir, "1")))

    helper.tear_down()


@pytest.mark.unit
def test_worker_to_cmd_gpu():
    """Tests morpheus_core.helpers.parallel_helper.worker_to_cmd for a gpu worker"""
    is_gpu = True
    worker = 1

    expected_string = f"CUDA_VISIBLE_DEVICES={worker} python main.py"

    actual_string = ph.worker_to_cmd(is_gpu, worker)

    assert expected_string == actual_string


@pytest.mark.unit
def test_worker_to_cmd_cpu():
    """Tests morpheus_core.helpers.parallel_helper.worker_to_cmd for a cpu worker"""
    is_gpu = False
    worker = 1

    expected_string = "CUDA_VISIBLE_DEVICES=-1 python main.py"

    actual_string = ph.worker_to_cmd(is_gpu, worker)

    assert expected_string == actual_string


@pytest.mark.unit
def test_check_procs():
    """Tests morpheus_core.helpers.parallel_helper.check_procs"""

    class MockProcess(object):
        def __init__(self, poll_value):
            self.poll_value = poll_value

        def poll(self):
            return self.poll_value

    procs = {1: MockProcess(1), 2: MockProcess(1), 3: MockProcess(None)}

    expected_num_running_procs = 1
    actual_num_running_procs = sum(ph.check_procs(procs))

    assert expected_num_running_procs == actual_num_running_procs


@pytest.mark.integration
def test_monitor_procs():
    """Tests morpheus_core.helpers.parallel_helper.monitor_procs"""

    class MockProcess(object):
        def __init__(self, counter_to_complete):
            self.counter_to_complete = counter_to_complete

        def poll(self):
            self.counter_to_complete -= 1
            return 0 if self.counter_to_complete <= 0 else None

    parallel_check_interval = 0.5

    procs = {1: MockProcess(2), 2: MockProcess(3)}

    start_time = time.time()
    ph.monitor_procs(procs, parallel_check_interval)
    end_time = time.time()

    expected_runtime = len(procs) * parallel_check_interval
    actual_runtime = end_time - start_time

    # we expected that the mock process dictionary to be queried at least
    # twice becuase of how the mock processes were instantiated, so we
    # can verify that the core was called to sleep at least that long by
    # seeing how much time has past
    assert actual_runtime >= expected_runtime


# TODO: This test feels hacky and will likely have issues depending on cpu,
#      write a test that detects running subprocess
@pytest.mark.flaky(reruns=5)
@pytest.mark.integration
def test_run_parallel_jobs():
    """Tests morpheus_core.helpers.parallel_helper.run_parallel_jobs"""

    helper.setup()

    workers = ["0", "1"]
    for w in workers:
        os.mkdir(os.path.join(helper.TMP_DIR, w))
        with open(os.path.join(helper.TMP_DIR, w, "main.py"), "w") as f:
            f.write(
                "\n".join(
                    [
                        "import time",
                        "def main():",
                        "    time.sleep(0.15)",
                        "",
                        "if __name__=='__main__':",
                        "    main()",
                    ]
                )
            )

    is_gpu = False
    out_dir = helper.TMP_DIR
    parallel_check_interval = 0.2

    start_time = time.time()
    ph.run_parallel_jobs(workers, is_gpu, out_dir, parallel_check_interval)
    end_time = time.time()

    run_time = end_time - start_time
    expected_max_time = 0.6
    # if the process runs in parallel then they will both be done before the
    # first check interval
    assert run_time < expected_max_time

    helper.tear_down()


@pytest.mark.unit
def test_get_merge_function_mean_var():
    """Tests morpheus_core.helpers.parallel_helper.get_merge_function"""
    method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR
    assert ph.get_merge_function(method) == ph.merge_parallel_mean_var


@pytest.mark.unit
def test_get_merge_function_rank_vote():
    """Tests morpheus_core.helpers.parallel_helper.get_merge_function"""
    method = morpheus_core.AGGREGATION_METHODS.RANK_VOTE
    assert ph.get_merge_function(method) == ph.merge_parallel_rank_vote


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_get_empty_output_array_rank_vote():
    """Tests morpheus_core.helpers.parallel_helper.get_empty_output_array for mean_var"""
    helper.setup()
    out_dir = helper.TMP_DIR
    method = morpheus_core.AGGREGATION_METHODS.RANK_VOTE

    in_shape = (100, 100, 5)
    expected_output_shape = (100, 100, 5)
    expected_n_shape = (100, 100)

    out_hdul, n_hdul, actual_output, actual_n = ph.get_empty_output_array(
        out_dir, *in_shape, method
    )

    assert actual_output.shape == expected_output_shape
    assert actual_n.shape == expected_n_shape

    out_hdul.close()
    n_hdul.close()

    helper.tear_down()


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_get_empty_output_array_mean_var():
    """Tests morpheus_core.helpers.parallel_helper.get_empty_output_array for mean_var"""
    helper.setup()
    out_dir = helper.TMP_DIR
    method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR

    in_shape = (100, 100, 5)
    expected_output_shape = (100, 100, 5, 2)
    expected_n_shape = (100, 100)

    out_hdul, n_hdul, actual_output, actual_n = ph.get_empty_output_array(
        out_dir, *in_shape, method
    )

    assert actual_output.shape == expected_output_shape
    assert actual_n.shape == expected_n_shape

    out_hdul.close()
    n_hdul.close()

    helper.tear_down()


@pytest.mark.unit
def test_get_start_y_idxs():
    """Tests morpheus_core.helpers.parallel_helper.get_start_y_idxs"""
    n_heights = [100, 100, 100]
    window_height = 10

    expected_idxs = np.array([0, 91, 182, 273])

    actual_idxs = ph.get_start_y_idxs(n_heights, window_height)

    np.testing.assert_array_equal(actual_idxs, expected_idxs)


@pytest.mark.unit
def test_get_data_from_worker():
    """Tests morpheus_core.helpers.parallel_helper.get_data_from_worker"""
    helper.setup()

    out_dir = helper.TMP_DIR
    worker = "1"

    os.mkdir(os.path.join(out_dir, worker))
    os.mkdir(os.path.join(out_dir, worker, "output"))

    output_path = os.path.join(out_dir, worker, "output", "output.fits")

    n_path = os.path.join(out_dir, worker, "output", "n.fits")

    mock_data = np.zeros([100, 100], dtype=np.float32)
    fits.PrimaryHDU(data=mock_data).writeto(output_path)
    fits.PrimaryHDU(data=mock_data).writeto(n_path)

    actual_output, actual_n = ph.get_data_from_worker(out_dir, worker)

    np.testing.assert_array_equal(mock_data, actual_output)
    np.testing.assert_array_equal(mock_data, actual_n)

    helper.tear_down()


@pytest.mark.unit
def test_merge_parallel_mean_var():
    """Tests morpheus_core.helpers.parallel_helper.merge_parallel_mean_var"""

    test_shape = [10, 10, 1, 2]
    combined_out = np.zeros(test_shape)
    combined_out[:5, ..., 0] = 1
    combined_n = np.zeros([10, 10])
    combined_n[:5, :] = 1
    combined_n[5:, :] = 0

    output = np.zeros([5, 10, 1, 2])
    output[..., 0] = 2
    output[..., 1] = 0
    n = np.ones([5, 10])

    start_y = 4

    ph.merge_parallel_mean_var(combined_out, combined_n, output, n, start_y)

    expected_out = np.zeros(test_shape)
    expected_out[:4, :, 0, 0] = 1
    expected_out[4, :, 0, 0] = 1.5
    expected_out[5:-1, :, 0, 0] = 2

    expected_out[:4, :, 0, 1] = 0
    expected_out[4, :, 0, 1] = 0.25
    expected_out[5:, :, 0, 1] = 0

    np.testing.assert_array_equal(expected_out, combined_out)


@pytest.mark.unit
def test_mege_parallel_rank_vote():
    """tests morpheus_core.helpers.parallel_helper.merge_parallel_rank_vote"""

    test_shape = [10, 10, 2]
    combined_out = np.zeros(test_shape)
    combined_out[:5, :, 0] = 0.3
    combined_out[:5, :, 1] = 0.7
    combined_n = np.zeros([10, 10])
    combined_n[:5, :] = 1
    combined_n[5:, :] = 0

    output = np.zeros([5, 10, 2])
    output[..., 0] = 0.5
    output[..., 1] = 0.5
    n = np.ones([5, 10])

    start_y = 4

    ph.merge_parallel_rank_vote(combined_out, combined_n, output, n, start_y)

    expected_out = np.zeros(test_shape)
    expected_out[:4, :, 0] = 0.3
    expected_out[:4, :, 1] = 0.7

    expected_out[4, :, 0] = 0.4
    expected_out[4, :, 1] = 0.6

    expected_out[5:-1, :, 0] = 0.5
    expected_out[5:-1, :, 1] = 0.5

    np.testing.assert_array_equal(expected_out, combined_out)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore astropy warning
def test_stitch_parallel_classifications():
    """tests morpheus_core.helpers.parallel_helper.stitch_parallel_classifications"""

    helper.setup()
    out_dir = helper.TMP_DIR
    test_shape = [10, 10, 1, 2]
    window_shape = [5, 5]
    workers = [0, 1]
    aggregation_method = morpheus_core.AGGREGATION_METHODS.MEAN_VAR

    for w in workers:
        os.mkdir(os.path.join(out_dir, str(w)))
        os.mkdir(os.path.join(out_dir, str(w), "output"))
        test_data = np.zeros(test_shape)
        test_data[:, :, 0, 0] = 1 + w

        fits.PrimaryHDU(data=test_data).writeto(
            os.path.join(TMP_DIR, str(w), "output", "output.fits")
        )

        fits.PrimaryHDU(data=np.ones(test_shape[:2])).writeto(
            os.path.join(TMP_DIR, str(w), "output", "n.fits")
        )

    hduls, outputs = ph.stitch_parallel_classifications(
        workers, out_dir, aggregation_method, window_shape
    )

    actual_output, actual_n = outputs

    expected_n = np.ones([16, 10])
    expected_n[6:10, :] = 2

    expected_output = np.zeros([16, 10, 1, 2])

    expected_output[:6, :, 0, 0] = 1
    expected_output[6:10, :, 0, 0] = 1.5
    expected_output[10:, :, 0, 0] = 2

    expected_output[:6, :, 0, 1] = 0
    expected_output[6:10, :, 0, 1] = 0.25
    expected_output[10:, :, 0, 1] = 0

    np.testing.assert_array_equal(actual_n, expected_n)
    np.testing.assert_array_equal(actual_output, expected_output)

    list(map(lambda h: h.close(), hduls))

    helper.tear_down()
