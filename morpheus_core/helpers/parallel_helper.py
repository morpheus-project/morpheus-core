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
import os
from os.path import split
import pickle
import shutil
import time
from itertools import repeat, starmap
from subprocess import Popen
from typing import Callable, Dict, Iterable, List, Tuple, Union

import dill
import numpy as np
from astropy.io import fits
from tqdm import tqdm

import morpheus_core.helpers.misc_helper as mh
import morpheus_core.helpers.fits_helper as fh
from morpheus_core import morpheus_core


def get_split_length(
    shape: List[int], num_workers: int, window_shape: Tuple[int]
) -> int:
    """Calculate the size of the sub images for classification.

    Args:
        shape (List[int]): the shape of the array to be split
        num_workers (int): the number of splits to make
        window_shape (Tuple[int]): The (height, width) tuple describing the size
                                   of the sliding window.


    Returns:
        The length of each split along axis 0

    TODO: Implement splits along other axes
    """

    return (shape[0] + (num_workers - 1) * window_shape[0]) // num_workers


def get_split_slice_generator(
    shape: Tuple[int], window_shape: Tuple[int], num_workers: int, split_length: int
) -> Iterable[slice]:
    """Creates a generator that yields `slice` objects to split imgs.

    Args:
        shape (Tuple[int]): The shape of the array to be split
        window_shape (Tuple[int]): The (height, width) tuple describing the size
                                   of the sliding window.
        num_workers (int): The number of splits to make
        split_length (int): The length each slice should be

    Returns
        A generator that yields slice objects

    TODO: Implement splits along other axes
    TODO: Refactor to a more functional implementation
    """

    start_ys = get_start_y_idxs(
        list(repeat(split_length, num_workers - 1)), window_height=window_shape[0]
    )

    end_ys = start_ys + split_length
    end_ys[-1] = shape[0]

    return starmap(slice, zip(start_ys, end_ys))

    # idx = 0
    # for i in range(num_workers):
    #     start_idx = max(idx - window_shape[0] - 1, 0)

    #     if i == num_workers - 1:
    #         end_idx = shape[0]
    #     else:
    #         end_idx = start_idx + split_length - 1

    #     idx = end_idx

    #     yield slice(start_idx, end_idx)


def make_runnable_file(
    path: str,
    input_fnames: List[str],
    n_classes: int,
    batch_size: int,
    window_size: Union[Tuple[int], List[int]],
    dilation: int,
    stride: Union[Tuple[int], List[int]],
    aggregate_method: str,
) -> None:
    """Creates a file at `path` that classfies local FITS files.

    Args:
        path (str): The dir to save the file in
        input_fnames (List[str]): The list of file names that contain the
                                  arrays to convert into batches and serve to
                                  the model
        n_classes (int): The number of classes that the models predicts for
        batch_size (int): The batch size for the model to use when classifying
                            the input
        window_size (Union[Tuple[int], List[int]]): The (h, w) of each example
                                                    in a batch
        stride (Union[Tuple[int], List[int]]): The stride size of the sliding
                                               window
        aggregate_method (str): how to process the output from the model. If
                                AGGREGATION_METHODS.MEAN_VAR record output using
                                mean and variance, If AGGREGATION_METHODS.RANK_VOTE
                                record output as the normalized vote count.

    Returns:
        None
    """

    # we need `local` so that we can import morpheus_core just in case the pip env
    # doesn't carry over to the new process
    local = os.path.dirname(os.path.dirname(__file__))
    text = [
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
        "        " + ",".join(["'" + i + "'" for i in input_fnames]),
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
        f"       {dilation},",
        f"       stride={stride},",
        "        update_map=update_map,",
        f"       aggregate_method='{aggregate_method}',",
        "        out_dir=output_dir,",
        "    )",
        "    sys.exit(0)",
        "if __name__=='__main__':",
        "    main()",
    ]

    with open(os.path.join(path, "main.py"), "w") as f:
        f.write("\n".join(text))


def build_parallel_classification_structure(
    model: Callable,
    arrs: List[np.ndarray],
    arr_fnames: List[str],
    n_classes: int,
    batch_size: int,
    window_shape: Tuple[int],
    dilation: int,
    stride: Union[Tuple[int], List[int]],
    update_map: np.ndarray,
    aggregate_method: str,
    out_dir: str,
    workers: List[int],
) -> None:
    """Sets up the subdirs and files to run the parallel classification.

    Args:
        arrs (List[np.ndarray]): List of arrays to split up in the order HJVZ
        arr_fnames (List[str]): The file names that hold the input arrays
                                `arrs`
        workers (List[int]): A list of worker ID's that can either be CUDA GPU
                             ID's or a list dummy numbers for cpu workers
        batch_size (int): The batch size for Morpheus to use when classifying
                          the input.
        window_shape (Tuple[int]): The (height, width) tuple describing the size
                                   of the sliding window.
        out_dir (str): the location to place the subdirs in

    Returns:
        None

    TODO: Refactor to a more functional implementation
    """

    shape = arrs[0].shape
    num_workers = len(workers)
    split_slices = get_split_slice_generator(
        shape,
        window_shape,
        num_workers,
        get_split_length(shape, num_workers, window_shape),
    )

    for worker, split_slice in tqdm(zip(sorted(workers), split_slices)):
        sub_output_dir = os.path.join(out_dir, str(worker))
        os.mkdir(sub_output_dir)

        # put sliced input files into subdir
        for name, data in zip(arr_fnames, arrs):
            tmp_location = os.path.join(sub_output_dir, os.path.split(name)[1])
            fits.PrimaryHDU(data=data[split_slice, ...]).writeto(tmp_location)

        # put model into subdir
        with open(os.path.join(sub_output_dir, "model.pkl"), "wb") as f:
            dill.dump(model, f)

        # put udpate_map into subdir

        if update_map is None:
            update_map = np.ones(window_shape)

        np.save(os.path.join(sub_output_dir, "update_map.npy"), update_map)

        make_runnable_file(
            sub_output_dir,
            arr_fnames,
            n_classes,
            batch_size,
            window_shape,
            dilation,
            stride,
            aggregate_method,
        )


def worker_to_cmd(is_gpu: bool, worker: int) -> str:
    """Returns a the bash command to run a worker job.

    Args:
        is_gpu (bool): True if worker is a gpu worker false if cpu worker
        worker (int): The worker id, this is the GPU id for gpu workers

    Returns:
        A string containing the bash command to run a worker job.
    """

    if is_gpu:
        return f"CUDA_VISIBLE_DEVICES={worker} python main.py"
    else:
        return f"CUDA_VISIBLE_DEVICES=-1 python main.py"


def check_procs(procs: Dict[int, Popen]) -> List[bool]:
    """Checks on the status of running jobs.

    Args:
        procs (Dict[int, Popen]): A dictionary where the keys are the worker
                                  ids and the values are the process objects
    Returns:
        A list of booleans indicating if the processes are finished.
    """
    return list(
        map(
            # if poll() returns None the process is still running
            lambda p: procs[p].poll() == None,
            procs,
        )
    )


def monitor_procs(procs: Dict[int, Popen], parallel_check_interval: int) -> None:
    """Monitors the progress of running subprocesses.

    Args:
        procs (Dict[int, Popen]): A dictionary where the keys are the worker ids
                                  and the values are the process objects
        parrallel_check_interval (int): An integer
    """

    wait_f = lambda: not bool(time.sleep(parallel_check_interval))

    all(
        map(
            # if there any running processes, then time.sleep will get called
            # which always returns None and therefore false which is negated
            # to continue the loop
            #
            # if there are no running processes, then the conditional is
            # shortcutted and the expression returns false ending the loop
            lambda running_procs: any(running_procs) and wait_f(),
            map(check_procs, repeat(procs)),
        )
    )


def run_parallel_jobs(
    workers: List[int], is_gpu: bool, out_dir: str, parallel_check_interval: float
) -> None:
    """Starts and tracks parallel job runs.

    WARNING: This will not finish running until all subprocesses are complete

    Args:
        workers (List[int]): A list of worker ID's to assign to a portion of an
                             image.
        is_gpu (bool): if True the worker ID's belong to NVIDIA GPUs and will
                       be used as an argument in CUDA_VISIBLE_DEVICES. If False,
                       then the ID's are assocaited with CPU workers
        out_dir (str): the location with the partitioned data
        parallel_check_interval (float): If gpus are given, then this is the
                                         number of minutes to wait between
                                         polling each subprocess for
                                         completetion.

    Returns:
        None
    """

    proc_cmds = [worker_to_cmd(is_gpu, w) for w in workers]
    subdirs = [os.path.join(out_dir, str(w)) for w in workers]

    processes = {
        w: Popen(p, shell=True, cwd=s) for w, p, s in zip(workers, proc_cmds, subdirs)
    }

    monitor_procs(processes, parallel_check_interval)


def merge_parallel_mean_var(
    combined_out: np.ndarray,
    combined_n: np.ndarray,
    output: np.ndarray,
    n: np.ndarray,
    start_y: int,
) -> None:
    """Merge the output from a worker into the total output for mean/var.

    Derived from:
    https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html

    Args:
        combined_out (np.ndarray): The total output array
        combined_n (np.ndarray): The total n array
        output (np.ndarray): The output to merge into the total output
        n (np.ndarray): The n to merge into the total n
        start_y (int): The y index to merge into output into combined_out

    Returns:
        None, the operation is performed inplace on combined_out and combined_n
    """

    ys = slice(start_y, start_y + output.shape[0])

    x1, x2 = combined_out[ys, ..., 0].copy(), output[..., 0].copy()
    s1, s2 = combined_out[ys, ..., 1].copy(), output[..., 1].copy()
    n1, n2 = combined_n[ys, :, np.newaxis].copy(), n[..., np.newaxis].copy()

    denominator = n1 + n2
    safe_divide_mask = denominator > 0

    xc_numerator = (n1 * x1) + (n2 * x2)
    xc = np.where(safe_divide_mask, xc_numerator / denominator, 0)

    sc_numerator = (n1 * (s1 + (x1 - xc) ** 2)) + (n2 * (s2 + (x2 - xc) ** 2))
    sc = np.where(safe_divide_mask, sc_numerator / denominator, 0)

    combined_out[ys, ..., 0] = xc
    combined_out[ys, ..., 1] = sc
    combined_n[ys, ...] = denominator[..., 0]


def merge_parallel_rank_vote(
    combined_out: np.ndarray,
    combined_n: np.ndarray,
    output: np.ndarray,
    n: np.ndarray,
    start_y: int,
) -> None:
    """Merge the output from a worker into the total output for rank vote.

    Args:
        combined_out (np.ndarray): The total output array
        combined_n (np.ndarray): The total n array
        output (np.ndarray): The output to merge into the total output
        n (np.ndarray): The n to merge into the total n
        start_y (int): The y index to merge into output into combined_out

    Returns:
        None, the operation is performed inplace on combined_out and combined_n
    """

    ys = slice(start_y, start_y + output.shape[0])

    x1, x2 = combined_out[ys, ...].copy(), output.copy()
    n1, n2 = combined_n[ys, :, np.newaxis].copy(), n[..., np.newaxis].copy()

    numerator = (n1 * x1) + (n2 * x2)
    denominator = n1 + n2
    mean = np.where(denominator > 0, numerator / denominator, 0)

    combined_out[ys, :] = mean
    combined_n[ys, :] = denominator[..., 0]


def get_merge_function(aggreation_method: str) -> Callable:
    """Returns the method for merging arrays based on the aggregation method.

    Args:
        aggregation_method (str): The aggregation method used one of
                                  morpheus_core.AGGREGATION_METHODS.MEAN_VAR or
                                  morpheus_core.AGGREGATION_METHODS.RANK_VOTE

    Returns:
        A function the use for merging output arrays

    """

    if aggreation_method == morpheus_core.AGGREGATION_METHODS.MEAN_VAR:
        return merge_parallel_mean_var
    else:
        return merge_parallel_rank_vote


def get_data_from_worker(out_dir: str, worker: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the n array and the output classifications for a given worker

    Args:
        out_dir (str): The directory where the workers are storing their ouptut
        worker (int): The worker id to get the data for

    Returns:
        A 2-Tuple where the first element is the worker output array and the
        second element is n array.
    """
    return (
        fits.getdata(os.path.join(out_dir, str(worker), "output", "output.fits")),
        fits.getdata(os.path.join(out_dir, str(worker), "output", "n.fits")),
    )


def get_empty_output_array(
    out_dir: str, height: int, width: int, n_classes: int, aggregation_method: str
) -> np.ndarray:
    """Creates an empty array in the output dir and returns a memmapped array for it

    Args:
        out_dir (str): The output directory to store the array in
        height (int): The output image height
        width (int): The output image width
        n_classes (int): The number classes the model predicts
        aggregation_method (str): The method to use for merging outputs one of
                                  morpheus_core.AGGREGATION_METHODS.MEAN_VAR or
                                  morpheus_core.AGGREGATION_METHODS.RANK_VOTE

    Returns:
        A 4-Tuple where the first element is the HDUL for output array, the
        second element is the HDUL for the n array, the third element is the
        output array, the fourth element is the n array.

    """
    if aggregation_method == morpheus_core.AGGREGATION_METHODS.MEAN_VAR:
        shape = [height, width, n_classes, 2]
    else:
        shape = [height, width, n_classes]

    out_file = os.path.join(out_dir, "output.fits")
    n_file = os.path.join(out_dir, "n.fits")

    fh.create_file(out_file, shape, dtype=np.float32)
    fh.create_file(n_file, shape[:2], dtype=np.float32)

    out_hdul, out_array = fh.open_file(out_file, mode="update")
    n_hdul, n_array = fh.open_file(n_file, mode="update")

    return (out_hdul, n_hdul, out_array, n_array)


def get_start_y_idxs(n_heights: List[int], window_height: int) -> List[int]:
    """Gets the y indexes to crop and merge arrays with.

    Args:
        n_heights (List[int]): The heights of the cropped arrays
        window_height (int): The height of the a single input/output from the
                             model

    Returns:
        The y index values to use for merging the arrays.
    """
    offset = window_height - 1
    return np.cumsum([0] + list(map(lambda y: y - offset, n_heights)))


def stitch_parallel_classifications(
    workers: List[int], out_dir: str, aggregation_method: str, window_shape: Tuple[int]
) -> Tuple[List[fits.HDUList], List[np.ndarray]]:
    """Merges all of the output from the workers into a single classification image.

    Args:
        workers (List[int]): List of integer ids associated with workers
        out_dir (str): The output directory that the worker classifications are
                       stored in.
        aggregation (str): The morpheus_core.AGGREGATION_METHODS value to use to merge
                           the output arrays
        window_shape (Tuple[int, int]): The (width, height) of the input output
                                        image data.

    Returns:
        A 2-Tuple, where the first element is a list of HDULs for the merged data
        and the second element is the merged arrays.
    """

    data_f = partial(get_data_from_worker, out_dir)
    outs, ns = list(zip(*map(data_f, workers)))

    total_y = sum(map(lambda x: x.shape[0], ns))
    offset_y = (window_shape[0] - 1) * (len(ns) - 1)
    new_y = total_y - offset_y
    new_x = outs[0].shape[1]
    n_classes = outs[0].shape[2]

    out_hdul, n_hdul, combined_out, combined_n = get_empty_output_array(
        out_dir, new_y, new_x, n_classes, aggregation_method
    )
    merge_f = partial(get_merge_function(aggregation_method), combined_out, combined_n)

    start_ys = get_start_y_idxs(list(map(lambda x: x.shape[0], ns)), window_shape[0])

    mh.apply(merge_f, zip(outs, ns, start_ys))
    mh.apply(lambda h: h.close(), [out_hdul, n_hdul])

    clean_up = lambda w: shutil.rmtree(os.path.join(out_dir, str(w)))
    mh.apply(clean_up, workers)

    return ([out_hdul, n_hdul], [combined_out, combined_n])
