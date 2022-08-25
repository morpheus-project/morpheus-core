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
from functools import partial
from itertools import islice, repeat, starmap, takewhile
from typing import Callable, List, Tuple, Union

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from morpheus_core.helpers import misc_helper
from morpheus_core.helpers import fits_helper
from morpheus_core.helpers import label_helper
from morpheus_core.helpers import parallel_helper

__all__ = ["AGGREGATION_METHODS", "predict"]


class AGGREGATION_METHODS:
    """Helper class with string constants to use as arguments in morpheus_core methods."""

    MEAN_VAR = "mean_var"
    RANK_VOTE = "rank_vote"

    INVALID_ERR = " ".join(
        [
            "Invalid aggregation method please select one of",
            "AGGREGATION_METHODS.MEAN_VAR or AGGREGATION_METHODS.RANK_VOTE",
        ]
    )


def build_batch(
    arr: List[np.ndarray],
    window_size: Tuple[int, int],
    batch_idxs: List[Tuple[int, int]],
):
    """Builds a batch of samples of `window_size` from `arr` at `batch_idxs`.

    Args:
        arr (List[np.ndarray]): array(s) to extract values from
        window_size (Tuple[int, int]): (height, width) of batch samples
        batch_idxs (List[Tuple[int, int]]): List of (y,x) locations to sample

    Returns:
        Returns a 2-Tuple where the first element is the batch and the second
        element is the list of batch idxs
    """

    def grab_slice(in_array, dim0, dim1):
        return in_array[dim0 : dim0 + window_size[0], dim1 : dim1 + window_size[1], ...]

    def grab_batch(in_array: np.ndarray):
        return np.array([grab_slice(in_array, dim0, dim1) for dim0, dim1 in batch_idxs])

    batches = list(map(grab_batch, arr))

    return batches, batch_idxs


def predict_batch(
    model_f: Callable, batch: List[np.ndarray], batch_idxs: List[Tuple[int, int]]
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Calls the model function on a batch.

    Args:
        model_f (Callable): Model function that predicts on a batch
        batch (List[np.ndarray]): batch values
        batch_idxs (List[Tuple[int, int]]): (y,x) locations for batch values

    Returns:
        A 2-Tuple where the first element is the output of model on the given
        batch and the second element is the batch indexes associated with the
        output.
    """
    return model_f(batch), batch_idxs


def update_output(
    aggregate_method: str,
    update_map: np.ndarray,
    stride: Tuple[int, int],
    dilation: float,
    n: np.ndarray,
    outputs: np.ndarray,
    batch_out: np.ndarray,
    batch_idx: Tuple[int, int],
) -> None:
    """Updates the total output with a single output value

    Args:
        aggregate_method (str): How to process the output from the model. If
                                AGGREGATION_METHODS.MEAN_VAR record output using
                                mean and variance, If AGGREGATION_METHODS.RANK_VOTE
                                record output as the normalized vote count.
        update_map (np.ndarray): A boolean mask that indicates what pixels in
                                 in each example to update
        stride (Tuple[int, int]): How many (rows, columns) to move through the
                                  image at each iteration.
        n (np.ndarray): The array containing the n values
        outputs (np.ndarray): The array containing the aggregated output values
        batch_out (np.ndarray): The output from the model to incorporate into
                                outputs
        batch_idx (Tuple[int, int]): The (y,x) location in the larger image that
                                     the batch_out should be incorporated into

    Returns:
        None

    Raises:
        ValueError if aggregate_method is not one of AGGREGATION_METHODS.MEAN_VAR
        or AGGREGATION_METHODS.RANK_VOTE
    """

    dialted_batch_idx = tuple(map(lambda x: int(dilation * x), batch_idx))

    if aggregate_method == AGGREGATION_METHODS.MEAN_VAR:
        label_helper.update_mean_var(
            update_map, stride, n, outputs, batch_out, dialted_batch_idx
        )
    elif aggregate_method == AGGREGATION_METHODS.RANK_VOTE:
        label_helper.update_rank_vote(
            update_map, stride, n, outputs, batch_out, dialted_batch_idx
        )
    else:
        raise ValueError(AGGREGATION_METHODS.INVALID_ERR)


def udpate_batch(
    aggregate_method: str,
    update_map: np.ndarray,
    stride: Tuple[int, int],
    dilation: float,
    n: np.ndarray,
    outputs: np.ndarray,
    batch_out: np.ndarray,  # [n, w, h, c]
    batch_idxs: List[Tuple[int, int]],  # [n, 2]
) -> None:
    """Updates the total output with the batch output values

    Args:
        aggregate_method (str): How to process the output from the model. If
                                AGGREGATION_METHODS.MEAN_VAR record output using
                                mean and variance, If AGGREGATION_METHODS.RANK_VOTE
                                record output as the normalized vote count.
        update_map (np.ndarray): A boolean mask that indicates what pixels in
                                 in each example to update
        stride (Tuple[int, int]): How many (rows, columns) to move through the
                                  image at each iteration.
        n (np.ndarray): The array containing the n values
        outputs (np.ndarray): The array containing the aggregated output values
        batch_out (np.ndarray): The output from the model to incorporate into
                                outputs
        batch_idx (List[Tuple[int, int]]): A list of (y,x) locations for each of
                                           the output array in `outputs`

    Returns:
        None
    """

    update_f = partial(
        update_output, aggregate_method, update_map, stride, dilation, n, outputs,
    )

    misc_helper.apply(update_f, zip(batch_out, batch_idxs))


def predict_arrays(
    model: Callable,
    model_inputs: List[np.ndarray],
    n_classes: int,
    batch_size: int,
    window_shape: Tuple[int, int],
    dilation: float = 1,
    stride: Tuple[int, int] = (1, 1),
    update_map: np.ndarray = None,
    aggregate_method: str = AGGREGATION_METHODS.RANK_VOTE,
    out_dir: str = None,
) -> Tuple[List[fits.HDUList], List[np.ndarray]]:
    """Uses applies the given model on the given inputs and returns the output.

    Args:
        model (Callable): The model to apply the the inputs
        model_inputs (List[np.ndarray]): The input arrays to a the model as a list
        n_classes (int): The number of output classes
        batch_size (int): The number of examples to include in each batch
        window_shape (int): The (height, width) of the samples to extract
        stride (Tuple[int, int]): How many (rows, columns) to move through the
                                  image at each iteration.
        update_map (np.narray): A 2D array of the same size as window height that
                                indicates which pixels to use to updates for each
                                example
        aggregate_method (str): How to process the output from the model. If
                                AGGREGATION_METHODS.MEAN_VAR record output using
                                mean and variance, If AGGREGATION_METHODS.RANK_VOTE
                                record output as the normalized vote count.
        out_dir (str): Where to store the output arrays
    """
    model_inputs = list(map(np.atleast_3d, model_inputs))
    in_shape = model_inputs[0].shape[:-1]

    valid_dilation_f = lambda _, y: y > 1 or not bool(y % float(1))
    if not all(starmap(valid_dilation_f, zip(in_shape, repeat(dilation)))):
        raise ValueError("Invalid dilation value.")

    out_shape = [*list(map(lambda x: int(x * dilation), in_shape)), n_classes]
    out_dir_f = lambda s: os.path.join(out_dir, s) if out_dir else None

    if update_map is None:
        update_map = np.ones(list(map(lambda x: x * dilation, window_shape)))

    if aggregate_method == AGGREGATION_METHODS.MEAN_VAR:
        hdul_lbl, arr_lbl = label_helper.get_mean_var_array(
            out_shape, out_dir_f("output.fits")
        )
    elif aggregate_method == AGGREGATION_METHODS.RANK_VOTE:
        hdul_lbl, arr_lbl = label_helper.get_rank_vote_array(
            out_shape, out_dir_f("output.fits")
        )
    else:
        raise ValueError(AGGREGATION_METHODS.INVALID_ERR)

    hdul_n, arr_n = label_helper.get_n_array(out_shape[:-1], out_dir_f("n.fits"))

    indicies = label_helper.get_windowed_index_generator(in_shape, window_shape, stride)

    window_dim0, window_dim1 = window_shape
    stride_dim0, stride_dim1 = stride
    num_idxs = ((in_shape[0] - window_dim0 + 1) // stride_dim0) * (
        (in_shape[1] - window_dim1 + 1) // stride_dim1
    )

    pbar = tqdm(total=num_idxs // batch_size, desc="classifying", unit="batch")

    batch_generator = (list(islice(indicies, batch_size)) for _ in repeat(None))
    batch_indices = takewhile(lambda x: len(x) > 0, batch_generator)

    batch_func = partial(build_batch, model_inputs, window_shape)
    batches_and_idxs = map(batch_func, batch_indices)

    classify_func = partial(predict_batch, model)

    # TODO: Implement an async queue system for predicting and updating results
    async_update = False
    if async_update:
        pass
    else:
        update_func = partial(
            udpate_batch,
            aggregate_method,
            update_map,
            stride,
            dilation,
            arr_n,
            arr_lbl,
        )

        for _ in starmap(update_func, starmap(classify_func, batches_and_idxs)):
            pbar.update()

    hduls = [hdul_lbl, hdul_n]
    outputs = [arr_lbl, arr_n]

    return hduls, outputs


def predict(
    model: Callable,
    model_inputs: List[Union[np.ndarray, str]],
    n_classes: int,
    batch_size: int,
    window_shape: Tuple[int, int],
    dilation: float = 1,
    stride: Tuple[int, int] = (1, 1),
    update_map: np.ndarray = None,
    aggregate_method: str = AGGREGATION_METHODS.RANK_VOTE,
    out_dir: str = None,
    gpus: List[int] = None,
    cpus: int = None,
    parallel_check_interval: float = 1,
) -> Tuple[List[fits.HDUList], List[np.ndarray]]:
    """Applies the `model` the `model_inputs`

    If you are using the parallel functionality, then `model` must be pickleable.


    Args:
        model (Callable): The model to apply to the inputs
        model_inputs (List[Union[np.ndarray, str]]): The inputs to classify
                                                     using the given `model`
        n_classes (int): The number of classes that are output
        batch_size (int): The number of examples to include in a batch
        window_shape (int): The (height, width) of the samples to extract
        stride (Tuple[int, int]): How many (rows, columns) to move through the
                                  image at each iteration.
        update_map (np.narray): A 2D array of the same size as window height that
                                indicates which pixels to use to updates for each
                                example
        aggregate_method (str): How to process the output from the model. If
                                AGGREGATION_METHODS.MEAN_VAR record output using
                                mean and variance, If AGGREGATION_METHODS.RANK_VOTE
                                record output as the normalized vote count.
        out_dir (str): The directory to save output files in if the `model_inputs`
                       are string locations.
        gpus (List[int]): The gpu ids to use for parallel processesing
        cpus (int): The number of cpus to use for parllel processing

    Returns:
        A 2-Tuple where the first element is the list of fits.HDULS for the
        outputfiles. The second element is a list of the output arrays from the
        model given the the input arrays.

    Raises:
        ValueError if `model_inputs` are not all of the same type
        ValueError if `model_inputs` are not str or np.ndarray
        ValueError if both gpus and cpus are given
        ValueError is cpus or gpus are given, but out_dir is not given
        ValueError if len(gpus)==1
        ValueError if cpus<2
    """

    inputs_are_str = misc_helper.vaidate_input_types_is_str(model_inputs)
    workers, is_gpu = misc_helper.validate_parallel_params(gpus, cpus, out_dir)

    if inputs_are_str:
        in_hduls, inputs = fits_helper.open_files(model_inputs, "readonly")
    else:
        in_hduls, inputs = [], model_inputs

    if len(workers) == 1:
        out_hduls, outputs = predict_arrays(
            model,
            inputs,
            n_classes,
            batch_size,
            window_shape,
            dilation,
            stride,
            update_map,
            aggregate_method,
            out_dir,
        )
    else:
        parallel_helper.build_parallel_classification_structure(
            model,
            inputs,
            model_inputs,
            n_classes,
            batch_size,
            window_shape,
            dilation,
            stride,
            update_map,
            aggregate_method,
            out_dir,
            workers,
        )

        parallel_helper.run_parallel_jobs(
            workers, is_gpu, out_dir, parallel_check_interval
        )

        out_hduls, outputs = parallel_helper.stitch_parallel_classifications(
            workers, out_dir, aggregate_method, window_shape
        )

    misc_helper.apply(lambda hdul: hdul.close(), in_hduls)

    return out_hduls, outputs
