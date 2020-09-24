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
from itertools import chain, product, repeat
from functools import partial
from typing import Iterable, List, Tuple, Union

import numpy as np
from astropy.io import fits

from morpheus_core.helpers import fits_helper


def get_mean_var_array(
    shape: Union[List[int], Tuple[int]], write_to: str = None
) -> Tuple[Union[fits.HDUList], np.ndarray]:
    """Make label arrays for storing the model output.

    Args:
        shape (Union[List[int], Tuple[int]]): Gets the array for storing `n`
                                              values
        write_to (str): If supplied is the place where to write the array.
                        Otherwise the array is created in memory

    Returns:
        A 2-tuple where the first item if `write_to` is supplied, otherwise
        None and the second item is a numpy array
    """

    out_array_shape = list(chain(shape, [2]))

    if write_to:
        # make the fits file and return a reference to the array
        fits_helper.create_file(write_to, out_array_shape, dtype=np.float32)
        hdul, array = fits_helper.open_file(write_to, mode="update")
    else:
        # make the array in memory
        hdul, array = None, np.zeros(out_array_shape, dtype=np.float32)

    return hdul, array


def get_rank_vote_array(
    shape: Union[List[int], Tuple[int]], write_to: str = None
) -> Tuple[Union[fits.HDUList], np.ndarray]:
    """Make label arrays for storing the model output.

    Args:
        shape (Union[List[int], Tuple[int]]): Gets the array for storing `n`
                                              values
        write_to (str): If supplied is the place where to write the array.
                        Otherwise the array is created in memory

    Returns:
        A 2-tuple where the first item if `write_to` is supplied, otherwise
        None and the second item is a numpy array
    """

    if write_to:
        # make the fits file and return a reference to the array
        fits_helper.create_file(write_to, shape, dtype=np.float32)
        hdul, array = fits_helper.open_file(write_to, mode="update")
    else:
        # make the array in memory
        hdul, array = None, np.zeros(shape, dtype=np.float32)

    return hdul, array


def get_n_array(
    shape: Union[List[int], Tuple[int]], write_to: str = None
) -> Tuple[Union[fits.HDUList, None], np.ndarray]:
    """Make label arrays for storing the model output.

    Args:
        shape (Union[List[int], Tuple[int]]): Gets the array for storing `n`
                                              values
        write_to (str): If supplied is the place where to write the array.
                        Otherwise the array is created in memory

    Returns:
        A 2-tuple where the first item if `write_to` is supplied, otherwise
        None and the second item is a numpy array
    """

    if write_to:
        # make the fits file and return a reference to the array
        fits_helper.create_file(write_to, shape, dtype=np.float32)
        hdul, array = fits_helper.open_file(write_to, mode="update")
    else:
        # make the array in memory
        hdul, array = None, np.zeros(shape, dtype=np.float32)

    return hdul, array


def get_windowed_index_generator(
    img_wh: Tuple[int, int],
    window_shape: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
) -> Iterable[Tuple[int, int]]:
    """Creates a generator that returns window limited indices over a 2d array.

    Args:
        img_wh (Tuple[int, int]): The (height, width) of the total image size
        window_shape (Tuple[int, int]): The (height, width) of the input/output
                                        to the classifier
        stride (Tuple[int, int]): The distance, in pixels, to move along the
                                  (height, width) of the image.

    Returns:
        An iterable containing tuples of ints that are the indexes to use to
        extract samples from the large image.
    """
    if len(img_wh) != 2 or len(window_shape) != 2 or len(stride) != 2:
        err = "img_wh, window_shape, and stride should have a length of 2"
        raise ValueError(err)

    window_dim0, window_dim1 = window_shape
    img_dim0, img_dim1 = img_wh
    stride_dim0, stride_dim1 = stride

    final_y = img_dim0 - window_dim0 + 1
    final_x = img_dim1 - window_dim1 + 1

    return product(range(0, final_y, stride_dim0), range(0, final_x, stride_dim1))


def get_final_map(
    total_shape: Tuple[int, int],
    update_mask_shape: Tuple[int, int],
    stride: Tuple[int, int],
    output_idx: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Creates a boolean array indicating which pixels have completed classification.

    Args:
        total_shape (Tuple[int, int]): The (height, width) of the total image
                                       indices in the array should be updated
        update_mask_shape (Tuple[int, int]): The (height, width) of the update
                                             mask
        stride (Tuple[int, int]): The distance, in pixels, to move along the
                                  (height, width) of the image.
        output_idx (Tuple[int, int]): the y, x value that idicate where in the
                                      image the update is happening

    Returns:
        A list of tuples that contain the (y,x) coordinates that are done being
        classified.
    """

    y, x = output_idx
    stride_y, stride_x = stride

    window_y, window_x = update_mask_shape

    end_y = y == (total_shape[0] - window_y - (total_shape[0] % stride_y))
    end_x = x == (total_shape[1] - window_x - (total_shape[1] % stride_x))

    if end_y and end_x:  # final image
        idxs = product(range(window_y), range(window_x))
    elif end_y:  # final row
        idxs = product(range(window_y), range(stride_x))
    elif end_x:  # final column
        idxs = product(range(stride_y), range(window_x))
    else:  # any other typical image
        idxs = product(range(stride_y), range(stride_x))

    return list(idxs)


def update_n(
    update_mask: np.ndarray, n: np.ndarray, output_idx: Tuple[int, int]
) -> np.ndarray:
    """Updates the counts that are stored in 'n' array.

    Args:
        update_mask (np.ndarray): a 2d boolean array indicating which
                                  indices in the array should be updated
        n (np.ndarray): a 2d array containing the number of terms used in the
                        mean
        output_idx (Tuple[int, int]): the y, x values that idicate where in the
                                      image the updates should happen

    Returns:
        The n array with updated values
    """
    window_y, window_x = update_mask.shape

    y, x = output_idx
    ys = slice(y, y + window_y)
    xs = slice(x, x + window_x)

    n_current = n[ys, xs].copy()
    n_update = update_mask.astype(np.int)
    n_updated = n_current + n_update
    n[ys, xs] = n_updated

    return n


def iterative_mean(
    n: np.ndarray, curr_mean: np.ndarray, x_n: np.ndarray, update_mask: np.ndarray
) -> np.ndarray:
    """Calculates the mean of collection in an online fashion.
    The values are calculated using the following equation:
    http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, eq. 4

    Args:
        n (np.ndarray): a 2d array containing the number of terms used in the
                        mean
        curr_mean (np.ndarray): the current calculated mean
        x_n (np.ndarray): the new values to add to the mean
        update_mask (np.ndarray): a 2d boolean array indicating which
                                    indices in the array should be updated

    Returns:
        An array with the same shape as the curr_mean with the updated mean
        values
    """
    n[n == 0] = 1
    return curr_mean + ((x_n - curr_mean) / n * update_mask)


def iterative_variance(
    prev_sn: np.ndarray,
    x_n: np.ndarray,
    curr_mean: np.ndarray,
    next_mean: np.ndarray,
    update_mask: np.ndarray,
) -> np.ndarray:
    """The first of two methods used to calculate the variance online.

    This method specifically calculates the $S_n$ value as indicated in
    equation 24 from:

    http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

    Args:
        prev_sn (np.ndarray): the $S_n$ value from the previous step
        x_n (np.ndarray): the current incoming values
        curr_mean (np.ndarray): the mean that was previously calculated
        next_mean (np.ndarray): the mean, including the current values
        update_mask (np.ndarray): a boolean mask indicating which values to
                                    update

    Returns:
        An np.ndarray containg the current value for $S_n$
    """
    return prev_sn + ((x_n - curr_mean) * (x_n - next_mean) * update_mask)


def finalize_variance(
    n: np.ndarray, final_map: List[Tuple[int, int]], curr_sn: np.ndarray
) -> np.ndarray:
    """The second of two methods used to calculate the variance online.

    This method calculates the final variance value using equation 25 from
    http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
    but without performing the square root.

    Args:
        n (np.ndarray): the current number of values included in the calculation
        final_map List[(y, x)]: a list of indices to calculate the final
                                variance for
        curr_sn (np.ndarray): the current $S_n$ values

    Returns:
        A np.ndarray with the current $S_n$ values and variance values for
        all indices in final_map
    """
    final_n = np.ones_like(n)
    ys, xs = zip(*final_map)
    final_n[ys, xs] = n[ys, xs]

    return curr_sn / final_n


def update_single_class_mean_var(
    update_mask: np.ndarray, n: np.ndarray, mean_var: np.ndarray, x_n: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Updates the mean and variance for a single class.

    Args:
        update_mask (np.ndarray): a 2d boolean array indicating which
                                  indices in the array should be updated
        n (np.ndarray): a 2d array containing the number of terms used in the
                        mean
        mean_var (np.ndarray): the current calculated mean and variance
        x_n (np.ndarray): the new values to add to update the mean and variance

    Returns:
        A tuple containing two numpy arrays that contain the updated mean and
        variance repsectively
    """

    prev_mean = mean_var[:, :, 0].copy()
    prev_var = mean_var[:, :, 1].copy()

    next_mean = iterative_mean(n, prev_mean, x_n, update_mask)
    next_var = iterative_variance(prev_var, x_n, prev_mean, next_mean, update_mask)

    return next_mean, next_var


def update_mean_var(
    update_mask: np.ndarray,
    stride: Tuple[int, int],
    n: np.ndarray,
    output: np.ndarray,
    single_out: np.ndarray,
    output_idx: Tuple[int, int],
) -> None:
    """Updates the mean and variance with the recently classified values.

    Args:
        update_mask (np.ndarray): a 2d boolean array indicating which
                                  indices in the array should be updated
        stride (Tuple[int, int]): How many (rows, columns) to move through the
                                  image at each iteration.
        n (np.ndarray): a 2d array containing the number of terms used in the
                        mean
        output (np.ndarray): The current running output array containing the
                             overall mean and variance
        single_out (np.ndarray): The new output values to update the mean and
                                 variance with
        output_idx (Tuple[int, int]): the y, x values that idicate where in the
                                      image the updates should happen

    Returns:
        None
    """
    y, x = output_idx
    window_y, window_x = update_mask.shape
    ys = slice(y, y + window_y)
    xs = slice(x, x + window_x)
    extract_output_class_values = lambda i: output[ys, xs, i, :].copy()
    extract_batch_out_class_values = lambda i: single_out[:, :, i].copy()

    # variables to use for update
    update_n(update_mask, n, output_idx)
    batch_ns = n[ys, xs].copy()

    n_classes = single_out.shape[2]
    x_ns = map(extract_batch_out_class_values, range(n_classes))
    single_class_values = map(extract_output_class_values, range(n_classes))

    # update partial function
    update_f = partial(update_single_class_mean_var, update_mask, batch_ns)

    # updated_values
    next_means, next_vars = zip(*map(update_f, single_class_values, x_ns))

    # finalize variance values
    final_map = get_final_map(n.shape, update_mask.shape, stride, output_idx)
    final_f = partial(finalize_variance, batch_ns, final_map)
    final_vars = map(final_f, next_vars)

    # [classes, window_y, window_y, 2]
    updated_values = np.array(
        [np.dstack((m, v)) for m, v in zip(next_means, final_vars)]
    )

    # permute dims to match output [window_y, window_x, classes, 2]
    output[ys, xs, :, :] = np.transpose(updated_values, axes=(1, 2, 0, 3))


def finalize_rank_vote(
    n: np.ndarray, final_map: List[Tuple[int, int]], output: np.ndarray
) -> np.ndarray:
    """Performs final calulation on completely classified pixels.

    Args:
        n (np.ndarray): an array containing the total number of times a each
                        pixel has been classified
        final_map (np.ndarray): an boolean array indicating which pixels are
                                finished being classified
        output (np.ndarray): an array containing the current running
                             classifications

    Returns:
        An array with the same shape as output with updated values according
        to the final_map parameter.
    """

    ys, xs = zip(*final_map)
    final_arr = np.zeros_like(output)
    final_arr[ys, xs] = 1

    n_with_dim = n[:, :, np.newaxis].copy()
    return np.divide(
        output, n_with_dim, out=output, where=np.logical_and(final_arr, n_with_dim > 0)
    )


def update_rank_vote(
    update_mask: np.ndarray,
    stride: Tuple[int, int],
    n: np.ndarray,
    output: np.ndarray,
    single_output: np.ndarray,
    output_idx: Tuple[int, int],
) -> None:
    """Updates the rank vote values with the recently classified output.


    Args:
        update_mask (np.ndarray): a 2d boolean array indicating which
                                  indices in the array should be updated
        stride (Tuple[int, int]): How many (rows, columns) to move through the
                                  image at each iteration.
        n (np.ndarray): an array containing the total number of times a each
                        pixel has been classified
        output (np.ndarray): an array containing the current running
                             classifications
        final_map (np.ndarray): an boolean array indicating which pixels are
                                finished being classified
        single_output (np.ndarray): The new output values to update the mean and
                                    variance with
        output_idx (Tuple[int, int]): the y, x values that idicate where in the
                                      image the updates should happen

    Returns:
        None
    """
    y, x = output_idx
    window_y, window_x = update_mask.shape
    ys = slice(y, y + window_y)
    xs = slice(x, x + window_x)

    update_n(update_mask, n, output_idx)

    # calling argsort twice returns a the rank for each item starting from 0
    ranked = single_output.argsort(axis=-1).argsort(axis=-1)
    top_votes = ranked == (ranked.shape[2] - 1)
    update = np.dstack(
        [update_mask * top_votes[:, :, i] for i in range(top_votes.shape[2])]
    )

    final_map = get_final_map(n.shape, update_mask.shape, stride, output_idx)
    finalized_values = finalize_rank_vote(
        n[ys, xs].copy(), final_map, output[ys, xs, :].copy() + update
    )

    output[ys, xs, :] = finalized_values
