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
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np


def vaidate_input_types_is_str(inputs: List[Union[str, np.ndarray]]) -> bool:
    """Validates that the inputs are all the same type and one of str or np.ndarray.


    Args:
        inputs (List[Union[str, np.ndarray]]): List of inputs to validate
    Returns:
        true if the inputs are str and false if the inputs are np.ndarray
    Raises:
        ValueError if all inputs are not the same type
        ValueError if the types are other than np.ndarray or str

    """
    types = set(map(type, inputs))
    if len(types) > 1:
        raise ValueError(
            "Mixed input type usuage. Ensure all are numpy arrays or strings."
        )

    t = types.pop()

    if t in [str, np.ndarray]:
        return t == str
    else:
        raise ValueError("Input type must either be np.ndarray or string")


def validate_parallel_params(
    gpus: List[int] = None, cpus: int = None, out_dir: str = None
) -> Tuple[List[int], bool]:
    """Validates that the parallel params.

    Args:
        gpus (List[int]): list GPU ids to use for parallel classification
        cpus (int): number of cpus to use for parallel classification
    Returns:
        A tuple where the first element is a List of integer id values for each
        worker. The second element is true if the ids are gpu ids and false if
        they are cpu ids
    Raises:
        ValueError if both gpus and cpus are given
        ValueError is cpus or gpus are given, but out_dir is not given
        ValueError if len(gpus)==1
        ValueError if cpus<2
    """
    if (gpus is not None) and (cpus is not None):
        raise ValueError("Please only give a value cpus or gpus, not both.")

    if (gpus is None) and (cpus is None):
        return [0], False

    if (gpus is not None or cpus is not None) and (out_dir is None):
        raise ValueError("Parallel classification requires an out_dir")

    if gpus is not None:
        if len(gpus) == 1:
            err = " ".join(
                [
                    "Only one gpu indicated. If you are trying to select",
                    "a single gpu, then use the CUDA_VISIBLE_DEVICES environment",
                    "variable. For more information visit:",
                    "https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/",
                ]
            )

            raise ValueError(err)
        else:
            return gpus, True
    else:
        if cpus < 2:
            raise ValueError("If passing cpus please indicate a value greater than 1.")

        return np.arange(cpus), False


def arrays_not_same_size(inputs: List[np.ndarray]) -> bool:
    """Validates that all input arrays are the same size.

    Args:
        inputs (List[np.ndarray]): Input arrays to validate
    Returns:
        true if the arrays are the same size and false if they are not
    """
    shapes = [i.shape for i in inputs]
    shp_first = shapes[0]
    shp_rest = shapes[1:]

    return not np.array_equiv(shp_first, shp_rest)


def apply(f: Callable, args: Iterable, kwargs: Iterable[dict] = None) -> None:
    """Applies the function f to the args and kwargs.

    Args:
        f (Callable): fucntion to apply
        args (Iterable): iterable to apply f to
        kwargs (Iterable[dict]): iterable of a dict of kwargs to apply with
                                 each element in args
    Returns:
        None
    """
    try:
        i_args = iter(args)
        i_kwargs = iter(kwargs) if kwargs else None
        while True:
            a = next(i_args)
            k = next(i_kwargs) if kwargs else {}
            if (type(a) is tuple) or (type(a) is list):
                f(*a, **k)
            else:
                f(a, **k)
    except StopIteration:
        pass
