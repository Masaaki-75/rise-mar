import os
import torch
import random
import numpy as np
from typing import Any
from collections.abc import Iterable
from torchvision.transforms.functional import rotate


def fix_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def concat_lists(list_of_lists):
    concatenated_list = []
    [concatenated_list.extend(sublist) for sublist in list_of_lists]
    return concatenated_list

 
def mix_lists(list1, list2):
    mixed_list = []
    min_len = min(len(list1), len(list2))

    for i in range(min_len):
        mixed_list.append(list1[i])
        mixed_list.append(list2[i])

    if len(list1) > min_len:
        mixed_list.extend(list1[min_len:])
    elif len(list2) > min_len:
        mixed_list.extend(list2[min_len:])

    return mixed_list

def equalize_lists(list1, list2):
    """
    Equalizes the lengths of two lists by extending the shorter list with randomly selected elements from itself.
    """
    # Determine which list is shorter
    if len(list1) < len(list2):
        shorter, longer = list1, list2
    else:
        shorter, longer = list2, list1

    # Calculate how many elements to add
    num_elems_diff = len(longer) - len(shorter)

    # Extend the shorter list with randomly chosen elements from itself
    if num_elems_diff > 0 and shorter:
        replace = True if len(shorter) < num_elems_diff else False
        extended_elements = random.choices(shorter, k=num_elems_diff, replace=replace)
        shorter.extend(extended_elements)

    return list1, list2

    
def issequenceiterable(obj: Any) -> bool:
    """From MONAI
    Determine if the object is an iterable sequence and is not a string.
    """
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def ensure_tuple_rep(tup: Any, dim: int):
    """From MONAI
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")


def rotate_tensor(x: torch.Tensor, angle_deg, center=None, fill=None):
    fill = x.min().item() if fill is None else fill
    rotated_x = rotate(x, angle_deg, center=center, fill=fill)
    return rotated_x
