#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Tur November 24 2022
# =============================================================================
"""Module for utils."""
# =============================================================================
# Imports
# =============================================================================
# standard library imports
import os
import random as rn
# related third party imports
import numpy as np


def seed_everything(seed):
    """Seed everything."""
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)

def kwargs_to_dict(*kwargs):
    """Convert kwarg NoneTypes to dicts.

    Parameters
    ----------
    kwargs : sequence
        sequence of either dicts or NoneTypes.

    Returns
    -------
    sequence of dictionaries
        Kwarg dicts are either empty or filled.
    """
    kwargs_list = list(kwargs)
    for i, kwarg in enumerate(kwargs):
        if kwarg is None:
            kwargs_list[i] = dict()
    return tuple(kwargs_list)


def subset_ary(idx, *arrs):
    """Subset arrays.

    Parameters
    ----------
    idx : ndarray
        1D array (`int` type) containing indices of subset.

    Returns
    -------
    sequence
        Sequence of subsetted 1D arrays.
    """
    arrs_list = [None] * len(arrs)
    for i, arr in enumerate(arrs):
        arrs_list[i] = arr[idx]
    return tuple(arrs_list)
