#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Mon December 5 2022
# =============================================================================
"""Testing script for utils."""
# =============================================================================
# Imports
# =============================================================================
# standard library imports
# related third party imports
import numpy as np
import pytest
# local application/library specific imports
from uncertainty_rejection.utils import (
    subset_ary,
    kwargs_to_dict
)

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ((None, None, {"a":1}), ({}, {}, {"a":1})),
        (({"a":1}, {"b":2}, {"c":3}), ({"a":1}, {"b":2}, {"c":3})),
        ((None, None, None), ({}, {}, {}))
    ]
)
def test_kwargs_to_dict(inputs, expected):
    actual = kwargs_to_dict(*inputs)
    assert actual == expected

@pytest.mark.parametrize(
    "inputs, expected",
    [
        ((np.array([0,1,2]), np.arange(0, 10, 1), np.arange(10, 0, -1)),
        (np.array([0, 1, 2]), np.array([10,  9,  8])))
    ]
)
def test_subset_ary(inputs, expected):
    actuals = subset_ary(*inputs)
    for actual, expect in zip(actuals, expected): # cannot directly assert tuple of arrays
        assert actual == pytest.approx(expect)
