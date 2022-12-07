#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Mon December 5 2022
# =============================================================================
"""Testing script for datasets."""
# =============================================================================
# Imports
# =============================================================================
# standard library imports
# related third party imports
import numpy as np
import pytest
import urllib.error
# local application/library specific imports
from uncertainty_rejection.datasets import (
    load_mnist_data,
    load_notmnist_data,
    load_example_predictions,
    get_file
)

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name

class TestLoadMnistData:
    def test_integration(self):
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert y_train.shape == (60000,)
        assert y_test.shape == (10000,)

class TestLoadNotMnistData:
    def test_integration(self):
        (x_train, y_train), (x_test, y_test) = load_notmnist_data()
        assert x_train.shape == (529114, 28, 28)
        assert x_test.shape == (18724, 28, 28)
        assert y_train.shape == (529114,)
        assert y_test.shape == (18724,)

class TestLoadExamplePredictions:
    def test_integration(self):
        y_stack_all, y_mean_all, y_label_all = load_example_predictions()
        assert y_stack_all.shape == (28724, 128, 10)
        assert y_mean_all.shape == (28724, 10)
        assert y_label_all.shape == (28724,)

class TestGetFile:
    def test_error1(self):
        with pytest.raises(ValueError):
            get_file(origin="invalid_path", fname=None)

    def test_error2 (self):
        with pytest.raises(Exception):
            get_file(origin="invalid_origin", fname="invalid_fname")
        