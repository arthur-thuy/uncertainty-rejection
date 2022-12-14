#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Tue December 6 2022
# =============================================================================
"""Testing script for plotting."""
# =============================================================================
# Imports
# =============================================================================
# standard library imports
# related third party imports
import numpy as np
import pytest
import matplotlib
# local application/library specific imports
from uncertainty_rejection.datasets import (
    load_mnist_data,
    load_notmnist_data,
    load_example_predictions
)

from uncertainty_rejection.plotting import (
    hist_unc_base,
    hist_unc_plot1,
    hist_unc_plot3,
    count_unc_base,
    count_unc_plot1,
    rejection_base,
    rejection_setmetric_plot1,
    rejection_setmetric_plot3,
    rejection_mixmetric_plot3
)

from uncertainty_rejection.analysis import (
    compute_uncertainty,
    compute_confidence,
    concat_get_idx
)

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name


@pytest.fixture
def y_stack():
    y_stack, _, _ = load_example_predictions()
    return y_stack


@pytest.fixture
def y_mean():
    _, y_mean, _ = load_example_predictions()
    return y_mean


@pytest.fixture
def y_label():
    _, _, y_label = load_example_predictions()
    return y_label


@pytest.fixture
def unc_tot():
    y_stack, _, _ = load_example_predictions()
    unc_tot, _, _ = compute_uncertainty(y_stack)
    return unc_tot


@pytest.fixture
def unc_ale():
    y_stack, _, _ = load_example_predictions()
    _, unc_ale, _ = compute_uncertainty(y_stack)
    return unc_ale


@pytest.fixture
def unc_epi():
    y_stack, _, _ = load_example_predictions()
    _, _, unc_epi = compute_uncertainty(y_stack)
    return unc_epi


@pytest.fixture
def conf():
    y_stack, _, _ = load_example_predictions()
    conf = compute_confidence(y_stack)
    return conf


@pytest.fixture
def y_true_all():
    # load MNIST data (do not need train data or test features)
    (_, _), (_, y_mnist) = load_mnist_data()
    # load Not-MNIST data (do not need features)
    (_, _), (_, y_notmnist) = load_notmnist_data()
    # give not-MNIST observation a different label -> should be incorrect
    y_notmnist.fill(999)
    # Concatenate true labels and get index vectors
    y_true_all, _, _, _, *_ = concat_get_idx(y_mnist, y_notmnist)
    return y_true_all


class TestHistUncBase:
    def test_integration(self, unc_tot):
        ax = hist_unc_base(unc_tot)
        assert isinstance(ax, matplotlib.axes.SubplotBase)


class TestHistUncPlot1:
    def test_integration1(self, unc_tot):
        ax = hist_unc_plot1(unc_tot, unc_type="TU", num_classes=10)
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration2(self, conf):
        ax = hist_unc_plot1(conf, unc_type="Conf", bars_scale=True)
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration3(self, conf):
        idx = np.arange(conf.shape[0])
        ax = hist_unc_plot1(conf, idx=idx, unc_type="Conf", bars_scale=True)
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_error1(self, unc_tot):
        with pytest.raises(ValueError):
            hist_unc_plot1(unc_tot, unc_type="test", num_classes=10)

    def test_error2(self, unc_tot):
        with pytest.raises(ValueError):
            hist_unc_plot1(unc_tot, unc_type="TU", num_classes=None)


class TestHistUncPlot3:
    def test_integration(self, unc_tot, unc_ale, unc_epi):
        axes = hist_unc_plot3(unc_tot, unc_ale, unc_epi, num_classes=10)
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration2(self, unc_tot, unc_ale, unc_epi):
        idx = np.arange(unc_tot.shape[0])
        axes = hist_unc_plot3(unc_tot, unc_ale, unc_epi, num_classes=10, idx=idx, bars_scale=True)
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_error1(self, unc_tot):
        with pytest.raises(ValueError):
            hist_unc_plot3(unc_tot, unc_ale, unc_epi, num_classes=-1)


class TestCountUncBase:
    def test_integration(self, unc_tot):
        ax = count_unc_base(unc_tot)
        assert isinstance(ax, matplotlib.axes.SubplotBase)


class TestCountUncPlot1:
    def test_integration1(self, unc_tot):
        ax = count_unc_plot1(unc_tot, unc_type="TU")
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration2(self, unc_tot):
        idx = np.arange(unc_tot.shape[0])
        ax = count_unc_plot1(unc_tot, unc_type="TU", idx=idx)
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_error(self, unc_tot):
        with pytest.raises(ValueError):
            count_unc_plot1(unc_tot, unc_type="test")


class TestRejectionBase:
    def test_integration1(self, y_true_all, y_stack, unc_tot):
        ax = rejection_base(y_true_all, y_stack, unc_tot,
                            metric="nra", unc_type="TU")
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration2(self, y_true_all, y_stack, unc_tot):
        ax = rejection_base(y_true_all, y_stack, unc_tot,
                            metric="nra", unc_type="TU", relative=False)
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration3(self, y_true_all, y_stack, conf):
        ax = rejection_base(y_true_all, y_stack, conf,
                            metric="nra", unc_type="TU", relative=False)
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_error(self, unc_tot):
        with pytest.raises(ValueError):
            rejection_base(y_true_all, y_stack, unc_tot,
                            metric="nra", unc_type="test")


class TestRejectionSetmetricPlot1:
    def test_integration1(self, y_true_all, y_stack, unc_tot):
        ax = rejection_setmetric_plot1(
            y_true_all, y_stack, unc_tot, metric="nra", unc_type="TU")
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration2(self, y_true_all, y_stack, conf):
        idx = np.arange(conf.shape[0])
        ax = rejection_setmetric_plot1(
            y_true_all, y_stack, conf, metric="nra", unc_type="Conf", relative=False, idx=idx)
        assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_error(self, unc_tot):
        with pytest.raises(ValueError):
            rejection_setmetric_plot1(
            y_true_all, y_stack, unc_tot, metric="nra", unc_type="test")


class TestRejectionSetmetricPlot3:
    def test_integration1(self, y_true_all, y_stack, unc_tot, unc_ale, unc_epi):
        axes = rejection_setmetric_plot3(y_true_all, y_stack, unc_tot, unc_ale, unc_epi,
                                         metric="nra")
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration2(self, y_true_all, y_stack, unc_tot, unc_ale, unc_epi):
        idx = np.arange(unc_tot.shape[0])
        axes = rejection_setmetric_plot3(y_true_all, y_stack, unc_tot, unc_ale, unc_epi,
                                         metric="nra", idx=idx, relative=False)
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_error(self, unc_tot):
        with pytest.raises(ValueError):
            rejection_setmetric_plot3(y_true_all, y_stack, unc_tot, unc_ale, unc_epi,
                                         metric="test")


class TestRejectionMixmetricPlot3:
    def test_integration1(self, y_true_all, y_stack, unc_tot):
        axes = rejection_mixmetric_plot3(
            y_true_all, y_stack, unc_tot, unc_type="TU")
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_integration2(self, y_true_all, y_stack, conf):
        idx = np.arange(conf.shape[0])
        axes = rejection_mixmetric_plot3(
            y_true_all, y_stack, conf, unc_type="TU", idx=idx)
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.SubplotBase)

    def test_error(self, unc_tot):
        with pytest.raises(ValueError):
            rejection_mixmetric_plot3(
            y_true_all, y_stack, unc_tot, unc_type="test")
