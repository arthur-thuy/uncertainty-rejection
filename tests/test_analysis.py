#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Mon November 21 2022
# =============================================================================
"""Testing script for analysis."""
# =============================================================================
# Imports
# =============================================================================
# standard library imports
# related third party imports
import numpy as np
import pytest
# local application/library specific imports
from uncertainty_rejection.analysis import (
    get_pos_neg_probs,
    get_y_mean_label,
    load_predictions,
    compute_uncertainty,
    compute_confidence,
    get_idx_correct,
    confusion_matrix_rej,
    compute_metrics_rej,
    compute_count_unc
)

# run with: `python3 -m pytest -v` from within src folder
# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name

@pytest.fixture
def pos_probs():
    return np.array([[0.72, 0.85, 0.80],
                     [0.88, 0.78, 0.92]])


@pytest.fixture
def y_stack():
    return np.array([[[0.28, 0.72],
                      [0.15, 0.85],
                      [0.2, 0.8]],

                     [[0.12, 0.88],
                      [0.22, 0.78],
                      [0.08, 0.92]]])


@pytest.fixture
def y_true_label():
    return np.array([0., 1., 1., 0., 1.])


@pytest.fixture
def y_pred_label():
    return np.array([0., 0., 1., 0., 0.])


@pytest.fixture
def unc_ary():
    return np.array([0.2, 0.8, 0.4, 0.6, 0.5])


def test_get_pos_neg_probs(pos_probs, y_stack):
    actual = get_pos_neg_probs(pos_probs)
    expected = y_stack
    np.testing.assert_allclose(
        actual, expected, err_msg='Negative probs computed or stacked incorrectly!')


class TestGetYMeanLabel:
    def test_unit(self, y_stack):
        actual = get_y_mean_label(y_stack)
        expected_y_mean = np.array([[(0.28+0.15+0.2)/3, (0.72+0.85+0.8)/3],
                                    [(0.12+0.22+0.08)/3, (0.88+0.78+0.92)/3]])
        expected_y_label = np.array([1., 1.])
        np.testing.assert_allclose(
            actual[0], expected_y_mean, err_msg='`y_mean` computed incorrectly!')
        np.testing.assert_allclose(
            actual[1], expected_y_label, err_msg='`y_label` computed incorrectly!')

    def test_error(self, pos_probs):
        with pytest.raises(ValueError):
            get_y_mean_label(pos_probs)


def test_load_predictions():
    y_stack, y_mean, y_label = load_predictions("tests/preds.npy")

    assert isinstance(y_stack, np.ndarray), "`y_stack` should be a np.ndarray"
    assert isinstance(y_mean, np.ndarray), "`y_mean` should be a np.ndarray"
    assert isinstance(y_label, np.ndarray), "`y_label` should be a np.ndarray"

    assert y_stack.ndim == 3, f"`y_stack` should have 3 dimensions, is rank {y_stack.ndim}"
    assert y_mean.ndim == 2, f"y_mean should be a tensor of rank 2, is rank {y_mean.ndim}"
    assert y_label.ndim == 1, f"y_label should be a tensor of rank 1, is rank {y_label.ndim}"


@pytest.mark.parametrize(
    "y_stack, unc_tuple",
    [
        (np.array([[[0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0]]]), (np.array([0.]), np.array([0.]), np.array([0.]))),
        (np.array([[[1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0]]]), (np.array([1.]), np.array([0.]), np.array([1.]))),
        (np.array([[[0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5]]]), (np.array([1.]), np.array([1.]), np.array([0.])))
    ]
)
def test_decompose_uncertainty(y_stack, unc_tuple):
    actual_unc_total, actual_unc_aleatoric, actual_unc_epistemic = compute_uncertainty(
        y_stack)
    assert actual_unc_total == pytest.approx(unc_tuple[0]), \
        f"Total uncertainty should be {float(unc_tuple[0][0])}, is {float(actual_unc_total[0])}."
    assert actual_unc_aleatoric == pytest.approx(unc_tuple[1]), \
        f"Aleatoric uncertainty should be {float(unc_tuple[1][0])}, is {float(actual_unc_aleatoric[0])}."
    assert actual_unc_epistemic == pytest.approx(unc_tuple[2]), \
        f"Epistemic uncertainty should be {float(unc_tuple[2][0])}, is {float(actual_unc_epistemic[0])}."


class TestComputeConfidence:
    def test_unit(self, y_stack):
        actual = compute_confidence(y_stack)
        expected_y_mean = np.array([[(0.28+0.15+0.2)/3, (0.72+0.85+0.8)/3],
                                    [(0.12+0.22+0.08)/3, (0.88+0.78+0.92)/3]])
        expected = np.max(expected_y_mean, axis=-1)
        np.testing.assert_allclose(
            actual, expected, err_msg='confidence computed incorrectly!')

    def test_error(self, pos_probs):
        with pytest.raises(ValueError):
            compute_confidence(pos_probs)


def test_get_idx_correct(y_true_label, y_pred_label):
    actual = get_idx_correct(y_true_label, y_pred_label)
    expected_idx_correct, expected_idx_incorrect = np.array(
        [0, 2, 3]), np.array([1, 4])
    np.testing.assert_array_equal(
        actual[0], expected_idx_correct, "`idx_correct` computed incorrectly!")
    np.testing.assert_array_equal(
        actual[1], expected_idx_incorrect, "`idx_incorrect` computed incorrectly!")


class TestConfMatrixRej:
    @pytest.mark.parametrize(
        "relative, threshold, matrix_rej",
        [
            (False, 0.45, (1, 2, 2, 0)),
            (False, 0.1, (3, 0, 2, 0)),
            (False, 0.9, (0, 3, 0, 2)),
            (True, 0.45, (1, 2, 2, 0)),
            (True, 0.1, (0, 3, 1, 1)),
            (True, 0.9, (3, 0, 2, 0))
        ]
    )
    def test_unit(self, y_true_label, y_pred_label, unc_ary, threshold, relative, matrix_rej):
        actual_n_cor_rej, actual_n_cor_nonrej, actual_n_incor_rej, actual_n_incor_nonrej = \
            confusion_matrix_rej(y_true_label, y_pred_label,
                                 unc_ary, threshold, relative=relative)
        expected_n_cor_rej, expected_n_cor_nonrej, expected_n_incor_rej, expected_n_incor_nonrej = matrix_rej
        assert actual_n_cor_rej == expected_n_cor_rej, \
            f"`n_cor_rej` should be {expected_n_cor_rej}, is {actual_n_cor_rej}."
        assert actual_n_cor_nonrej == expected_n_cor_nonrej, \
            f"`n_cor_nonrej` should be {expected_n_cor_nonrej}, is {actual_n_cor_nonrej}."
        assert actual_n_incor_rej == expected_n_incor_rej, \
            f"`n_incor_rej` should be {expected_n_incor_rej}, is {actual_n_incor_rej}."
        assert actual_n_incor_nonrej == expected_n_incor_nonrej, \
            f"`n_incor_nonrej` should be {expected_n_incor_nonrej}, is {actual_n_incor_nonrej}."


class TestComputeMetricsRej:
    @pytest.mark.parametrize(
        "relative, threshold, metrics_rej",
        [
            (False, 0.45, (1.0, 0.8, 3.0)),
            (False, 0.1, (0.0, 0.4, 1.0)),
            (False, 0.9, (0.6, 0.6, 1.0)),
            (True, 0.45, (1.0, 0.8, 3.0)),
            (True, 0.1, (0.75, 0.8, np.inf)),
            (True, 0.9, (0.0, 0.4, 1.0))
        ]
    )
    def test_unit(self, y_true_label, y_pred_label, unc_ary, threshold, relative, metrics_rej):
        actual_nonrej_acc, actual_class_quality, actual_rej_quality = \
            compute_metrics_rej(threshold, y_true_label,
                                y_pred_label, unc_ary, relative=relative)
        expected_nonrej_acc, expected_class_quality, expected_rej_quality = metrics_rej
        np.testing.assert_allclose(
            actual_nonrej_acc, expected_nonrej_acc, \
                err_msg=f"`nonrej_acc` should be {expected_nonrej_acc}, is {actual_nonrej_acc}.")
        np.testing.assert_allclose(
            actual_class_quality, expected_class_quality, \
                err_msg=f"`class_quality` should be {expected_class_quality}, is {actual_class_quality}.")
        np.testing.assert_allclose(
            actual_rej_quality, expected_rej_quality, \
                err_msg=f"`rej_quality` should be {expected_rej_quality}, is {actual_rej_quality}.")


@pytest.mark.parametrize(
    "threshold, count_unc",
    [
        (0.45, 3),
        (0.1, 5),
        (0.9, 0)
    ]
)
def test_compute_count_unc(unc_ary, threshold, count_unc):
    actual = compute_count_unc(threshold, unc_ary)
    expected = count_unc
    assert actual == actual, f"`count_unc` should be {expected}, is {actual}."
