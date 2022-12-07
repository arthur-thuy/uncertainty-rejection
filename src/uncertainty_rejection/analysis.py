#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Tur November 24 2022
# =============================================================================
"""Module for analysis."""
# =============================================================================
# Imports
# =============================================================================
# standard library imports

# related third party imports
import numpy as np
from tabulate import tabulate
from scipy.stats import entropy

# local application/library specific imports
from uncertainty_rejection.utils import (
    subset_ary
)


def get_y_mean_label(y_pred_stack):
    """Compute mean predicted probabilities for all classes and predicted label.

    Parameters
    ----------
    y_pred_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.

    Returns
    -------
    y_mean : ndarray
        2D array (`float` type) of shape `(observations, classes)`.
    y_label : ndarray
        1D array (`float` type) of shape `(observations,)`.

    Examples
    --------
    TODO
    """
    if not y_pred_stack.ndim == 3:
        raise ValueError(
            f"`y_stack` should have 3 dimensions, has {y_pred_stack.ndim}")
    # average over samples (axis=-2)
    y_mean = np.mean(y_pred_stack, axis=-2)
    y_label = np.argmax(y_mean, axis=-1)
    return y_mean, y_label


def get_pos_neg_probs(y_pred_pos, axis=-1):
    """Compute probabilities for negative class, \
        and stack with probabilities for positive class.

    Parameters
    ----------
    y_pred_pos : ndarray
        2D array (`float` type) of shape `(observations, samples)`.
    axis : int, optional
        The axis in the result array along which the arrays are stacked.
        Default: -1

    Returns
    -------
    y_pred_pos_neg : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.

    Examples
    --------
    >>> a = np.array([[0.72, 0.85, 0.79],
                      [0.88, 0.78, 0.92]])
    >>> get_pos_neg_probs(a)
    array([[[0.28, 0.72],
            [0.15, 0.85],
            [0.21, 0.79]],
           [[0.12, 0.88],
            [0.22, 0.78],
            [0.08, 0.92]]])
    """
    y_pred_neg = np.subtract(1., y_pred_pos)
    # probability vector should sum to 1
    y_pred_pos = np.subtract(1., y_pred_neg)
    y_pred_pos_neg = np.stack([y_pred_neg, y_pred_pos], axis=axis)
    return y_pred_pos_neg


def load_predictions(preds_path):
    """Load array predictions and compute mean predicted probabilities for all classes \
        and predicted label.

    Parameters
    ----------
    preds_path : file-like object, string, or pathlib.Path
        The file to read.

    Returns
    -------
    y_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.
    y_mean : ndarray
        2D array (`float` type) of shape `(observations, classes)`.
    y_label : ndarray
        1D array (`float` type) of shape `(observations,)`.
    """
    y_stack = np.load(preds_path)
    if y_stack.ndim <= 2:
        y_stack = get_pos_neg_probs(y_stack, axis=-1)
    if y_stack.ndim <= 2:
        y_stack = np.expand_dims(y_stack, axis=-2)

    y_mean, y_label = get_y_mean_label(y_stack)
    print(f"y_stack shape: \t{y_stack.shape}")
    print(f"y_mean shape: \t{y_mean.shape}")
    print(f"y_label shape: \t{y_label.shape}")

    return y_stack, y_mean, y_label


def compute_uncertainty(y_pred_stack):
    """Calculate total uncertainty (TU), \
        and decompose into aleatoric uncertainty (AU) and epistemic uncertainty (EU).

    Parameters
    ----------
    y_pred_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.

    Returns
    -------
    unc_total : ndarray
        1D ndarray (`float` type) containing total uncertainty values.
    unc_aleatoric : ndarray
        1D ndarray (`float` type) containing aleatoric uncertainty values.
    unc_epistemic : ndarray
        1D ndarray (`float` type) containing epistemic uncertainty values.
    """
    # total: (observations, samples, classes) => (observations, classes) => (observations,)
    unc_total = entropy(np.mean(y_pred_stack, axis=-2),
                        base=2, axis=-1)
    # aleatoric: (observations, samples, classes) => (observations, samples) => (observations,)
    unc_aleatoric = np.mean(entropy(y_pred_stack, base=2, axis=-1), axis=-1)
    # epistemic: (observations,)
    unc_epistemic = np.subtract(unc_total, unc_aleatoric)
    return unc_total, unc_aleatoric, unc_epistemic


def compute_confidence(y_pred_stack):
    """Compute confidence.

    Parameters
    ----------
    y_pred_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.

    Returns
    -------
    conf : ndarray
        1D ndarray (`float` type) containing confidence values.
    """
    if not y_pred_stack.ndim == 3:
        raise ValueError(
            f"`y_stack` should have 3 dimensions, is rank {y_pred_stack.ndim}")
    y_pred_mean, _ = get_y_mean_label(y_pred_stack)
    conf = np.max(y_pred_mean, axis=-1)
    return conf


def concat_get_idx(*y_true_subset):
    """Concatenate true y labels and compute index vectors.

    Returns
    -------
    y_true_all : ndarray
        1D array (`float` type) containing all true labels.
    id_arr : ndarray
        1D array (`int` type) containing id numbers of each subset.
    idx_tuple : sequence of ndarray
        Sequence of 1D arrays (`int` type) containing indices of each subset.
    """
    id_list = [[i]*len(x) for i, x in enumerate(y_true_subset)]
    id_arr = np.concatenate(id_list)

    y_true_all = np.concatenate(y_true_subset, axis=0)
    idx_list = [np.where(id_arr == i)[0] for i, x in enumerate(y_true_subset)]
    idx_tuple = tuple(idx_list)
    return y_true_all, id_arr, *idx_tuple

# # pytest
# y_a = np.arange(5)
# y_b = np.arange(5)
# y_c = np.arange(5)
# y_true_all, idlist, idx_a, idx_b, idx_c, *_ = concat_get_idx(y_a, y_b, y_c)
# print(y_true_all, idlist, idx_a, idx_b, idx_c)


def get_idx_correct(y_true_label, y_pred_label):
    """Get indices of correct/incorrect predictions.

    Parameters
    ----------
    y_true_label : ndarray
        1D array (`float` type) containing true labels.
    y_pred_label : ndarray
        1D array (`float` type) containing predicted labels.

    Returns
    -------
    idx_correct : ndarray
        1D array (`int` type) containing indices of correct predictions.
    idx_incorrect : ndarray
        1D array (`int` type) containing indices of incorrect predictions.
    """
    is_correct = np.equal(y_true_label, y_pred_label)
    idx_correct = np.where(is_correct)[0]
    idx_incorrect = np.where(~is_correct)[0]
    return idx_correct, idx_incorrect


def confusion_matrix_rej(y_true_label, y_pred_label, unc_ary, threshold, relative=True, show=False,
                        seed=44):
    """Compute confusion matrix with 2 axes: (i) correct/incorrect, (ii) rejected/non-rejected.

    Parameters
    ----------
    y_true_label : ndarray
        1D array (`float` type) containing true labels.
    y_pred_label : ndarray
        1D array (`float` type) containing predicted labels.
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    threshold : float
        Rejection threshold.
    relative : bool, optional
        Use relative rejection, otherwise absolute rejection.
        Default: True
    show : bool, optional
        Print confusion matrix to console.
        Default: False
    seed: int, optional
        Seed value for random rejection.
        Default 44

    Returns
    -------
    n_cor_rej : int
        Number of correct observations that are rejected.
    n_cor_nonrej : int
        Number of correct observations that are not rejected.
    n_incor_rej : int
        Number of incorrect observations that are rejected.
    n_incor_nonrej : int
        Number of incorrect observations that are not rejected.
    """
    # axis 0: correct or incorrect
    idx_correct, idx_incorrect = get_idx_correct(y_true_label, y_pred_label)

    # axis 1: rejected or non-rejected
    if relative:
        # relative rejection
        # get indices of lowest uncertainties
        n_preds = y_true_label.shape[0]
        n_preds_nonrej = int((1-threshold)*n_preds)
        # sort by unc_ary, then by random numbers random_draws
        # -> if values equal e.g. 1.0 -> rejected randomly
        np.random.seed(seed=seed)
        random_draws = np.random.random(unc_ary.size)
        idx = np.lexsort((random_draws, unc_ary))
        idx_nonrej = idx[:n_preds_nonrej]
        idx_rej = idx[n_preds_nonrej:]
    else:
        # absolute rejection
        y_reject = np.where(unc_ary >= threshold, 1, 0)
        idx_rej = np.where(y_reject == 1)[0]
        idx_nonrej = np.where(y_reject == 0)[0]

    # intersections
    idx_cor_rej = np.intersect1d(idx_correct, idx_rej)
    idx_cor_nonrej = np.intersect1d(idx_correct, idx_nonrej)
    idx_incor_rej = np.intersect1d(idx_incorrect, idx_rej)
    idx_incor_nonrej = np.intersect1d(idx_incorrect, idx_nonrej)
    n_cor_rej = idx_cor_rej.shape[0]
    n_cor_nonrej = idx_cor_nonrej.shape[0]
    n_incor_rej = idx_incor_rej.shape[0]
    n_incor_nonrej = idx_incor_nonrej.shape[0]
    if show:
        print(tabulate([["", "Non-rejected", "Rejected"], ["Correct", n_cor_nonrej, n_cor_rej],
                        ["Incorrect", n_incor_nonrej, n_incor_rej]],
                       headers="firstrow"))
    return n_cor_rej, n_cor_nonrej, n_incor_rej, n_incor_nonrej


def compute_metrics_rej(threshold, y_true_label, y_pred_label, unc_ary, idx=None, relative=True,
                        show=True, seed=44):
    """Compute 3 rejection metrics using relative or absolute threshold:
    - non-rejeced accuracy (NRA)
    - classification quality (CQ)
    - rejection quality (RQ)

    Parameters
    ----------
    threshold : float
        Rejection threshold.
    y_true_label : ndarray
        1D array (`float` type) containing true labels.
    y_pred_label : ndarray
        1D array (`float` type) containing predicted labels.
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    idx : ndarray, optional
        1D array (`int` type) containing indices of test subset.
        Default: None
    relative : bool, optional
        Use relative rejection, otherwise absolute rejection.
        Default: True
    show : bool, optional
        Print confusion matrix to console.
        Default: True
    seed: int, optional
        Seed value for random rejection.
        Default 44

    Returns
    -------
    nonrej_acc : float
        Non-rejeced accuracy (NRA).
    class_quality : float
        Classification quality (CQ).
    rej_quality : float
        Rejection quality (RQ).

    Notes
    -----
    - rejection quality is undefined when `n_cor_rej=0`
        - if any observation is rejected: RQ = positive infinite
        - if no sample is rejected: RQ = 1
        - see: `Condessa et al. (2017) <https://doi.org/10.1016/j.patcog.2016.10.011>`_
    """
    if idx is not None:
        y_true_label, y_pred_label, unc_ary, *_ = subset_ary(idx, y_true_label, y_pred_label, unc_ary)
    n_cor_rej, n_cor_nonrej, n_incor_rej, n_incor_nonrej = confusion_matrix_rej(
        y_true_label, y_pred_label, unc_ary, threshold=threshold, show=show, relative=relative,
        seed=seed)

    # 3 metrics
    try:
        nonrej_acc = n_cor_nonrej / (n_incor_nonrej + n_cor_nonrej)
    except ZeroDivisionError:
        nonrej_acc = np.inf # invalid
    try:
        class_quality = (n_cor_nonrej + n_incor_rej) / \
            (n_cor_rej + n_cor_nonrej + n_incor_rej + n_incor_nonrej)
    except ZeroDivisionError:
        class_quality = np.inf # invalid
    try:
        rej_quality = (n_incor_rej / n_cor_rej) / \
            ((n_incor_rej + n_incor_nonrej) / (n_cor_rej + n_cor_nonrej))
    except ZeroDivisionError:
        if (n_incor_rej + n_cor_rej) > 0:
            rej_quality = np.inf
        else:
            rej_quality = 1.0
    if show:
        data = [[nonrej_acc, class_quality, rej_quality]]
        print("\n"+tabulate(data, headers=["Non-rejected accuracy", "Classification quality",
                                           "Rejection quality"], floatfmt=".4f"))
    return nonrej_acc, class_quality, rej_quality


def compute_count_unc(threshold, unc_ary):
    """Compute number of observations with uncertainty >= `threshold`.

    Parameters
    ----------
    threshold : float
        Rejection threshold (absolute).
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.

    Returns
    -------
    float
        Number of observations with uncertainty >= `threshold`.
    """
    count_unc = np.where(unc_ary >= threshold)[0].shape[0]
    return count_unc
