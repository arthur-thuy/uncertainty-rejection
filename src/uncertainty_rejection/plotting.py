#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Tur November 24 2022
# =============================================================================
"""Module for analysis visualization."""
# =============================================================================
# Imports
# =============================================================================
# standard library imports

# related third party imports
import numpy as np
import matplotlib.pyplot as plt

# local application/library specific imports
from uncertainty_rejection.analysis import (
    compute_count_unc,
    compute_metrics_rej,
    get_y_mean_label,
)

from uncertainty_rejection.utils import (
    kwargs_to_dict,
    subset_ary
)


def hist_unc_base(unc_ary, bins=20, ax=None, xlim=None, vline=True, hist_kwargs=None,
                  axvline_kwargs=None):
    """Plot histogram of an uncertainty metric and return Axes object.

    Parameters
    ----------
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    bins : int, optional
        Number of bins.
        Default: 20
    ax : Axes, optional
        Matplotlib Axes object.
        Default: None
    xlim : (float, float), optional
        Sequence of lower and upper limit.
        Default: None
    vline : bool, optional
        Whether to plot vertical line at mean value.
        Default: True
    hist_kwargs : dict, optional
        Histogram properties.
        Default: None
    axvline_kwargs : dict, optional
        Vertical line properties.
        Default: None

    Returns
    -------
    Axes
        Matplotlib Axes object.
    """
    if ax is None:
        ax = plt.gca()
    hist_kwargs, axvline_kwargs, *_ = kwargs_to_dict(hist_kwargs, axvline_kwargs)
    if xlim is not None:
        range_x = xlim[1] - xlim[0]
        binwidth = (xlim[1] - xlim[0]) / bins
        bins = np.arange(xlim[0]-0.05*range_x, xlim[1] + 0.05*range_x, binwidth) #  + binwidth
    ax.hist(unc_ary, bins=bins, **hist_kwargs)
    if vline:
        ax.axvline(x=np.mean(unc_ary), color="red",
                   linestyle="--", linewidth=3, **axvline_kwargs)
    return ax


def hist_unc_plot1(unc_ary, unc_type, idx=None, bins=20, ax=None, num_classes=None,
                bars_scale=False, save=False, hist_kwargs=None, axvline_kwargs=None,
                savefig_kwargs=None):
    """Plot histogram of some uncertainty metric and return plot.

    Parameters
    ----------
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    idx : ndarray, optional
        1D array (`int` type) containing indices of test subset.
        Default: None
    bins : int, optional
        Number of bins.
        Default: 20
    ax : Axes, optional
        Matplotlib Axes object.
        Default: None
    unc_type : {'TU', 'AU', 'EU', 'Conf'}, optional
        Type of uncertainty values.
    num_classes : int, optional
        Number of output classes. Used to adjust the xlim for entropy-based uncertainties.
        Default: None
    bars_scale : bool, optional
        Whether to adjust bar width to full xlim.
        Default: False
    save : bool, optional
        Whether to save the figure.
        Default: False
    hist_kwargs : dict, optional
        Histogram properties.
        Default: None
    axvline_kwargs : dict, optional
        Vertical line properties.
        Default: None
    savefig_kwargs : dict, optional
        Figure saving properties.
        Default: None
    """
    unc_types = ['TU', 'AU', 'EU', 'Conf']
    if unc_type not in unc_types:
        raise ValueError("Invalid uncertainty type. Expected one of: %s" % unc_types)
    if (unc_type in ['TU', 'AU', 'EU']) and (num_classes is None):
        raise ValueError("`num_classes` argument is required for entropy-based uncertainties.")

    if idx is not None:
        unc_ary, *_ = subset_ary(idx, unc_ary)
    hist_kwargs, axvline_kwargs, savefig_kwargs, *_ = kwargs_to_dict(hist_kwargs, axvline_kwargs,
                                                                     savefig_kwargs)
    x_label_dict = {"TU": "Total uncertainty", "AU": "Aleatoric uncertainty",
    "EU": "Epistemic uncertainty", "Conf": "Confidence"}
    if unc_type == 'Conf':
        xlim = (0, 1)
    elif (unc_type in ['TU', 'AU', 'EU']) and (num_classes is not None):
        xlim = (0, np.log2(num_classes))
    if bars_scale:
        xlim_base = xlim
    else:
        xlim_base = None
    plt.figure()
    out_ax = hist_unc_base(unc_ary, bins=bins, ax=ax, xlim=xlim_base, hist_kwargs=hist_kwargs,
                           axvline_kwargs=axvline_kwargs)

    out_ax.set(xlabel=x_label_dict[unc_type], ylabel='Frequency')
    if xlim is not None:
        range_x = xlim[1] - xlim[0]
        out_ax.set_xlim((xlim[0]-0.05*range_x, xlim[1]+0.05*range_x))
    plt.grid(linestyle="dashed")

    if save:
        plt.tight_layout()
        plt.savefig(**savefig_kwargs)
    return out_ax


def hist_unc_plot3(unc_tot, unc_ale, unc_epi, num_classes, idx=None, bins=20, bars_scale=False,
            save=False, hist_kwargs=None, axvline_kwargs=None, savefig_kwargs=None):
    """Plot uncertainty hist for 3 types of uncertainty and return plot.

    Parameters
    ----------
    unc_tot : ndarray
        1D ndarray (`float` type) containing total uncertainty values.
    unc_ale : ndarray
        1D ndarray (`float` type) containing aleatoric uncertainty values.
    unc_epi : ndarray
        1D ndarray (`float` type) containing epistemic uncertainty values.
    idx : ndarray, optional
        1D array (`int` type) containing indices of test subset.
        Default: None
    bins : int, optional
        Number of bins.
        Default: 20
    num_classes : int
        Number of output classes (integer > 0). Used to adjust the xlim for entropy-based uncertainties.
    bars_scale : bool, optional
        Whether to adjust bar width to full xlim.
        Default: False
    save : bool, optional
        Whether to save the figure.
        Default: False
    hist_kwargs : dict, optional
        Histogram properties.
        Default: None
    axvline_kwargs : dict, optional
        Vertical line properties.
        Default: None
    savefig_kwargs : dict, optional
        Figure saving properties.
        Default: None
    """
    if idx is not None:
        unc_tot, unc_ale, unc_epi, *_ = subset_ary(idx, unc_tot, unc_ale, unc_epi)
    if isinstance(num_classes, int) and (num_classes <= 0):
        raise ValueError("`num_classes` should be an integer > 0.")
    hist_kwargs, axvline_kwargs, savefig_kwargs, *_ = kwargs_to_dict(hist_kwargs, axvline_kwargs,
                                                                     savefig_kwargs)
    if num_classes is not None:
        xlim = (0, np.log2(num_classes))
    if bars_scale:
        xlim_base = xlim
    else:
        xlim_base = None
    _, axes = plt.subplots(ncols=3, figsize=(20, 4))
    hist_unc_base(unc_tot, ax=axes[0], bins=bins, xlim=xlim_base, hist_kwargs=hist_kwargs,
                  axvline_kwargs=axvline_kwargs)
    hist_unc_base(unc_ale, ax=axes[1], bins=bins, xlim=xlim_base, hist_kwargs=hist_kwargs,
                  axvline_kwargs=axvline_kwargs)
    hist_unc_base(unc_epi, ax=axes[2], bins=bins, xlim=xlim_base, hist_kwargs=hist_kwargs,
                  axvline_kwargs=axvline_kwargs)
    xlabels = ["Total uncertainty",
               "Aleatoric uncertainty", "Epistemic uncertainty"]
    for i, ax in enumerate(axes):
        ax.set(xlabel=xlabels[i], ylabel='Frequency')
        if xlim is not None:
            range_x = xlim[1] - xlim[0]
            ax.set_xlim((xlim[0]-0.05*range_x, xlim[1]+0.05*range_x))
        ax.grid(linestyle="dashed")
    if save:
        plt.tight_layout()
        plt.savefig(**savefig_kwargs)
    return axes


def count_unc_base(unc_ary, space_bins=20, ax=None, **plt_kwargs):
    """Plot count vs uncertainty and return axis.

    Parameters
    ----------
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    space_bins : int, optional
        Number of evaluation points in the line plot.
        Default: 20
    ax : Axes, optional
        Matplotlib Axes object.
        Default: None

    Returns
    -------
    Axes
        Matplotlib Axes object.
    """
    threshold_ary = np.linspace(start=0, stop=np.max(unc_ary), num=space_bins)
    compute_count_unc_v = np.vectorize(
        compute_count_unc, excluded=["unc_ary"])
    count_unc = compute_count_unc_v(threshold_ary, unc_ary=unc_ary)

    # plot on existing axis or new axis
    if ax is None:
        ax = plt.gca()
    ax.plot(threshold_ary, count_unc, **plt_kwargs)
    return ax


def count_unc_plot1(unc_ary, unc_type, idx=None, space_bins=20, save=False, plt_kwargs=None,
                    savefig_kwargs=None):
    """Plot count vs confidence and return plot.

    Parameters
    ----------
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    unc_type : {'TU', 'AU', 'EU', 'Conf'}
        Type of uncertainty values.
    idx : ndarray, optional
        1D array (`int` type) containing indices of test subset.
        Default: None
    space_bins : int, optional
        Number of evaluation points in the line plot.
        Default: 20
    save : bool, optional
        Whether to save the figure.
        Default: False
    hist_kwargs : dict, optional
        Histogram properties.
        Default: None
    savefig_kwargs : dict, optional
        Figure saving properties.
        Default: None

    Raises
    ------
    ValueError
        If `unc_type` is invalid.
    """
    if idx is not None:
        unc_ary, *_ = subset_ary(idx, unc_ary)
    plt_kwargs, savefig_kwargs, *_ = kwargs_to_dict(plt_kwargs, savefig_kwargs)
    unc_types = ['TU', 'AU', 'EU', 'Conf']
    if unc_type not in unc_types:
        raise ValueError(
            "Invalid uncertainty type. Expected one of: %s" % unc_types)
    plt.figure()
    out_ax = count_unc_base(unc_ary, space_bins, **plt_kwargs)
    out_ax.set(title=f'Count vs {unc_type}', xlabel=f'Uncertainty {unc_type} u',
               ylabel=f'Number of observations {unc_type} >= u', ylim=(0, None))
    plt.grid(linestyle="dashed")
    if save:
        plt.tight_layout()
        plt.savefig(**savefig_kwargs)
    return out_ax


def rejection_base(y_true_label, y_pred_stack, unc_ary, metric, unc_type, relative=True, seed=44,
                   space_start=0.001, space_stop=0.99, space_bins=100, ax=None, **plt_kwargs):
    """Plot 3 metrics for varying rejection percentage and return axis.

    Parameters
    ----------
    y_true_label : ndarray
        1D array (`float` type) containing true labels.
    y_pred_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    metric : {'nra', 'cq', 'rq'}
        Metric to calculate.
    unc_type : {'TU', 'AU', 'EU', 'Conf'}
        Type of uncertainty values.
    relative : bool, optional
        Whether to use a relative or absolute threshold.
        Default: True
    seed: int, optional
        Seed value for random rejection.
        Default 44
    space_start : float, optional
        At which threshold value to start figure.
        Default: 0.001
    space_stop : float, optional
        At which threshold value to stop figure.
        Default: 0.99
    space_bins : int, optional
        Number of evaluation points in the line plot.
        Default: 100
    ax : Axes, optional
        Matplotlib Axes object.
        Default: None

    Returns
    -------
    Axes
        Matplotlib Axes object.

    Raises
    ------
    ValueError
        If `unc_type` is invalid.
    """
    unc_types = ['TU', 'AU', 'EU', 'Conf']
    if unc_type not in unc_types:
        raise ValueError(
            "Invalid uncertainty type. Expected one of: %s" % unc_types)
    _, y_pred_label = get_y_mean_label(y_pred_stack)
    if relative:
        treshold_ary = np.linspace(start=space_start, stop=space_stop, num=space_bins)
        reject_ary = treshold_ary
        plot_ary = treshold_ary
    elif not relative and unc_type == 'Conf':
        treshold_ary = np.linspace(start=(1-space_start), stop=(1-space_stop), num=space_bins)
        reject_ary = treshold_ary
        plot_ary = np.flip(treshold_ary, 0)
    elif not relative and unc_type in ['TU', 'AU', 'EU']:
        max_entropy = np.log2(y_pred_stack.shape[-1])  # equal to range
        treshold_ary = np.linspace(start=(1-space_start)*max_entropy, stop=(1-space_stop)*max_entropy, num=space_bins)
        reject_ary = treshold_ary
        plot_ary = treshold_ary

    compute_metrics_rej_v = np.vectorize(compute_metrics_rej, excluded=["y_true_label",
                                                                        "y_pred_label", "unc_ary",
                                                                        "show", "relative", "seed"])
    nonrej_acc, class_quality, rej_quality = compute_metrics_rej_v(
        reject_ary, y_true_label=y_true_label, y_pred_label=y_pred_label, unc_ary=unc_ary,
        show=False, relative=relative, seed=seed)

    # plot on existing axis or new axis
    if ax is None:
        ax = plt.gca()
    if metric == "nra":
        ax.plot(plot_ary, nonrej_acc, **plt_kwargs)
    elif metric == "cq":
        ax.plot(plot_ary, class_quality, **plt_kwargs)
    elif metric == "rq":
        ax.plot(plot_ary, rej_quality, **plt_kwargs)

    if not relative and unc_type in ['TU', 'AU', 'EU']:
        ax.set_xlim(max_entropy+0.05*max_entropy, 0-0.05*max_entropy)
    return ax


def rejection_setmetric_plot1(y_true_label, y_pred_stack, unc_ary, metric, unc_type, idx=None,
                              relative=True, seed=44, space_start=0.001, space_stop=0.99,
                              space_bins=100, save=False, savefig_kwargs=None, plt_kwargs=None):
    """Plot 1 metric based on 1 uncertainties for varying rejection percentage and return 1 plot.

    Parameters
    ----------
    y_true_label : ndarray
        1D array (`float` type) containing true labels.
    y_pred_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    metric : {'nra', 'cq', 'rq'}
        Metric to calculate.
    unc_type : {'TU', 'AU', 'EU', 'Conf'}
        Type of uncertainty values.
    idx : ndarray, optional
        1D array (`int` type) containing indices of test subset.
        Default: None
    relative : bool, optional
        Whether to use a relative or absolute threshold.
        Default: True
    seed: int, optional
        Seed value for random rejection.
        Default 44
    space_start : float, optional
        At which threshold value to start figure.
        Default: 0.001
    space_stop : float, optional
        At which threshold value to stop figure.
        Default: 0.99
    space_bins : int, optional
        Number of evaluation points in the line plot.
        Default: 100
    save : bool, optional
        Whether to save the figure.
        Default: False
    plt_kwargs : dict, optional
        Histogram properties.
        Default: None
    savefig_kwargs : dict, optional
        Figure saving properties.
        Default: None

    Raises
    ------
    ValueError
        If `unc_type` is invalid.
    """
    unc_types = ['TU', 'AU', 'EU', 'Conf']
    if unc_type not in unc_types:
        raise ValueError("Invalid uncertainty type. Expected one of: %s" % unc_types)
    if unc_type == 'Conf':
        unc_ary = (1. - unc_ary)
    if idx is not None:
        y_true_label, y_pred_stack, unc_ary, *_ = subset_ary(idx, y_true_label, y_pred_stack, unc_ary)
    plt_kwargs, savefig_kwargs, *_ = kwargs_to_dict(plt_kwargs, savefig_kwargs)
    out_ax = rejection_base(y_true_label, y_pred_stack, unc_ary, metric, unc_type, relative,
                            seed, space_start, space_stop, space_bins, **plt_kwargs)
    out_ax.grid(linestyle="dashed")
    if relative:
        out_ax.set(xlabel='Relative threshold', ylabel='Metric')
    else:
        out_ax.set(xlabel='Absolute threshold', ylabel='Metric')
    if save:
        plt.tight_layout()
        plt.savefig(**savefig_kwargs)
    return out_ax

def rejection_setmetric_plot3(y_true_label, y_pred_stack, unc_tot, unc_ale, unc_epi,
                              metric, unc_type, idx=None, relative=True, seed=44, space_start=0.001,
                              space_stop=0.99, space_bins=100, save=False, savefig_kwargs=None,
                              plt_kwargs=None):
    """Plot 1 metric based on 3 uncertainties for varying rejection percentage and return 3 plots.

    Parameters
    ----------
    y_true_label : ndarray
        1D array (`float` type) containing true labels.
    y_pred_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.
    unc_tot : ndarray
        1D ndarray (`float` type) containing total uncertainty values.
    unc_ale : ndarray
        1D ndarray (`float` type) containing aleatoric uncertainty values.
    unc_epi : ndarray
        1D ndarray (`float` type) containing epistemic uncertainty values.
    metric : {'nra', 'cq', 'rq'}
        Metric to calculate.
    unc_type : {'TU', 'AU', 'EU', 'Conf'}
        Type of uncertainty values.
    idx : ndarray, optional
        1D array (`int` type) containing indices of test subset.
        Default: None
    relative : bool, optional
        Whether to use a relative or absolute threshold.
        Default: True
    seed: int, optional
        Seed value for random rejection.
        Default 44
    space_start : float, optional
        At which threshold value to start figure.
        Default: 0.001
    space_stop : float, optional
        At which threshold value to stop figure.
        Default: 0.99
    space_bins : int, optional
        Number of evaluation points in the line plot.
        Default: 100
    save : bool, optional
        Whether to save the figure.
        Default: False
    plt_kwargs : dict, optional
        Histogram properties.
        Default: None
    savefig_kwargs : dict, optional
        Figure saving properties.
        Default: None

    Raises
    ------
    ValueError
        If `unc_type` is invalid.
    """
    unc_types = ['TU', 'AU', 'EU', 'Conf']
    if unc_type not in unc_types:
        raise ValueError("Invalid uncertainty type. Expected one of: %s" % unc_types)
    if unc_type == 'Conf':
        unc_ary = (1. - unc_ary)
    if idx is not None:
        y_true_label, y_pred_stack, unc_tot, unc_ale, unc_epi, *_ = subset_ary(idx, y_true_label,
                                                                            y_pred_stack, unc_tot,
                                                                            unc_ale, unc_epi)

    plt_kwargs, savefig_kwargs, *_ = kwargs_to_dict(plt_kwargs, savefig_kwargs)
    _, axes = plt.subplots(ncols=3, figsize=(20, 4))
    rejection_base(y_true_label, y_pred_stack, unc_tot, metric, unc_type, relative, seed, space_start,
    space_stop, space_bins, ax=axes[0], **plt_kwargs)
    rejection_base(y_true_label, y_pred_stack, unc_ale, metric, unc_type, relative, seed, space_start,
    space_stop, space_bins, ax=axes[1], **plt_kwargs)
    rejection_base(y_true_label, y_pred_stack, unc_epi, metric, unc_type, relative, seed, space_start,
    space_stop, space_bins, ax=axes[2], **plt_kwargs)
    titles = ["Total uncertainty",
              "Aleatoric uncertainty", "Epistemic uncertainty"]
    for i, ax in enumerate(axes):
        ax.grid(linestyle="dashed")
        if relative:
            ax.set(xlabel='Relative threshold',
                   ylabel='Metric', title=titles[i])
        else:
            ax.set(xlabel='Absolute threshold',
                   ylabel='Metric', title=titles[i])
    if save:
        plt.tight_layout()
        plt.savefig(**savefig_kwargs)
    return axes


def rejection_mixmetric_plot3(y_true_label, y_pred_stack, unc_ary, unc_type, idx=None,
                              relative=True, seed=44, space_start=0.001, space_stop=0.99, space_bins=100,
                              save=False, savefig_kwargs=None, plt_kwargs=None):
    """Plot 3 metrics for varying rejection percentage and return 3 plots.

    Parameters
    ----------
    y_true_label : ndarray
        1D array (`float` type) containing true labels.
    y_pred_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.
    unc_ary : ndarray
        1D ndarray (`float` type) containing uncertainty values.
    unc_type : {'TU', 'AU', 'EU', 'Conf'}
        Type of uncertainty values.
    idx : ndarray, optional
        1D array (`int` type) containing indices of test subset.
        Default: None
    relative : bool, optional
        Whether to use a relative or absolute threshold.
        Default: True
    seed: int, optional
        Seed value for random rejection.
        Default 44
    space_start : float, optional
        At which threshold value to start figure.
        Default: 0.001
    space_stop : float, optional
        At which threshold value to stop figure.
        Default: 0.99
    space_bins : int, optional
        Number of evaluation points in the line plot.
        Default: 100
    save : bool, optional
        Whether to save the figure.
        Default: False
    plt_kwargs : dict, optional
        Histogram properties.
        Default: None
    savefig_kwargs : dict, optional
        Figure saving properties.
        Default: None

    Raises
    ------
    ValueError
        If `unc_type` is invalid.
    """
    unc_types = ['TU', 'AU', 'EU', 'Conf']
    if unc_type not in unc_types:
        raise ValueError(
            "Invalid uncertainty type. Expected one of: %s" % unc_types)
    if unc_type == 'Conf':
        unc_ary = (1. - unc_ary)
    if idx is not None:
        y_true_label, y_pred_stack, unc_ary, *_ = subset_ary(idx, y_true_label, y_pred_stack, unc_ary)
    plt_kwargs, savefig_kwargs, *_ = kwargs_to_dict(plt_kwargs, savefig_kwargs)
    label_dict = {"nra": "Non-rejected accuracy",
                  "cq": "Classification quality", "rq": "Rejection quality"}
    _, axes = plt.subplots(ncols=3, figsize=(20, 4))
    rejection_base(y_true_label, y_pred_stack, unc_ary, list(label_dict.keys())[0], unc_type,
                   relative, seed, space_start, space_stop, space_bins, ax=axes[0], **plt_kwargs)
    rejection_base(y_true_label, y_pred_stack, unc_ary, list(label_dict.keys())[1], unc_type,
                   relative, seed, space_start, space_stop, space_bins, ax=axes[1], **plt_kwargs)
    rejection_base(y_true_label, y_pred_stack, unc_ary, list(label_dict.keys())[2], unc_type,
                   relative, seed, space_start, space_stop, space_bins, ax=axes[2], **plt_kwargs)
    titles = list(label_dict.values())
    for i, ax in enumerate(axes):
        ax.set(xlabel='Rejection', ylabel='Metric', title=titles[i])
        ax.grid(linestyle="dashed")
    if save:
        plt.tight_layout()
        plt.savefig(**savefig_kwargs)
    return axes
