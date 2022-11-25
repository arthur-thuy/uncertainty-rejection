#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Fri November 25 2022
# =============================================================================
"""Module for datasets."""
# =============================================================================
# Imports
# =============================================================================
# standard library imports
import os
import logging
import urllib
# related third party imports
import numpy as np
# local application/library specific imports
from uncertainty_rejection.analysis import (
    load_predictions
)


def get_file(origin, fname=None, cache_dir=None):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.uncertainty_rejection`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.uncertainty_rejection/datasets/example.txt`.

    Function adapted from `keras.utils.get_file`.

    Parameters
    ----------
    origin : str
        Original URL of the file.
        Default: None
    fname : str, optional
        Name of the file. If `None`, the name of the file at `origin` will be used.
    cache_dir : str, optional
        Location to store cached files, when None it defaults to the default directory
         `~/.uncertainty_rejection/`.

    Returns
    -------
    str
        Path to the downloaded file.

    Raises
    ------
    ValueError
        If `origin` argument is not provided.
    """
    if origin is None:
        raise ValueError(
            'Please specify the "origin" argument (URL of the file '
            'to download).'
        )
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".uncertainty_rejection")
    datadir_base = os.path.expanduser(cache_dir)
    _makedirs_exist_ok(datadir_base)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".uncertainty_rejection")
    datadir = os.path.join(datadir_base, "datasets")
    _makedirs_exist_ok(datadir)
    if not fname:
        fname = os.path.basename(origin)
        if not fname:
            raise ValueError(
                "Can't parse the file name from the origin provided: "
                f"'{origin}'."
                "Please specify the `fname` as the input param."
            )
    fpath = os.path.join(datadir, fname)


    download = False
    if not os.path.exists(fpath):
        download = True
    if download:
        logging.info("Downloading data from %s", origin)

        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                urllib.request.urlretrieve(origin, fpath)
            except urllib.error.HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg)) from e
            except urllib.error.URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason)) from e
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
    else:
        logging.info("Data already exists on local disk")
    return fpath


def _makedirs_exist_ok(datadir):
    """Check if directory already exists. If not, create directory.

    Parameters
    ----------
    datadir : str
        Name of the directory.
    """
    os.makedirs(datadir, exist_ok=True)


def load_mnist_data(path="mnist.npz"):
    """Loads the MNIST dataset.

    This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
    along with a test set of 10,000 images.
    More info can be found at the
    [MNIST homepage](http://yann.lecun.com/exdb/mnist/).

    Function adapted from `keras.datasets.mnist.load_data`.

    Parameters
    ----------
    path : str, optional
        Path where to cache the dataset locally
        (relative to `~/.uncertainty_rejection/datasets`).
        Default: "mnist.npz"

    Returns
    -------
    Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    x_train : ndarray
        uint8 NumPy array of grayscale image data with shapes
        `(60000, 28, 28)`, containing the training data. Pixel values range
        from 0 to 255.
    y_train : ndarray
        uint8 NumPy array of digit labels (integers in range 0-9)
        with shape `(60000,)` for the training data.
    x_test : ndarray
        uint8 NumPy array of grayscale image data with shapes
        (10000, 28, 28), containing the test data. Pixel values range
        from 0 to 255.
    y_test : ndarray
        uint8 NumPy array of digit labels (integers in range 0-9)
        with shape `(10000,)` for the test data.

    Example
    -------
    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    ```

    License
    -------
    Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
    which is a derivative work from original NIST datasets.
    MNIST dataset is made available under the terms of the
    [Creative Commons Attribution-Share Alike 3.0 license.](
    https://creativecommons.org/licenses/by-sa/3.0/)
    """
    # get download link: https://sites.google.com/site/gdocs2direct/
    origin_folder = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/")
    path = get_file(
        origin=os.path.join(origin_folder, "mnist.npz"),
        fname=path
    )
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)

def load_notmnist_data(path="not_mnist.npz"):
    """Loads the Not-MNIST dataset.

    This is a dataset of 529,114 28x28 grayscale images of letters A-J,
    along with a test set of 18,724 images.
    More info can be found at the
    [Not-MNIST homepage](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).

    Function adapted from `keras.datasets.mnist.load_data`.

    Parameters
    ----------
    path : str, optional
        Path where to cache the dataset locally
        (relative to `~/.uncertainty_rejection/datasets`).
        Default: "not_mnist.npz"

    Returns
    -------
    Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    x_train : ndarray
        uint8 NumPy array of grayscale image data with shapes
        `(529114, 28, 28)`, containing the training data. Pixel values range
        from 0 to 255.
    y_train : ndarray
        uint32 NumPy array of digit labels (letters in range A-J)
        with shape `(529114,)` for the training data.
    x_test : ndarray
        uint8 NumPy array of grayscale image data with shapes
        (18724, 28, 28), containing the test data. Pixel values range
        from 0 to 255.
    y_test : ndarray
        uint32 NumPy array of digit labels (letters in range A-J)
        with shape `(18724,)` for the test data.

    Example
    -------
    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (529114, 28, 28)
    assert x_test.shape == (18724, 28, 28)
    assert y_train.shape == (529114,)
    assert y_test.shape == (18724,)

    License
    -------
    Yaroslav Bulatov holds the copyright of Not-MNIST dataset.
    """
    # get download link: https://sites.google.com/site/gdocs2direct/
    # add "&confirm=t" to circumvent warning about virus
    origin = (
        "https://drive.google.com/uc?export=download&id=1ZZSTT3qALwHk7UT1a9JhPpcoDgPk2xdM&confirm=t"
    )
    path = get_file(
        origin=origin,
        fname=path,
    )
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)

def load_example_predictions(path="example_preds_mnist_notmnist.npy"):
    """Loads the example predictions for the MNIST and Not-MNIST datasets.

    Parameters
    ----------
    path : str, optional
        Path where to cache the dataset locally
        (relative to `~/.uncertainty_rejection/datasets`).
        Default: "example_preds_mnist_notmnist.npy"

    Returns
    -------
    y_stack : ndarray
        3D array (`float` type) of shape `(observations, samples, classes)`.
    y_mean : ndarray
        2D array (`float` type) of shape `(observations, classes)`.
    y_label : ndarray
        1D array (`float` type) of shape `(observations,)`.
    """
    origin = (
        "https://drive.google.com/uc?export=download&id=1Ncz8_E3hMLkVOR131MOB7UBmzGurdYMd&confirm=t"
    )
    path = get_file(
        origin=origin,
        fname=path
    )
    y_stack_all, y_mean_all, y_label_all = load_predictions(path)
    return y_stack_all, y_mean_all, y_label_all
