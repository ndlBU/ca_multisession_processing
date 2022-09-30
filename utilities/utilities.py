#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Created on 08-Sep-2022 at 3:21 PM

# @author: jad
"""

import h5py
import logging
import numpy as np
from scipy.sparse import csc_matrix
import numbers


def save_dict_to_hdf5(dic, filename):
    """
    Saves a dictionary to a hdf5 file
    """
    with h5py.File(filename, 'w') as h5file:
        _recursively_save_dict_contents_to_group(h5file, '/', dic)


def _recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float, tuple)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            logging.debug('Saving {}'.format(key))
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif isinstance(item, list):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', dict(zip([str(i) for i in range(len(item))], item)))
        elif isinstance(item, csc_matrix):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', csc_to_dict(item))
        else:
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(filename):
    """
    Loads a dictionary from a hdf5 file
    """
    with h5py.File(filename, 'r') as h5file:
        return _recursively_load_dict_contents_from_group(h5file, '/')


def _recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def dict_to_csc(x):
    """
    Generates a csc matrix from its dictionary representation
    :param x:
    :return:
    """
    A = csc_matrix(np.empty(x['shape']))
    A.data = x['data'].copy()
    A.indices = x['indices'].copy()
    A.indptr = x['indptr'].copy()

    return A


def csc_to_dict(x: csc_matrix):
    """
    Generates a dictionary representation from a csc matrix
    :param x: csc_matrix
    :return: A: dictionary that allows reconstruction of csc_matrix
    """
    A = {'shape': x.shape,
         'data': x.data,
         'indices': x.indices,
         'indptr': x.indptr
         }
    return A


def old_div(a, b):
    """
    DEPRECATED: import ``old_div`` from ``past.utils`` instead.
    Equivalent to ``a / b`` on Python 2 without ``from __future__ import
    division``.
    """
    if isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
        return a // b
    else:
        return a / b
