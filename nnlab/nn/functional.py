#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

from nnlab.env import FLOAT_TYPE, INT_TYPE
from nnlab import nn


def placeholder(dtype, shape, name=None):
    """Create a placeholder tensor. """
    placeholder = tf.placeholder(dtype, shape, name)
    return nn.Tensor(placeholder)

def placeholder_with_default(data, shape, name=None):
    """Create a placeholder with default value. """
    placeholder = tf.placeholder_with_default(data, shape, name)
    return tensor_from_tf(placeholder)

def tensor_from_tf(obj):
    """Convert tensorflow tensors in the object to nnlab tensors."""
    from nnlab.utils.conver_tensor import convert_tensor
    return convert_tensor(obj, tf_to_nn=True)

def tensor_to_tf(obj):
    """Convert nnlab tensors in the object tensorflow tensors."""
    from nnlab.utils.conver_tensor import convert_tensor
    return convert_tensor(obj, tf_to_nn=False)

def float_tensor(data, is_batch=False, name=None):
    """Create a float tensor with data."""
    data = np.array(data, FLOAT_TYPE)
    shape = list(data.shape)
    if is_batch:
        shape[0] = None
    return placeholder_with_default(data, tuple(shape), name)

def int_tensor(data, is_batch=False, name=None):
    """Create an int tensor with data."""
    data = np.array(data, INT_TYPE)
    shape = list(data.shape)
    if is_batch:
        shape[0] = None
    return placeholder_with_default(data, tuple(shape), name)
    
def new_tensor(data, is_batch=False, name=None):
    """Create a tensor with data. """
    if data.dtype.startswith("float"):
        data = np.array(data, FLOAT_TYPE)
    elif data.dtype.startswith("int"):
        data = np.array(data, INT_TYPE)
    shape = list(data.shape)
    if is_batch:
        shape[0] = None
        return placeholder_with_default(data, tuple(shape), name)
    
def predefined_tensor(name):
    """Load a predefined tensor."""
    from nnlab.utils.tensor_zoo import load_predefined_tensor
    return load_predefined_tensor(name)
