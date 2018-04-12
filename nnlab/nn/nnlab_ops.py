#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

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


def float_tensor(data, is_batch=True, is_sequence=False, name=None):
    """Create a float tensor with data."""
    data = np.array(data, FLOAT_TYPE)
    shape = list(data.shape)
    if is_batch:
        shape[0] = None
    if is_sequence:
        shape[1] = None
    return placeholder_with_default(data, tuple(shape), name)


def int_tensor(data, is_batch=True, is_sequence=False, name=None):
    """Create an int tensor with data."""
    data = np.array(data, INT_TYPE)
    shape = list(data.shape)
    if is_batch:
        shape[0] = None
    if is_sequence:
        shape[1] = None
    return placeholder_with_default(data, tuple(shape), name)


def new_tensor(data, is_batch=False, is_sequence=False, name=None):
    """Create a tensor with data. """
    if hasattr(data, "dtype"):
        if data.dtype.name.startswith("float"):
            data = np.array(data, FLOAT_TYPE)
        elif data.dtype.name.startswith("int"):
            data = np.array(data, INT_TYPE)
    else:
        data = np.array(data, FLOAT_TYPE)
    shape = list(data.shape)
    if is_batch:
        shape[0] = None
    if is_sequence:
        shape[1] = None
    return placeholder_with_default(data, tuple(shape), name)

def new_parameter(name, shape=None, fill_value=None, data=None, trainable=True):
    from ..core.runtime import nn_runtime
    name = nn_runtime.unique_name(name)
    if shape is None:
        shape = []
    if data is not None:
        var = tf.get_variable(data, dtype=FLOAT_TYPE)
    elif fill_value is not None:
        var = tf.get_variable(name, dtype=FLOAT_TYPE, initializer=tf.zeros(shape, FLOAT_TYPE) + fill_value, trainable=trainable)
        # nn_runtime.initialize_vars(var)
    else:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        var = tf.get_variable(name, shape, dtype=FLOAT_TYPE, initializer=initializer)
        # nn_runtime.initialize_vars(var)
    return var

def average_gradients(multi_grads):
    """Average multiple gradient matrices."""
    multi_grads = tensor_from_tf(multi_grads)
    average_grads = []
    for grads in zip(*multi_grads):
        concat_grads = []
        none_grad = False
        for g in grads:
            if g is None:
                none_grad = True
                break
            expanded_g = tf.expand_dims(g, 0)
            concat_grads.append(expanded_g)
        
        if none_grad:
            average_grads.append(None)
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=concat_grads)
            grad = tf.reduce_mean(grad, 0)
            average_grads.append(grad)
    return tensor_to_tf(average_grads)

def predefined_tensor(name):
    """Load a predefined tensor."""
    from nnlab.utils.tensor_zoo import load_predefined_tensor
    return load_predefined_tensor(name)


def flatten(inputs, outputs_collections=None, scope=None):
    """Flatten a tensor."""
    from tensorflow.contrib.layers import flatten
    result = flatten(inputs, outputs_collections, scope)
    return tensor_from_tf(result)


def _expand_multiple_dims(x, axes):
    x = tensor_to_tf(x)
    for i in sorted(axes):
        x = tf.expand_dims(x, axis=i, name="expand_axis_%i" % i)
    return tensor_from_tf(x)

def epsilon():
    return tensor_from_tf(tf.constant(1e-20, dtype=FLOAT_TYPE))

def dimshuffle(x, axes):
    """
    Dimshuffle for tensorflow.
    This function is taken from https://github.com/rwth-i6/returnn/
    """
    x = tensor_to_tf(x)
    with tf.name_scope("dimshuffle"):
        assert all([i == "x" or isinstance(i, int) for i in axes])
        real_axes = [i for i in axes if isinstance(i, int)]
        bc_axes = [i for (i, j) in enumerate(axes) if j == "x"]
        if x.get_shape().ndims is None:
            x_shape = tf.shape(x)
            x = tf.reshape(x, [x_shape[i] for i in range(max(real_axes) + 1)])  # will have static ndims
        assert x.get_shape().ndims is not None
    
        # First squeeze missing axes.
        i = 0
        while i < x.get_shape().ndims:
            if i not in real_axes:
                x = tf.squeeze(x, axis=i)
                real_axes = [(j if (j < i) else (j - 1)) for j in real_axes]
            else:
                i += 1
    
        # Now permute.
        assert list(sorted(real_axes)) == list(range(x.get_shape().ndims))
        if real_axes != list(range(x.get_shape().ndims)):
            x = tf.transpose(x, real_axes)
    
        # Now add broadcast dimensions.
        if bc_axes:
            x = _expand_multiple_dims(x, bc_axes)
        assert len(axes) == x.get_shape().ndims
        return tensor_from_tf(x)
