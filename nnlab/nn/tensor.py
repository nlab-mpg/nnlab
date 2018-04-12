#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import tensorflow as tf
from tensorflow.python.framework import dtypes

from . import functional as F

class Tensor(tf.Tensor):
    """Basic tensors in the computational graph.
    
    A wrapping of tensorflow tensor to provide extra functions.
    """
    
    def __init__(self, tftensor):
        self._tftensor = tftensor

    def apply(self, func):
        """
        Apply a function to tensors.
        """
        return Tensor(func(self._tftensor))
    
    def sum(self, axis=None,
               keepdims=None,
               name=None,
               reduction_indices=None):
        return F.reduce_sum(self, axis, keepdims, name, reduction_indices)
    
    def reshape(self, shape, name=None):
        return F.reshape(self, shape, name)
    
    def transpose(self, perm=None, name="transpose"):
        return F.transpose(self, perm, name)
    
    def expand_dims(self, axis=None, name=None, dim=None):
        return F.expand_dims(self, axis, name, dim)
    
    def squeeze(self, axis=None, name=None, squeeze_dims=None):
        return F.squeeze(self, axis, name, squeeze_dims)
    
    def dimshuffle(self, axes):
        return F.dimshuffle(self, axes)
    
    def max(self, axis=None, keep_dims=False,
               name=None, reduction_indices=None):
        return F.reduce_max(self, axis, keep_dims, name, reduction_indices)
    
    def argmax(self, axis=None, name=None, dimension=None, output_type=dtypes.int64):
        return F.argmax(self, axis=axis, name=name, output_type=output_type)
    
    def min(self, axis=None, keep_dims=False, name=None, reduction_indices=None):
        return F.reduce_min(self, axis, keep_dims, name, reduction_indices)
    
    def argmin(self, axis=None, name=None, output_type=dtypes.int64):
        return F.argmin(self, axis=axis, name=name, output_type=output_type)
    
    def mean(self, axis=None, keep_dims=False, name=None, reduction_indices=None):
        return F.reduce_mean(self, axis=axis, keep_dims=keep_dims, name=name, reduction_indices=reduction_indices)
    
    def dot(self, other):
        return F.matmul(self, other)
    
    def flatten(self):
        return F.flatten(self)
    
    def _other_tf(self, other):
            return other.tensor if isinstance(other, Tensor) else other
        
    def __getitem__(self, index):
        return Tensor(self._tftensor[index])

    def __call__(self, *args, **kwargs):
        args = F.tensor_to_tf(args)
        kwargs = F.tensor_to_tf(kwargs)
        return F.tensor_from_tf(self._tftensor(*args, **kwargs))
    
    def __getattr__(self, item):
        return getattr(self._tftensor, item, None)

    def __hash__(self):
        return id(self._tftensor)
    
    def __eq__(self, other):
        return F.equal(self, other)
    
    def __ne__(self, other):
        return F.not_equal(self, other)
    
    def get_shape(self):
        return self._tftensor.get_shape()
    
    @property
    def shape(self):
        return tf.shape(self._tftensor)
    
    @property
    def T(self):
        return self.transpose()
    
    @property
    def last_dim(self):
        return self._tftensor.shape[-1]
    
    @property
    def tf(self):
        return self._tftensor

    def __str__(self):
        pp = "tensor: {} ~ {}\n".format(str(self._tftensor.dtype.name), str(self._tftensor.shape))
        pp += str(self._tftensor.eval())
        return pp

    def __repr__(self):
        return self.__str__()

# Overload operators
def op_func(op_str, tensor, *args, **kwargs):
    args = F.tensor_to_tf(args)
    kwargs = F.tensor_to_tf(kwargs)
    tf_func = getattr(tensor.tf, op_str, None)
    if tf_func is None:
        raise SystemError("Tensorflow tensor does not have operation: {}".format(op_str))

    return F.tensor_from_tf(tf_func(*args, **kwargs))
for op_str in tf.Tensor.OVERLOADABLE_OPERATORS:
    exec("""def WRAP_OP_STR(tensor, *args, **kwargs):
                return op_func("OP_STR", tensor, *args, **kwargs)
         """.replace("OP_STR", op_str))
    setattr(Tensor, op_str, globals()["WRAP_{}".format(op_str)])
