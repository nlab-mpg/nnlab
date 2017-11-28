#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
from functools import partial

import tensorflow as tf

from . import functional as F


class Tensor(tf.Tensor):
    """Basic tensors in the computational graph.
    
    A wrapping of tensorflow tensor to provide extra functions.
    """
    
    def __init__(self, tftensor):
        self._tftensor = tftensor

    def __getattr__(self, name):
        return Tensor(getattr(self._tftensor, name))

    def apply(self, func):
        """
        Apply a function to tensors.
        """
        return Tensor(func(self._tftensor))

    def set_test_value(self, value):
        self._tftensor.tag.test_value = value

    def dim(self):
        return self.output_dim

    def _other_tf(self, other):
        return other.tensor if isinstance(other, Tensor) else other

    def __getitem__(self, index):
        return Tensor(self._tftensor[index])

    def __call__(self, *args, **kwargs):
        args = F.tensor_to_tf(args)
        kwargs = F.tensor_to_tf(kwargs)
        
        return F.tensor_from_tf(self._tftensor(*args, **kwargs))

    def monitor(self, name=""):
        from deepy.debug.monitor import monitor_tensor
        self._tftensor += monitor_tensor(self._tftensor, name=name)
        return self
    
    @property
    def tf(self):
        return self._tftensor

    @property
    def test_value(self):
        if hasattr(self._tftensor.tag, 'test_value'):
            return self._tftensor.tag.test_value
        else:
            return None

    @property
    def last_dim(self):
        return self._tftensor.shape[-1]

    def __str__(self):
        pp = "tensor: {} ~ {}\n".format(str(self._tftensor.dtype.name), str(self._tftensor.shape))
        pp += str(self._tftensor.eval())
        return pp

    def __repr__(self):
        return self.__str__()

# Overload operators
def op_func(op_str, self, *args, **kwargs):
    args = F.tensor_to_tf(args)
    kwargs = F.tensor_to_tf(kwargs)
    tf_func = getattr(self._tftensor, op_str, None)
    if tf_func is None:
        raise SystemError("Tensorflow tensor does not have operation: {}".format(op_str))
    return F.tensor_from_tf(tf_func(*args, **kwargs))
    
for op_str in tf.Tensor.OVERLOADABLE_OPERATORS:
    if hasattr(Tensor, op_str):
        continue
    overload_func = partial(op_func, op_str)
    setattr(Tensor, op_str, overload_func)
