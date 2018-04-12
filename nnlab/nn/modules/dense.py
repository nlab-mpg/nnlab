#!/usr/bin/env python
# -*- coding: utf-8 -*-tensor

from __future__ import absolute_import, print_function, division

from .module import Module
from .. import functional as F

class Dense(Module):
    """Fully connected layer."""
    
    def __init__(self, input_dim, output_dim, use_bias=True):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._use_bias = use_bias
        super(Dense, self).__init__()
    
    def prepare(self):
        self._w = self.new_parameter("W", (self._input_dim, self._output_dim))
        if self._use_bias:
            self._b = self.new_parameter("B", (self._output_dim,), fill_value=0)
        else:
            self._b = None
    
    def forward(self, tensor):
        shape = tensor.get_shape().as_list()
        if len(shape) > 2:
            outputs = F.tensordot(tensor, self._w, [[len(shape) - 1], [0]])
            output_shape = shape[:-1] + [self._output_dim]
            outputs.set_shape(output_shape)
        else:
            outputs = F.matmul(tensor, self._w)
        if self._use_bias:
            outputs = F.bias_add(outputs, self._b)
        return F.tensor_from_tf(outputs)
