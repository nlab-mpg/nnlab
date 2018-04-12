#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

from .module import Module
from .. import functional as F
import numpy as np

class Conv1D(Module):
    """1D convolution layer."""

    def __init__(self, channel_n, input_dim, output_dim, ):
        self._channel_n = channel_n
        self._input_dim = input_dim
        self._output_dim = output_dim
        super(Conv1D, self).__init__()
    
    def prepare(self):
        self._filter = self.new_parameter("filter", (self._channel_n, self._input_dim, self._output_dim))
    
    def forward(self, tensor, stride=1, padding="SAME"):
        return F.conv1d(tensor, self._filter, stride, padding)
    
