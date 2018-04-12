#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
from six.moves import xrange, zip

import tensorflow as tf
from .tensor import Tensor

class Graph(object):
    """The class for defining computational graph."""
    
    def __init__(self, loss=None, modules=None, inputs=None, outputs=None, monitors=None):
        self._loss = loss
        self._modules = modules if modules is not None else []
        self._inputs = inputs
        self._outputs = outputs
        self._monitors = monitors
        self._check_arguments(loss, modules, inputs, outputs, monitors)
    
    def _check_arguments(self, loss, modules, inputs, outputs, monitors):
        """Verify the arguments."""
        if loss is not None and not isinstance(loss, Tensor):
            raise Exception("loss must be a tensor")
        if modules is not None and not isinstance(modules, list):
            raise Exception("modules must be a list")
        if inputs is not None and not self._check_type(inputs):
            raise Exception("input must be a tensor/list/dict")
        if outputs is not None and not self._check_type(outputs):
            raise Exception("output must be a tensor/list/dict")
        if monitors is not None and not isinstance(monitors, dict):
            raise Exception("monitors must be a dict")
        
    def _check_type(self, obj):
        """Check whether the type is either a tensor or list or dict"""
        return isinstance(obj, Tensor) or isinstance(obj, list) or isinstance(obj, dict)
    
    @property
    def loss(self):
        return self._loss
    
    @property
    def modules(self):
        return self._modules
    
    @property
    def inputs(self):
        return self._inputs
    
    
    
    
    
