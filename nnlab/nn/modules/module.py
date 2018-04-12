#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from nnlab.env import FLOAT_TYPE
from nnlab.core.runtime import nn_runtime
from .. import functional as F

class Module(object):
    """ Base class of neural network layers."""
    
    def __init__(self, name=""):
        self._name = name
        self._parameters = []
        self.prepare()
        
    def prepare(self):
        pass
    
    def register_parameters(self, *parameters):
        """Register parameters to this module."""
        self._parameters.extend(parameters)
        
    def parameters(self):
        """Get parameters."""
        return self._parameters
    
    def new_parameter(self, name, shape=None, fill_value=None, data=None, trainable=True):
        name = "{}_{}".format(self.__class__.__name__, name)
        return F.new_parameter(name, shape, fill_value=fill_value, data=data, trainable=trainable)
