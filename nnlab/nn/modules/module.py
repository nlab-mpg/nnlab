#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import tensorflow as tf

from nnlab.env import FLOAT_TYPE
from nnlab.core.runtime import nn_runtime

class Module(object):
    """ Base class of neural network layers."""
    
    def __init__(self, name=""):
        self._name = name
        self._parameters = []

    def register_parameters(self, *parameters):
        """Register parameters to this module."""
        self._parameters.extend(parameters)
        
    def parameters(self):
        """Get parameters."""
        return self._parameters
    
    def new_parameter(self, shape, data=None, name=None):
        if not name:
            name = "{}_param_{}".format(self.__class__.__name__, len(self._parameters) + 1)
        else:
            name = "{}_{}".format(self.__class__.__name__, name)
        var = tf.get_variable(name, shape, dtype=FLOAT_TYPE)
        nn_runtime.initialize_vars(var)
        return var
