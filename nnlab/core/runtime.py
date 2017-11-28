#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
from six.moves import xrange, zip

import tensorflow as tf

from ..nn import functional as F

class NNRuntime(object):
    """Runtime manager for executing computational graph."""
    
    def __init__(self):
        self._tf_session = tf.InteractiveSession()
    
    def run(self, tensor):
        """Run a given tensor."""
        tensor = F.tensor_to_tf(tensor)
        return self._tf_session.run(tensor)
    
    def initialize_vars(self, *vars):
        """Initialize variables."""
        tf.variables_initializer(vars).run()
    
    def get_session(self):
        """Return tensorflow session."""
        return self._tf_session
    
    def close(self):
        """Close the session."""
        self._tf_session.close()
        
if "nn_runtime" not in globals():
    nn_runtime = NNRuntime()
