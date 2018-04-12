#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
from six.moves import xrange, zip
from collections import defaultdict

import tensorflow as tf

from ..nn import functional as F

class NNRuntime(object):
    """Runtime manager for executing computational graph."""
    
    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        self._name_counter = defaultdict(int)
        # self._tf_session = tf.InteractiveSession(config=config)
    
    def run(self, tensor, feed_dict=None, options=None, run_metadata=None):
        """Run a given tensor."""
        tensor, feed_dict, options, run_metadata = F.tensor_to_tf([tensor, feed_dict, options, run_metadata])
        return self._tf_session.run(tensor, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
    
    def initialize_vars(self, *vars):
        """Initialize variables."""
        tf.variables_initializer(vars).run()
        
    def initialize_global_vars(self):
        self._tf_session.run(tf.global_variables_initializer())
    
    def get_session(self):
        """Return tensorflow session."""
        return self._tf_session
    
    def unique_name(self, name):
        self._name_counter[name] += 1
        return "{}//{}".format(name, self._name_counter[name])
        
    def close(self):
        """Close the session."""
        self._tf_session.close()
        
if "nn_runtime" not in globals():
    nn_runtime = NNRuntime()
