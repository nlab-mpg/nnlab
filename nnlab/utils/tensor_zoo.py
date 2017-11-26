#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import numpy as np

from nnlab.env import global_numpy_rand
import nnlab.nn.functional as F

def load_predefined_tensor(template_name):
    """Load a predefined tensor.
    
    This is useful to create computational graph without real data.
    """
    
    if template_name == 'sequence_2d':
        data = np.array([[1,2,3,4], [1,2,3,0], [1,2,0,0]])
        return F.int_tensor(data, is_batch=True, name=template_name)
    else:
        raise NotImplementedError
