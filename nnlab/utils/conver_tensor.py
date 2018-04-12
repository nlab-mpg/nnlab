#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import tensorflow as tf

from nnlab import nn
from nnlab.utils.mapdict import MapDict

def convert_tensor(obj, tf_to_nn=True):
    """Convert any tf tensor in the object to nn tensor."""
    if type(obj) == tuple:
        return tuple(convert_tensor(list(obj), tf_to_nn))
    if type(obj) == list:
        return [convert_tensor(o, tf_to_nn) for o in obj]
    elif type(obj) == dict:
        normal_map = {}
        for key in obj:
            new_obj = convert_tensor(obj[key], tf_to_nn)
            new_key = convert_tensor(key, tf_to_nn)
            normal_map[new_key] = new_obj
        return normal_map
    elif type(obj) == MapDict:
        normal_map = {}
        for key in obj:
            new_obj = convert_tensor(obj[key], tf_to_nn)
            new_key = convert_tensor(key, tf_to_nn)
            normal_map[new_key] = new_obj
        return MapDict(normal_map)
    elif isinstance(obj, nn.Tensor):
        return obj if tf_to_nn else obj.tf
    elif isinstance(obj, tf.Tensor):
        return nn.Tensor(obj) if tf_to_nn else obj
    elif type(obj) == slice:
        args = []
        for arg in [obj.start, obj.stop, obj.step]:
            new_obj = convert_tensor(arg, tf_to_nn)
            args.append(new_obj)
        return slice(*args)
    else:
        return obj
