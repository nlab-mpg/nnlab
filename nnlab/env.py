#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import os
from os import path
from dotenv import load_dotenv

import numpy as np
import tensorflow as tf

HOME_PATH = os.environ.get("HOME")
load_dotenv(path.join(HOME_PATH, '.nnlab.env'))

# Float precision
FLOAT_PRECISION = os.environ.get("FLOAT_PRECISION", "32")
if int(FLOAT_PRECISION) == 16:
    TF_FLOAT_TYPE = tf.float16
    FLOAT_TYPE = "float16"
elif int(FLOAT_PRECISION) == 64:
    TF_FLOAT_TYPE = tf.float64
    FLOAT_TYPE = "float64"
else:
    TF_FLOAT_TYPE = tf.float32
    FLOAT_TYPE = "float32"

# Int precision
INT_PRECISION = os.environ.get("INT_PRECISION", "32")
if int(INT_PRECISION) == 16:
    TF_INT_TYPE = tf.int16
    INT_TYPE = "int16"
elif int(INT_PRECISION) == 64:
    TF_INT_TYPE = tf.int64
    INT_TYPE = "int64"
else:
    TF_INT_TYPE = tf.int32
    INT_TYPE = "int32"

# Random vars
if "global_numpy_rand" not in globals():
    global_numpy_rand = np.random.RandomState(int(os.environ.get("RANDOM_SEED", "3")))
