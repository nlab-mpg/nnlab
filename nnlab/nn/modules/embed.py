#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function


import tensorflow as tf

from nnlab import nn
from .module import Module
from .. import functional as F

class Embedding(Module):
    """Word embedding module."""
    
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self._vocab_size = vocab_size
        self._embed_size = embed_dim
        self._prepare()
    
    def _prepare(self):
        self._weight = self.new_parameter("embedding", (self._vocab_size, self._embed_size))
    
    def get(self, token_ids):
        """Get the embeddings for given token ids"""
        assert isinstance(token_ids, nn.Tensor)
        tf_embed = tf.nn.embedding_lookup(self._weight, token_ids.tf)
        return F.tensor_from_tf(tf_embed)
    
    @property
    def weight_matrix(self):
        """Return the weight matrix of embeddings."""
        return self._weight
