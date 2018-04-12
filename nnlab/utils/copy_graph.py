#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
from six.moves import xrange, zip

from copy import deepcopy

import tensorflow as tf


def copy_to_graph(org_instance, to_graph, namespace=""):
    """
    Makes a copy of the Operation/Tensor instance 'org_instance'
    for the graph 'to_graph', recursively. Therefore, all required
    structures linked to org_instance will be automatically copied.
    'copied_variables' should be a dict mapping pertinent copied variable
    names to the copied instances.

    The new instances are automatically inserted into the given 'namespace'.
    If namespace='', it is inserted into the graph's global namespace.
    However, to avoid naming conflicts, its better to provide a namespace.
    If the instance(s) happens to be a part of collection(s), they are
    are added to the appropriate collections in to_graph as well.
    For example, for collection 'C' which the instance happens to be a
    part of, given a namespace 'N', the new instance will be a part of
    'N/C' in to_graph.

    Returns the corresponding instance with respect to to_graph.
copy_graph
    TODO: Order of insertion into collections is not preserved
    """

    # The name of the new instance
    if namespace != '':
        new_name = namespace + '/' + org_instance.name
        print(new_name)
    else:
        new_name = org_instance.name


    # If an instance of the same name exists, return appropriately
    try:
        already_present = to_graph.as_graph_element(new_name,
                                                    allow_tensor=True,
                                                    allow_operation=True)
        return already_present
    except:
        pass

    # Get the collections that the new instance needs to be added to.
    # The new collections will also be a part of the given namespace.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
            if namespace == '':
                collections.append(name)
            else:
                collections.append(namespace + '/' + name)

    # Take action based on the class of the instance
    
    if isinstance(org_instance, tf.Tensor):
        
        # If its a Tensor, it is one of the outputs of the underlying
        # op. Therefore, copy the op itself and return the appropriate
        # output.
        op = org_instance.op
        new_op = copy_to_graph(op, to_graph, namespace)
        output_index = op.outputs.index(org_instance)
        new_tensor = new_op.outputs[output_index]
        # Add to collections if any
        for collection in collections:
            to_graph.add_to_collection(collection, new_tensor)
        
        return new_tensor
        
    elif isinstance(org_instance, tf.IndexedSlices):
        
        values = org_instance.values
        indices = org_instance.indices
        dense_shape = org_instance.dense_shape
        new_values = copy_to_graph(values, to_graph, namespace)
        new_indices = copy_to_graph(indices, to_graph, namespace)
        new_dense_shape = copy_to_graph(dense_shape, to_graph, namespace) if dense_shape is not None else None
        return tf.IndexedSlices(new_values, new_indices, new_dense_shape)
        
    elif isinstance(org_instance, tf.Operation):
        
        op = org_instance

        # If it has an original_op parameter, copy it
        if op._original_op is not None:
            new_original_op = copy_to_graph(op._original_op, to_graph, namespace)
        else:
            new_original_op = None

        # If it has control inputs, call this function recursively on each.
        new_control_inputs = [copy_to_graph(x, to_graph, namespace)
                              for x in op.control_inputs]

        # If it has inputs, call this function recursively on each.
        new_inputs = [copy_to_graph(x, to_graph, namespace)
                      for x in op.inputs]

        # Make a new node_def based on that of the original.
        # An instance of tensorflow.core.framework.graph_pb2.NodeDef, it
        # stores String-based info such as name, device and type of the op.
        # Unique to every Operation instance.
        new_node_def = deepcopy(op._node_def)
        # Change the name
        new_node_def.name = new_name

        # Copy the other inputs needed for initialization
        output_types = op._output_types[:]
        input_types = op._input_types[:]

        # Make a copy of the op_def too.
        # Its unique to every _type_ of Operation.
        op_def = deepcopy(op._op_def)

        # Initialize a new Operation instance
        new_op = tf.Operation(new_node_def,
                                                   to_graph,
                                                   new_inputs,
                                                   output_types,
                                                   new_control_inputs,
                                                   input_types,
                                                   new_original_op,
                                                   op_def)
        # Use Graph's hidden methods to add the op
        to_graph._add_op(new_op)
        to_graph._record_op_seen_by_control_dependencies(new_op)
        for device_function in reversed(to_graph._device_function_stack):
            new_op._set_device(device_function(new_op))
        
        return new_op
    
    else:
        raise TypeError("Could not copy instance: " + str(org_instance))


def get_copied(original, graph, namespace=""):
    """
    Get a copy of the instance 'original', present in 'graph', under
    the given 'namespace'.
    'copied_variables' is a dict mapping pertinent variable names to the
    copy instances.
    """

    # The name of the copied instance
    if namespace != '':
        new_name = namespace + '/' + original.name
    else:
        new_name = original.name
    
    return graph.as_graph_element(new_name, allow_tensor=True,
                                  allow_operation=True)
