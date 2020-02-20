#!/usr/bin/env python3

"""Model to classify mugs

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 1

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 1

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different mugs.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: An input layer specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """

    # TODO: Code of your solution
    model = None

    # TODO: Return the compiled model
    return model
