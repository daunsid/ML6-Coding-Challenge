#!/usr/bin/env python3

"""Train and evaluate the model

This file trains the model upon the training data and evaluates it with
the eval data.
It uses the arguments it got via the gcloud command.
"""

import os
import argparse
import logging

import numpy as np
import tensorflow as tf

import trainer.data as data
import trainer.model as model


def train_model(params):
    """The function gets the training data from the training folder,
    the evaluation data from the eval folder and trains your solution
    from the model.py file with it.

    Parameters:
        params: parameters for training the model
    """
    (train_data, train_labels) = data.create_data_with_labels("data/train/")
    (eval_data, eval_labels) = data.create_data_with_labels("data/eval/")
    #train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=4)    
    #eval_labels = tf.keras.utils.to_categorical(eval_labels, num_classes=4)    
    
    #train_data = np.append(train_data, eval_data, axis=0)
    #train_labels = np.append(train_labels, eval_labels, axis=0)

    img_shape = train_data.shape[1:]
    input_layer = tf.keras.Input(shape=img_shape, name='input_image')

    ml_model = model.solution(input_layer)

    if ml_model is None:
        print("No model found. You need to implement one in model.py")
    else:
        ml_model.fit(train_data, train_labels,
                     batch_size=model.get_batch_size(),
                     epochs=model.get_epochs())
        ml_model.evaluate(eval_data, eval_labels, verbose=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level / 10)

    train_model(args)
