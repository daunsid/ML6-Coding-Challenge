#!/usr/bin/env python3

"""Train and export the model

This file trains the model upon all data with the arguments it got via
the gcloud command.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import trainer.data as data
import trainer.model as model

def prepare_prediction_image(image_str_tensor):
    """Prepare an image tensor for prediction.
    Takes a string tensor containing a binary jpeg image and returns
    a tensor object of the image with dtype float32.

    Parameters:
        image_str_tensor: a tensor containing a binary jpeg image as a string
    Returns:
        image: A tensor representing an image.
    """
    image_str_tensor = tf.cast(image_str_tensor, tf.string)
    image = tf.image.decode_jpeg(image_str_tensor, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = data.preprocess_image(image)
    return image

def prepare_prediction_image_batch(image_str_tensor):
    """Prepare a batch of images for prediction."""
    return tf.map_fn(prepare_prediction_image, image_str_tensor,
                     dtype=tf.float32)

def export_model(ml_model, export_dir, model_dir='exported_model'):
    """Prepare model for strings representing image data and export it.

    Before the model is exported the initial layers of the model need to be
    adapted to handle prediction of images contained in JSON files.

    Parameters:
        ml_model: A compiled model
        export_dir: A string specifying the
        model_dir: A string specifying the name of the directory to
            which the model is written.
    """
    optimizer = ml_model.optimizer
    loss = ml_model.loss_functions
    metrics = ml_model.metrics

    ml_model.layers.pop(0)
    prediction_input = tf.keras.Input(
        dtype=tf.string, name='prediction_image', shape=())
    prediction_output = tf.keras.layers.Lambda(
        prepare_prediction_image_batch)(prediction_input)
    prediction_output = ml_model(prediction_output)
    ml_model = tf.keras.models.Model(prediction_input, prediction_output)
    weights = ml_model.get_weights()

    ml_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_path = Path(export_dir) / model_dir
    if model_path.exists():
        timestamp = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
        model_path = Path(str(model_path) + timestamp)

    with tf.compat.v1.Session() as sess:
        init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                           tf.compat.v1.local_variables_initializer())
        sess.run(init_op)
        ml_model.set_weights(weights)

        inputs = {"bytes": ml_model.input}
        outputs = {
            "CLASSES": tf.argmax(ml_model.output, 1),
            "PROBABILITIES": ml_model.output,
        }
        tf.compat.v1.saved_model.simple_save(sess, str(model_path),
                                             inputs, outputs)

def train_and_export_model(params):
    """The function gets the training data from the training folder and
    the eval folder.
    Your solution in the model.py file is trained with this training data.
    The evaluation in this method is not important since all data was already
    used to train.

    Parameters:
        params: Parameters for training and exporting the model
    """
    (train_data, train_labels) = data.create_data_with_labels("data/train/")
    (eval_data, eval_labels) = data.create_data_with_labels("data/eval/")

    train_data = np.append(train_data, eval_data, axis=0)
    train_labels = np.append(train_labels, eval_labels, axis=0)

    n_train_data, img_shape = train_data.shape[:1], train_data.shape[1:]
    input_layer = tf.keras.Input(shape=img_shape, name='input_image')

    ml_model = model.solution(input_layer)

    ml_model.fit(train_data, train_labels,
                 batch_size=model.get_batch_size(),
                 epochs=model.get_epochs())
    export_model(ml_model, export_dir=params.job_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        default='output',
        help='directory to store checkpoints'
    )

    args = parser.parse_args()
    tf.logging.set_verbosity('INFO')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__['INFO'] / 10)

    HPARAMS = hparam.HParams(**args.__dict__)
    train_and_export_model(HPARAMS)
