#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import gzip
import os

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels, DataSet
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

tf.logging.set_verbosity(tf.logging.INFO)

indir = '..\\..\\dataset\\gzip'

train_data_dir = os.path.join(indir, "emnist-letters-train-images-idx3-ubyte.gz")
train_labels_dir = os.path.join(indir, "emnist-letters-train-labels-idx1-ubyte.gz")
eval_data_dir = os.path.join(indir, "emnist-letters-test-images-idx3-ubyte.gz")
eval_labels_dir = os.path.join(indir, "emnist-letters-test-labels-idx1-ubyte.gz")


# https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/contrib/learn/python/learn/datasets/__init__.py

def read_data_sets(fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   ):
    if fake_data:
        def fake():
            return DataSet(
                [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    with gfile.Open(train_data_dir, 'rb') as f:
        train_images = extract_images(f)

    with gfile.Open(train_labels_dir, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    with gfile.Open(eval_data_dir, 'rb') as f:
        test_images = extract_images(f)

    with gfile.Open(eval_labels_dir, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                         .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)


# http://tensorlayercn.readthedocs.io/zh/latest/_modules/tensorlayer/files.html
# def load_mnist_images(filename):
#     filepath = os.path.join(indir, filename)
#     print(filepath)
#     # Read the inputs in Yann LeCun's binary format.
#     with gzip.open(filepath, 'rb') as f:
#         data = np.frombuffer(f.read(), np.uint8, offset=16)
#     # The inputs are vectors now, we reshape them to monochrome 2D images,
#     # following the shape convention: (examples, channels, rows, columns)
#     data = data.reshape((-1, 784))
#     # The inputs come as bytes, we convert them to float32 in range [0,1].
#     # (Actually to range [0, 255/256], for compatibility to the version
#     # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
#     return data / np.float32(256)
#
#
# def load_mnist_labels(filename):
#     filepath = os.path.join(indir, filename)
#     # Read the labels in Yann LeCun's binary format.
#     with gzip.open(filepath, 'rb') as f:
#         data = np.frombuffer(f.read(), np.uint8, offset=8)
#     # The labels are vectors of integers now, that's exactly what we want.
#     return data


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    tf.summary.image(
        name='input',
        tensor=input_layer,
        max_outputs=20,
        collections=None,
        family=None
    )

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=34,
        kernel_size=7,
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=7,
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 150]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=200,
        kernel_size=7,
        padding="same",
        activation=tf.nn.relu)



    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 128]
    # Output Tensor Shape: [batch_size, 7 * 7 * 128]
    pool2_flat = tf.reshape(conv3, [-1, 7 * 7 * 200])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 150]
    # Output Tensor Shape: [batch_size, 7350]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 27]
    logits = tf.layers.dense(inputs=dropout, units=27)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)

    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=27)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    eval_metric_ops = {"accuracy": accuracy}
    # tf.summary.scalar('accuracy', accuracy)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data

    # train_data = load_mnist_images("emnist-digits-train-images-idx3-ubyte.gz")
    # train_labels = np.asarray(load_mnist_labels("emnist-digits-train-labels-idx1-ubyte.gz"), dtype=np.int32)
    # eval_data = load_mnist_images("emnist-digits-test-images-idx3-ubyte.gz")
    # eval_labels = np.asarray(load_mnist_labels("emnist-digits-test-labels-idx1-ubyte.gz"), dtype=np.int32)

    #
    # with open(train_data_dir, 'rb') as f:
    #     train_data = np.asarray(extract_images(f), dtype=np.float32)
    # with open(train_labels_dir, 'rb') as f:
    #     train_labels = np.asarray(extract_labels(f), dtype=np.int32)
    #
    # with open(eval_data_dir, 'rb') as f:
    #     eval_data = np.asarray(extract_images(f), dtype=np.float32)
    # with open(eval_labels_dir, 'rb') as f:
    #     eval_labels = np.asarray(extract_labels(f), dtype=np.int32)

    # with open(train_data_dir, 'rb') as f:
    #     train_data = extract_images(f)
    # with open(train_labels_dir, 'rb') as f:
    #     train_labels = extract_labels(f, one_hot=False)
    #
    # with open(eval_data_dir, 'rb') as f:
    #     eval_data = extract_images(f)
    # with open(eval_labels_dir, 'rb') as f:
    #     eval_labels = extract_labels(f, one_hot=False)

    emnist = read_data_sets()
    train_data = emnist.train.images  # Returns np.array
    train_labels = np.asarray(emnist.train.labels, dtype=np.int32)
    eval_data = emnist.test.images  # Returns np.array
    eval_labels = np.asarray(emnist.test.labels, dtype=np.int32)

    # Create the Estimator
    emnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/Emnist_7conv-1-32-P-64-P-200-10_10k")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    emnist_classifier.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        # batch_size=128,
        num_epochs=1,
        shuffle=False)
    eval_results = emnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
