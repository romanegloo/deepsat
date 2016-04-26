#!/usr/bin/python
# Filename: ds_input.py

from __future__ import absolute_import                                          
from __future__ import division                                                 
from __future__ import print_function   

import gzip                                                                     
import os
import re
import sys
import tarfile 

from scipy.io import loadmat
import numpy as np
import tensorflow as tf
                                                                                
FLAGS = tf.app.flags.FLAGS

IMAGE_SIZE = 28
DATA_PATH = '/home/jiho/612/deepsat/dataset'
DATA_FILE = 'sat-6-full.mat'

def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

# def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    # batch_size, shuffle):
  # """Construct a queued batch of images and labels.

  # Args:
    # image: 3-D Tensor of [height, width, 3] of type.float32.
    # label: 1-D Tensor of type.int32
    # min_queue_examples: int32, minimum number of samples to retain
      # in the queue that provides of batches of examples.
    # batch_size: Number of images per batch.
    # shuffle: boolean indicating whether to use a shuffling queue.

  # Returns:
    # images: Images. 4D tensor of [batch_size, height, width, 3] size.
    # labels: Labels. 1D tensor of [batch_size] size.
  # """
  # # Create a queue that shuffles the examples, and then
  # # read 'batch_size' images + labels from the example queue.
  # num_preprocess_threads = 16
  # if shuffle:
    # images, label_batch = tf.train.shuffle_batch(
        # [image, label],
        # batch_size=batch_size,
        # num_threads=num_preprocess_threads,
        # capacity=min_queue_examples + 3 * batch_size,
        # min_after_dequeue=min_queue_examples)
  # else:
    # images, label_batch = tf.train.batch(
        # [image, label],
        # batch_size=batch_size,
        # num_threads=num_preprocess_threads,
        # capacity=min_queue_examples + 3 * batch_size)

  # # Display the training images in the visualizer.
  # tf.image_summary('images', images)

  # return images, tf.reshape(label_batch, [batch_size])

def load_images(eval_data):
  """Construct input for evaluation using the Rader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, w, h, 3] size
    labels: Labels. 1D tensor of [batch_size] size
  """ 
  filename = os.path.join(DATA_PATH, 'sat-6-full.mat')
  if not tf.gfile.Exists(filename):
    raise ValueError('Failed to find data file: ' + filename)

  # load .mat file using scipy.io.loadmat
  data = loadmat(filename);

  # Image pre-processing
  if not eval_data:
      x_ = data['train_x']
      y_ = data['train_y']
  else:
      x_ = data['test_x']
      y_ = data['test_y']

  # reshape labels - ex. [0, 0, 1, 0, 0, 0] to 2, where the index of nonzero 
  # element represents the label
  y_lbl = y_.transpose().nonzero()[1]
  
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Read examples in and Image processing for evaluation.
  reshaped_image = tf.cast(np.rollaxis(x_, 3)[:,:,:,:3], tf.float32)
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  #min_fraction_of_examples_in_queue = 0.4
  #min_queue_examples = int(num_examples_per_epoch *
  #                         min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  #return _generate_image_and_label_batch(float_image, read_input.label,
  #                                       min_queue_examples, batch_size,
  #                                       shuffle=False)
  # and ignore NIR values
  return float_image, y_lbl

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      #assert images.shape[3] == 1
      #images = images.reshape(images.shape[0],
      #                        images.shape[1] * images.shape[2])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()

  filename = os.path.join(DATA_PATH, DATA_FILE)
  if not tf.gfile.Exists(filename):
    raise ValueError('Failed to find data file: ' + filename)

  # load .mat file using scipy.io.loadmat
  data = loadmat(filename);

  TOY_SIZE_TR = 30000
  TOY_SIZE_TS = 10000
  VALIDATION_SIZE = 5000

  # reshape labels - ex. [0, 0, 1, 0, 0, 0] to 2, where the index of nonzero 

  tr_x = np.rollaxis(data['train_x'], 3)[VALIDATION_SIZE:TOY_SIZE_TR,:,:,:3]
  tr_y = np.rollaxis(data['train_y'], 1)[VALIDATION_SIZE:TOY_SIZE_TR]
  tr_y = np.transpose(np.nonzero(tr_y))[:,1]

  vl_x = np.rollaxis(data['train_x'], 3)[:VALIDATION_SIZE,:,:,:3]
  vl_y = np.rollaxis(data['train_y'], 1)[:VALIDATION_SIZE]
  vl_y = np.transpose(np.nonzero(vl_y))[:,1]

  ts_x = np.rollaxis(data['test_x'], 3)[:TOY_SIZE_TS,:,:,:3]
  ts_y = np.rollaxis(data['test_y'], 1)[:TOY_SIZE_TS]
  ts_y = np.transpose(np.nonzero(ts_y))[:,1]

  data_sets.train = DataSet(tr_x, tr_y, dtype=tf.float32)
  data_sets.validation = DataSet(vl_x, vl_y, dtype=tf.float32)
  data_sets.test = DataSet(ts_x, ts_y, dtype=tf.float32)

  return data_sets
