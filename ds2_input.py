from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from scipy.io import loadmat
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

DATA_PATH = '/home/jiho/612/deepsat/dataset'
DATA_FILE = 'sat-6-full.mat'

TOY_SIZE_TR = 200000
TOY_SIZE_TS = 50000
VALIDATION_SIZE = 50000 

# Constants describing the data
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

class DataSet(object):

  def __init__(self, images, labels, dtype=tf.float32):
    """Construct a DataSet.
    `dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32`
    to rescale into `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]

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

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
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

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the data in the .run() loop.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                      IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def read_data_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()

  filename = os.path.join(DATA_PATH, DATA_FILE)
  if not tf.gfile.Exists(filename):
    raise ValueError('Failed to find data file: ' + filename)

  # load .mat file using scipy.io.loadmat
  data = loadmat(filename);

  # just use one channel for simplicity
  tr_x = np.rollaxis(data['train_x'], 3)[VALIDATION_SIZE:TOY_SIZE_TR,...,1]
  tr_x = tr_x.reshape((-1,IMAGE_PIXELS))
  tr_y = np.rollaxis(data['train_y'], 1)[VALIDATION_SIZE:TOY_SIZE_TR]
  tr_y = np.transpose(np.nonzero(tr_y))[:,1]

  vl_x = np.rollaxis(data['train_x'], 3)[:VALIDATION_SIZE,...,1]
  vl_x = vl_x.reshape((-1,IMAGE_PIXELS))
  vl_y = np.rollaxis(data['train_y'], 1)[:VALIDATION_SIZE]
  vl_y = np.transpose(np.nonzero(vl_y))[:,1]

  ts_x = np.rollaxis(data['test_x'], 3)[:TOY_SIZE_TS,...,1]
  ts_x = ts_x.reshape((-1,IMAGE_PIXELS))
  ts_y = np.rollaxis(data['test_y'], 1)[:TOY_SIZE_TS]
  ts_y = np.transpose(np.nonzero(ts_y))[:,1]

  data_sets.train = DataSet(tr_x, tr_y, dtype=tf.float32)
  data_sets.validation = DataSet(vl_x, vl_y, dtype=tf.float32)
  data_sets.test = DataSet(ts_x, ts_y, dtype=tf.float32)

  return data_sets
