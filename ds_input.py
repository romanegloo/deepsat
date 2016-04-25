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

DATA_PATH = '/home/jiho/612/deepsat/dataset/'

def load_images(eval_data=False):
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
  
  # Subtract off the mean and divide by the variance of the pixels

  # and ignore NIR values
  return np.rollaxis(x_,3)[:,:,:,:3], y_lbl
