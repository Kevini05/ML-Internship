import utils
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from keras.models import Sequential, model_from_json
import keras.backend as K
import tensorflow as tf

def BSC_channel(x, epsilon):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = K.random_uniform(K.shape(x),minval=0.0, maxval=1000.0)/(1000.0) < epsilon
  return tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),tf.cast(2,tf.float32)) # Signal transmis + Bruit

def return_output_shape(input_shape):
  print('*****************************************************************',input_shape)
  return input_shape


with tf.compat.v1.Session() as sess:
  x = tf.constant([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
  y_test = BSC_channel(x, 0.5)
  print(y_test.eval())
