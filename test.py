import utils
import BACchannel as bac
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from keras.models import Sequential, model_from_json
import keras.backend as K
import tensorflow as tf
import keras

def BAC_channel(x, epsilon0, epsilon1):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  y = tf.cast(2,tf.float32)
  n0 = K.random_uniform(K.shape(x),minval=0.0, maxval=1.0) < epsilon0
  n1 = K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1
  n = tf.cast(n0,tf.float32)*tf.math.floormod(x+1,y) + tf.cast(n1,tf.float32)*tf.math.floormod(x,y)
  return tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),y) # Signal transmis + Bruit

## Debugginf for noise BSC layer
with tf.compat.v1.Session() as sess:
  x = tf.constant([0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.])
  y_test = BAC_channel(x, 0.5, 0.5)
  print(y_test.eval())


# # load weights into new model
# model_decoder = keras.models.load_model("model/model_decoder_bsc_rep-1000_epsilon-0.07_layerSize_2_epoch-100_k_4_N-8.h5")
# print("Loaded model from disk")
#
# import time
#
# for i in range(10):
#   start = time.time()
#   X = np.reshape([0,0,0,0,0,0,0,0],[1,8])
#   print(np.argmax(model_decoder.predict(X)))
#   end = time.time()
#   print('Prediction time', end - start)