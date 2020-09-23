import sys
import utils
import BACchannel as bac
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from keras.models import Sequential, model_from_json
import keras.backend as K
import tensorflow as tf
import keras
import time

# def gradient_stopper(x):
#   return tf.stop_gradient(tf.math.round(x)+x)-x
#
# ## Debugginf for noise BSC layer
# with tf.compat.v1.Session() as sess:
#   x = tf.constant([[0.6,0.3,0.51,0.49,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.],[1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.]])
#   y_test = gradient_stopper(x)
#   aux = y_test.eval()
#   print(aux)
#   # print(tf.rank(aux))

# # load weights into new model
# model_encoder = keras.models.load_model("autoencoder/model_encoder.h5")
# model_decoder = keras.models.load_model("autoencoder/model_decoder.h5")
# print("Loaded model from disk")
#
# codebook = []
# k = 4
# N = 8
# one_hot = np.eye(2**k)
# for X in one_hot:
#   X = np.reshape(X,[1,2**k])
#
#   c = [np.round(x) for x in model_encoder.predict(X)[0]]
#   codebook.append(c)
#
#   c = np.reshape(c,[1,N])
#   U_t = model_decoder.predict(c)
#   print('Coded',np.argmax(X),'c', c, 'Decoded',np.argmax(U_t))
#   print()
# aux = []
# for code in codebook:
#   if code in aux:
#     # print('****repeated code******')
#     a=1
#   else:
#     aux.append(code)
# print('+++++++++++++++++++Repeated Codes = ',len(codebook)-len(aux))

# load weights into new model
model_encoder = keras.models.load_model("autoencoder/model_encoder.h5")
print("Loaded model from disk")

codebook = []
k = 4
N = 16
one_hot = np.eye(2**k)
for X in one_hot:
  X = np.reshape(X,[1,2**k])
  # print('X',X)

  c = [round(x) for x in model_encoder.predict(X)[0]]
  codebook.append(c)
  # print('c',c)
  c = np.reshape(c,[1,N])
print(np.array(codebook))

aux = []
for code in codebook:
  if code in aux:
    # print('****repeated code******')
    a=1
  else:
    aux.append(code)
print('+++++++++++++++++++Repeated Codes = ',len(codebook)-len(aux))