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

# def BAC_channel(x, epsilon0):
#   """ Entrée : Symboles à envoyer
#       Sortie : Symboles reçus, bruités """
#   # epsilon1 = K.random_uniform(shape=(2,1),minval=0.5, maxval=0.5)
#   epsilon1 = np.random.uniform(low=0.0, high=1.0, size=(2,1))
#
#   e = [int(x*4) for x in epsilon1]
#   interval = np.eye(4)[e]
#   print(epsilon1,interval)
#   interval = tf.cast(interval,tf.float32)
#
#
#   print('******************************************************************************')
#   print('input',x)
#
#   y = tf.cast(2,tf.float32)
#   n0 = K.random_uniform(K.shape(x),minval=0.0, maxval=1.0) < epsilon0
#   n1 = K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1
#   n = tf.cast(n0,tf.float32)*tf.math.floormod(x+1,y) + tf.cast(n1,tf.float32)*tf.math.floormod(x,y)
#
#   print('bruit',n)
#   X = tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),y)
#   print('signal+bruit',X,'intervale', interval)
#   X1 = tf.concat([X,interval],1) # Signal transmis + Bruit
#   print('concat', X1)
#   return X1
#
# e0 = float(sys.argv[1])
# e1 = float(sys.argv[2])
# ## Debugginf for noise BSC layer
# with tf.compat.v1.Session() as sess:
#   x = tf.constant([[0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.],[1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.]])
#   y_test = BAC_channel(x, e0)
#   aux = y_test.eval()
#   print(aux)
#   # print(tf.rank(aux))


# # load weights into new model
# model_decoder = keras.models.load_model("model/model_decoder_BAC_rep-1000_epsilon-0.07_layerSize_4_epoch-100_k_4_N-8.h5")
# print("Loaded model from disk")
#
# import time
#
# for i in range(10):
#   start = time.time()
#   X = np.reshape([0,0,0,0,0,0,0,0,0,0,0,1],[1,12])
#   print(np.argmax(model_decoder.predict(X)))
#   end = time.time()
#   print('Prediction time', end - start)

# # load weights into new model
# model_encoder = keras.models.load_model("autoencoder/model_encoder_BSC_rep-1000_epsilon-0.07_layerSize_4_epoch-100_k_4_N-8.h5")
# model_decoder = keras.models.load_model("autoencoder/model_decoder_BSC_rep-1000_epsilon-0.07_layerSize_4_epoch-100_k_4_N-8.h5")
# print("Loaded model from disk")
#
# codebook = []
# k = 4
# N = 8
# one_hot = np.eye(2**k)
# for X in one_hot:
#   X = np.reshape(X,[1,2**k])
#   print('X',X)
#
#   c = [round(x) for x in model_encoder.predict(X)[0]]
#   codebook.append(c)
#   print('c',c)
#   c = np.reshape(c,[1,N])
#   U_t = model_decoder.predict(c)
#   print(np.argmax(U_t))
#
# aux = []
# for code in codebook:
#   if code in aux:
#     # print('****repeated code******')
#     a=1
#   else:
#     aux.append(code)
# print('+++++++++++++++++++Repeated Codes = ',len(codebook)-len(aux))

# load weights into new model
model_encoder = keras.models.load_model("autoencoder/model_encoder_BSC_rep-100_epsilon-0.07_layerSize_5_epoch-10_k_4_N-8.h5")
print("Loaded model from disk")

codebook = []
k = 4
N = 8
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