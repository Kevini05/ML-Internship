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

# # ==============================================================================
# shape = 1.5
# scale = 0.1
# # epsilon = np.random.gamma(1.5, 0.1, 100000)
# # epsilon = np.random.chisquare(4, 100000) * 0.01
# # epsilon = np.random.exponential(0.05, 100000)
# epsilon = np.random.lognormal(mean=-2.5,sigma=0.4, size=100000)
# # epsilon = np.random.uniform(low=0.0, high=0.07, size=100000)
#
# s = [(int(3*np.log10(s)+4) if s >=0.1 and s<1.0 else 0) for s in epsilon]
# plt.hist(s, 100)
# plt.hist(epsilon, 100)
# plt.show()
# # ==============================================================================
k = 4
rep = 10
In = np.eye(2**k) # List of outputs of NN
In = np.tile(In,(rep,1))
batch_size = int(len(In)/1)

Interval = []

idx =[0.5,0.3,0.15,0.05]
for i in range(4):
  for j in range(int(len(In)*idx[i])):
    Interval.append(np.eye(4)[i].tolist())


# Interval =  np.reshape(Interval, (len(In), 4))
print('Interval \n',Interval, len(Interval))


# def BSC_noise(inputs, epsilon_max,batch_size,k,N):
#   """ parameter : Symboles à envoyer
#       return : Symboles reçus, bruités """
#   x = inputs[0]
#
#   minibatch = 2**k
#   noise_len = int(batch_size/minibatch)
#
#   epsilon = np.random.uniform(low=0.0, high=epsilon_max, size=(noise_len, 1))
#   # epsilon = np.random.lognormal(mean=-2.5, sigma=0.4, size=(batch_size, 1))
#   epsilon = np.reshape(np.repeat(epsilon, minibatch), (batch_size, 1))
#
#   two = tf.cast(2, tf.float32)
#   n = tf.cast( K.random_uniform(shape=K.shape(x), minval=0.0, maxval=1.) < epsilon,tf.float32)
#   # n = K.reshape(K.repeat(n, n=minibatch), shape=(batch_size,N))
#
#   interval = np.eye(4)[[int(s*4/epsilon_max)for s in epsilon]]
#   interval = tf.cast(interval, tf.float32)
#
#   y = tf.math.floormod(x+n,two)
#   # K.print_tensor(epsilon, 'epsilon \n')
#   # K.print_tensor(x, 'x \n')
#   # K.print_tensor(n,'n \n')
#   # K.print_tensor(y, 'y \n')
#   # return tf.concat([y,interval],1)
#   # return tf.concat([x,interval],1)
#   return y # Signal transmis + Bruit
