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

# def BAC_channel(x, epsilon0, epsilon1):
#   """ input : Symbols to be sent
#       :return: Symboles reçus, bruités """
#   # print('e0 ',epsilon0)
#   # print('e1 ',epsilon1)
#   x = np.array(x)
#   n0 = np.array([int(b0<epsilon0) for b0 in np.random.uniform(0.0, 1.0, len(x))])
#   n1 = np.array([int(b1<epsilon1) for b1 in np.random.uniform(0.0, 1.0, len(x))])
#   n = n0*(x+1)+n1*x
#   return np.mod(n+x,2) # Signal transmis + Bruit
#
#
# x = [0,1]*5
# # x = [0]*10
# # x = [1]*10
# y = BAC_channel(x,1,1)
# print(y)

shape = 1.5
scale = 0.1
# epsilon = np.random.gamma(1.5, 0.1, 100000)
# epsilon = np.random.chisquare(4, 100000) * 0.01
# epsilon = np.random.exponential(0.05, 100000)
epsilon = np.random.lognormal(mean=-2.5,sigma=0.4, size=100000)
# epsilon = np.random.uniform(low=0.0, high=0.07, size=100000)

s = [(int(3*np.log10(s)+4) if s >=0.1 and s<1.0 else 0) for s in epsilon]
plt.hist(s, 100)
plt.hist(epsilon, 100)
plt.show()