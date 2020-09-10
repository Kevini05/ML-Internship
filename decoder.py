# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Activation, Dense, Lambda
import keras.backend as K
import tensorflow as tf

import polar_codes_generator as polar
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def sequence_generator(k):
  """
  Entr�e : Nombre de bits � envoyer
  Sortie : all possible codewords
  """
  codewords = np.ones((2**k,k))
  for i in range(2**k):
     nb = bin(i)[2:].zfill(k)
     for j in range(k):
        codewords[i][j] = int(nb[j])
  return codewords

def FEC_encoder(k,G):
  """
  Entr�e : sequence de bits
  Sortie : sequence g�n�r�e gr�ce � la matrice g�n�ratrice
  """
  c = []
  messages = sequence_generator(k)
  for b in messages:
    c.append(np.dot(np.transpose(G),np.transpose(b))%2)
  return c

#Parameters

###### \Python3\python.exe BAC decoder.py 8 4 10

N = int(sys.argv[2])
k = int(sys.argv[3])
channel = sys.argv[1]
G, infoBits = polar.polar_generator_matrix(N,k)

k = len(G)      #Nombre de bits � envoyer
N = len(G[1])   #codeword length
rep = 1000
train_epsilon = 0.07
epoch = int(sys.argv[4])
S = 2
c = FEC_encoder(k,G)
print('size C: ',len(c), 'size Cn: ', len(c[0]))
c = np.array(c)

print(type(c[0]))
In = np.eye(2**k) # List of outputs of NN
c = np.tile(c,(rep,1))
In = np.tile(In,(rep,1))
print('size C: ',len(c), 'size Cn: ', len(c[0]))

########### Neural Network Generator ###################
optimizer = 'adam'
loss = 'categorical_crossentropy'                # or 'mse'



def BSC_channel(x, epsilon):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = K.random_uniform(K.shape(x),minval=0.0, maxval=1.0) < epsilon
  return tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),tf.cast(2,tf.float32)) # Signal transmis + Bruit

def BAC_channel(x, epsilon0, epsilon1):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  y = tf.cast(2,tf.float32)
  n0 = K.random_uniform(K.shape(x),minval=0.0, maxval=1.0) < epsilon0
  n1 = K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1
  n = tf.cast(n0,tf.float32)*tf.math.floormod(x+1,y) + tf.cast(n1,tf.float32)*tf.math.floormod(x,y)
  return tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),y) # Signal transmis + Bruit

def return_output_shape(input_shape):
  print('*****************************************************************',input_shape)
  return input_shape

### Layers definitions
inputs = keras.Input(shape =(N,))

x = Dense(units=S*2**(k), input_dim=k, activation='relu')(inputs)

outputs = Dense(units=2**k, activation='softmax')(x)

### Model Build
model_decoder = keras.Model(inputs=inputs, outputs=outputs)

inputs_meta = keras.Input(shape =(N,))
if channel == 'BSC':
  noisy_bits = Lambda(BSC_channel, arguments={'epsilon':train_epsilon}, output_shape=return_output_shape)(inputs_meta)
elif channel == 'BAC':
  noisy_bits = Lambda(BAC_channel, arguments={'epsilon0':train_epsilon,'epsilon1':train_epsilon+0.07}, output_shape=return_output_shape)(inputs_meta)
decoded_bits = model_decoder(inputs=noisy_bits)

meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits)

### Model print summary
# model_decoder.summary()
meta_model.summary()

### Compile our model
meta_model.compile(loss=loss, optimizer=optimizer)

### Fit the model
history = meta_model.fit(c, In, epochs=epoch,verbose=1)
print("The model is ready to be used...")

### serialize model to JSON
model_decoder.save(f"./model/model_decoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")


# Summarize history for loss
plt.semilogy(history.history['loss'])
plt.title('Loss function w.r.t. Epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'])
plt.grid()
plt.show()

