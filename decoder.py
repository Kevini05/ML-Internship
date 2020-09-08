# Import deep learning library
import keras
from keras.layers.core import Activation, Dense, Lambda
import keras.backend as K
import tensorflow as tf

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
# G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
#               [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
#               [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) # Matrice g�n�ratrice
G = np.array([ [1, 1, 1, 1, 0, 0, 0, 0],
               [1, 1, 0, 0, 1, 1, 0, 0],
               [1, 0, 1, 0, 1, 0, 1, 0],
               [1, 1, 1, 1, 1, 1, 1, 1]])
k = len(G)      #Nombre de bits � envoyer
N = len(G[1])   #codeword length

c = FEC_encoder(k,G)
print('size C: ',len(c), 'size Cn: ', len(c[0]))

c = np.array(c)
In = np.eye(2**k) # List of outputs of NN

print(c.shape)
########### Neural Network Generator ###################
optimizer = 'adam'
loss = 'categorical_crossentropy'                # or 'mse'
train_epsilon = 0.1


def BSC_channel(x, epsilon):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = K.random_uniform(K.shape(x),minval=0.0, maxval=1000.0)/(1000.0) < epsilon
  return tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),tf.cast(2,tf.float32)) # Signal transmis + Bruit

def return_output_shape(input_shape):
  print('*****************************************************************',input_shape)
  return input_shape

### Layers definitions
inputs = keras.Input(shape =(N,))

x = Dense(units=2*2**(k), input_dim=k, activation='relu')(inputs)

outputs = Dense(units=2**k, activation='softmax')(x)

### Model Build
model_decoder = keras.Model(inputs=inputs, outputs=outputs)

inputs_meta = keras.Input(shape =(N,))
noisy_bits = Lambda(BSC_channel, arguments={'epsilon':train_epsilon}, output_shape=return_output_shape)(inputs_meta)
decoded_bits = model_decoder(inputs=noisy_bits)

meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits)

### Model print summary
model_decoder.summary()
meta_model.summary()

### Compile our model
model_decoder.compile(loss=loss, optimizer=optimizer)

### Fit the model
print('---------------------------------------fit----------------------------------------')
history = model_decoder.fit(c, In, epochs=100000,verbose=0)

plt.semilogy(history.history['loss'])
plt.title('Loss function w.r.t. Epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'])
plt.grid()
plt.show()