# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
import keras.backend as K
import tensorflow as tf

import polar_codes_generator as polar
import numpy as np
import matplotlib.pyplot as plt

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

def BSC_channel(x, epsilon):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = K.random_uniform(K.shape(x),minval=0.0, maxval=1.0) < epsilon
  return tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),tf.cast(2,tf.float32)) # Signal transmis + Bruit

def BAC_channel(x, epsilon0):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles bruités + intervale crossover probability"""
  epsilon1 = np.random.uniform(low=0.0, high=0.25, size=(32, 1))
  interval = np.eye(4)[[int(x * 16) for x in epsilon1]]
  interval = tf.cast(interval, tf.float32)

  y = tf.cast(2,tf.float32)
  n0 = K.random_uniform(K.shape(x),minval=0.0, maxval=1.0) < epsilon0
  n1 = K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1
  n = tf.cast(n0,tf.float32)*tf.math.floormod(x+1,y) + tf.cast(n1,tf.float32)*tf.math.floormod(x,y)

  X = tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),y) # Signal transmis + Bruit
  return tf.concat([X,interval],1) # Signal transmis + Bruit + Intervale

def return_output_shape(input_shape):
  print('*****************************************************************',input_shape)
  return input_shape

#Parameters

###### \Python3\python.exe decoder.py BAC 8 4 10

channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
G, infoBits = polar.polar_generator_matrix(N,k)
G = [ [1, 0, 0, 1, 0, 1, 1, 0],
      [0, 1, 0, 1, 0, 1, 0, 1],
      [0, 0, 1, 1, 0, 0, 1, 1],
      [0, 0, 0, 0, 1, 1, 1, 1]]

k = len(G)      #Nombre de bits � envoyer
N = len(G[1])   #codeword length
rep = 1000
train_epsilon = 0.07
epoch = int(sys.argv[4])
S = 5
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

### Decoder Layers definitions
if channel == 'BSC':
  inputs_decoder = keras.Input(shape=N)
elif channel == 'BAC':
  inputs_decoder = keras.Input(shape = N+4)
x = Dense(units=S*2**(k), activation='relu')(inputs_decoder)
outputs_decoder = Dense(units=2**k, activation='softmax')(x)
### Model Build
model_decoder = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)

### Meta model Layers definitions
inputs_meta = keras.Input(shape = N)
if channel == 'BSC':
  noisy_bits = Lambda(BSC_channel, arguments={'epsilon':train_epsilon}, output_shape=return_output_shape)(inputs_meta)
elif channel == 'BAC':
  noisy_bits = Lambda(BAC_channel, arguments={'epsilon0':train_epsilon})(inputs_meta)
decoded_bits = model_decoder(inputs=noisy_bits)
meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits)

### Model print summary
# model_decoder.summary()
meta_model.summary()

### Compile our models
model_decoder.compile(loss=loss, optimizer=optimizer)
meta_model.compile(loss=loss, optimizer=optimizer)

### Fit the model
history = meta_model.fit(c, In, epochs=epoch,verbose=1, batch_size=32)
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


