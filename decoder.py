# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
import keras.backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import  ber_bler_calculator as test
import utils

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

def BSC_noise(x, epsilon):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = K.random_uniform(K.shape(x),minval=0.0, maxval=1.0) < epsilon
  return tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),tf.cast(2,tf.float32)) # Signal transmis + Bruit

def BAC_noise(x, epsilon0,batch_size):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles bruités + intervale crossover probability"""

  epsilon1_train_max = 0.25

  mini_batch = 1
  noise_size = int(batch_size / mini_batch)
  epsilon1 = np.random.uniform(low=0.0, high=epsilon1_train_max, size=(noise_size, 1))
  epsilon1 = np.reshape(np.repeat(epsilon1, mini_batch), (batch_size, 1))

  interval = np.eye(4)[[int(s * 4/epsilon1_train_max) for s in epsilon1]]
  interval = tf.cast(interval, tf.float32)

  y = tf.cast(2,tf.float32)
  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon0, tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1, tf.float32)
  n = tf.math.floormod(n0*(x+1) + n1*x, y)

  X = tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),y) # Signal transmis + Bruit
  return tf.concat([X,interval],1) # Signal transmis + Bruit + Intervale


def return_output_shape(input_shape):
  print('*****************************************************************',input_shape)
  return input_shape

def bler_metric(u_true,u_predict):
  return K.mean(K.not_equal(K.argmax(u_true, 1),K.argmax(u_predict, 1)))

#Parameters

###### \Python3\python.exe decoder.py BAC 8 4 10

channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
epoch = int(sys.argv[4])
if k == 4:
  if N == 8:
    G = [ [1, 0, 0, 1, 0, 1, 1, 0],
          [0, 1, 0, 1, 0, 1, 0, 1],
          [0, 0, 1, 1, 0, 0, 1, 1],
          [0, 0, 0, 0, 1, 1, 1, 1]]
  if N == 16:
    G = [[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
         [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]

if k == 8 and N == 16:
  G = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
       [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1]]

k = len(G)      #Nombre de bits � envoyer
N = len(G[1])   #codeword length

rep = 500
train_epsilon = 0.07
S = 3

################### Coding
U_k = utils.symbols_generator(k)  # all possible messages
cn = utils.matrix_codes(U_k, k, G, N)
# print('codebook',np.array(cn))
print('size C: ',len(cn), 'size Cn: ', len(cn[0]))
c = np.array(cn)
print(type(c[0]))

In = np.eye(2**k) # List of outputs of NN
c = np.tile(c,(rep,1))
In = np.tile(In,(rep,1))
print('size C: ',len(c), 'size Cn: ', len(c[0]))
batch_size = len(In)

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
  noisy_bits = Lambda(BSC_noise, arguments={'epsilon':train_epsilon}, output_shape=return_output_shape)(inputs_meta)
elif channel == 'BAC':
  noisy_bits = Lambda(BAC_noise, arguments={'epsilon0':train_epsilon,'batch_size':batch_size})(inputs_meta)
decoded_bits = model_decoder(inputs=noisy_bits)
meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits)

### Model print summary
# model_decoder.summary()
meta_model.summary()

### Compile our models
model_decoder.compile(loss=loss, optimizer=optimizer)
meta_model.compile(loss=loss, optimizer=optimizer, metrics=[bler_metric])

### Fit the model
history = meta_model.fit(c, In, epochs=epoch,verbose=2, batch_size=batch_size)
print("The model is ready to be used...")

### serialize model to JSON
# model_decoder.save(f"./model/model_decoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
model_decoder.save(f"./autoencoder/model_decoder.h5")

# Summarize history for loss
plt.semilogy(history.history['loss'],label='Loss (training data)')
plt.title('Loss function w.r.t. Epoch')
plt.semilogy(history.history['bler_metric'],label='bler_metric (training data)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.grid()

MAP_test = False
def BER_NN(codebook,nb_pkts=100):
  e0 = np.logspace(-3, 0, 15)
  # e0 = np.linspace(0.001, 1, 11)
  e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
  e1 = [t for t in e0 if t <= 0.5]
  metric = test.read_ber_file(N, k)
  BER = test.saved_results(metric,N, k)

  BER['dec'] = utils.bit_error_rate_NN(N, k, codebook, nb_pkts, e0, e1,channel)
  print("metric['BKLC-NN'] = ",BER['dec'])
  if MAP_test:
    BER['MAP'] = utils.bit_error_rate(k, codebook, nb_pkts, e0, e1)

  test.plot_ber(BER, N,k,e0)

if len(sys.argv) > 5:
  if sys.argv[5] == 'BER':
    nb_pkts = int(sys.argv[6])
    BER_NN(cn,nb_pkts)
plt.show()



