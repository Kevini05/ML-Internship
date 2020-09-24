# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
import keras.backend as K
import tensorflow as tf

import BACchannel as bac
import polar_codes_generator as polar
import  ber_bler_calculator as test
import utils

import numpy as np
import matplotlib.pyplot as plt
import time

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
  n = K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon
  return tf.math.floormod(tf.add(x, tf.cast(n, tf.float32)), tf.cast(2, tf.float32))  # Signal transmis + Bruit

def BAC_channel(x, epsilon0):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles bruités + intervale crossover probability"""
  epsilon1 = np.random.uniform(low=0.0, high=1.0, size=(32, 1))
  interval = np.eye(4)[[int(x * 4) for x in epsilon1]]
  interval = tf.cast(interval, tf.float32)

  y = tf.cast(2,tf.float32)
  n0 = K.random_uniform(K.shape(x),minval=0.0, maxval=1.0) < epsilon0
  n1 = K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1
  n = tf.cast(n0,tf.float32)*tf.math.floormod(x+1,y) + tf.cast(n1,tf.float32)*tf.math.floormod(x,y)

  X = tf.math.floormod(tf.add(x,tf.cast(n,tf.float32)),y) # Signal transmis + Bruit
  return tf.concat([X,interval],1) # Signal transmis + Bruit + Intervale

def return_output_shape(input_shape):
  print('*****************************************************************', input_shape)
  return input_shape

def gradient_stopper(x):
  return tf.stop_gradient(tf.math.round(x)-x)+x

###### \Python3\python.exe autoencoder.py BSC 8 4 10
#Parameters
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
epoch = int(sys.argv[4])

rep = 200
train_epsilon = 0.07
S = 5

In = np.eye(2**k) # List of outputs of NN
In = np.tile(In,(rep,1))
# print(In)
G, infoBits = polar.polar_generator_matrix(N,k)
c = FEC_encoder(k,G)
# print('size C: ',len(c), 'size Cn: ', len(c[0]))
c = np.tile(np.array(c),(rep,1))
# print(np.array(c))
# print('size C: ',len(c), 'size Cn: ', len(c[0]))

########### Neural Network Generator ###################
optimizer = 'adam'
loss = 'categorical_crossentropy'                # or 'mse'

### Encoder Layers definitions
def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape = 2**k)
  # x = Dense(units=S*2**(k), activation='relu')(inputs_encoder)
  x = Dense(units=S*2**(k), activation='relu')(inputs_encoder)
  outputs_encoder = Dense(units=N, activation='sigmoid')(x)
  ### Model Build
  model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder)
  return model_enc

###Meta model encoder
# inputs_meta = keras.Input(shape = 2**k)
# encoded_bits = model_encoder(inputs=inputs_meta)
# rounded_bits = Lambda(gradient_stopper)(encoded_bits)
# meta_model = keras.Model(inputs=inputs_meta, outputs=rounded_bits)

### Decoder Layers definitions
def decoder_generator(N,k,channel):
  print(k,type(k))
  if channel == 'BSC':
    inputs_decoder = keras.Input(shape=N)
  elif channel == 'BAC':
    inputs_decoder = keras.Input(shape = N+4)
  x = Dense(units=S*2**(k), activation='relu')(inputs_decoder)
  outputs_decoder = Dense(units=2 ** k, activation='softmax')(x)
  ### Model Build
  model_dec = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)
  return  model_dec

### Meta model Layers definitions
def meta_model_generator(k,channel,model_enc,model_dec):
  inputs_meta = keras.Input(shape = 2**k)
  encoded_bits = model_enc(inputs=inputs_meta)
  if channel == 'BSC':
    noisy_bits = Lambda(BSC_channel, arguments={'epsilon':train_epsilon}, output_shape=return_output_shape)(encoded_bits)
  elif channel == 'BAC':
    noisy_bits = Lambda(BAC_channel, arguments={'epsilon0':train_epsilon})(encoded_bits)
  decoded_bits = model_dec(inputs=noisy_bits)
  meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits)
  return meta_model

model_encoder = encoder_generator(N,k)
model_decoder = decoder_generator(N,k,channel)
meta_model = meta_model_generator(k,channel,model_encoder,model_decoder)

### Model print summary
# model_encoder.summary()
# model_decoder.summary()
# meta_model.summary()

### Compile our models
model_encoder.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy()])
model_decoder.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy()])
meta_model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy()])

### Fit the model
history = meta_model.fit(In, In, epochs=epoch,verbose=1, shuffle=True, batch_size=32, validation_data=(In,In))
# history = model_encoder.fit(In, c, epochs=epoch,verbose=1)
print("The model is ready to be used...")

### save Model
model_decoder.save(f"./autoencoder/model_decoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
model_encoder.save(f"./autoencoder/model_encoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
# model_decoder.save(f"./autoencoder/model_decoder.h5")
# model_encoder.save(f"./autoencoder/model_encoder.h5")

# Summarize history for loss
plt.semilogy(history.history['loss'])
plt.title('Loss function w.r.t. Epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'])
plt.grid()

#####################################################
## TEST


def NN_encoder(k,N):
  codebook = []
  one_hot = np.eye(2 ** k)
  for X in one_hot:
    X = np.reshape(X, [1, 2**k])
    c = [int(round(x)) for x in model_encoder.predict(X)[0]]
    codebook.append(c)
  return codebook

def bit_error_rate_NN(N, k, C, B, e0, e1, channel = 'decoder' ):
  U_k = bac.symbols_generator(k)  # all possible messages
  ber = {}
  count = 0
  for ep0 in e0:
    ber_ep0 = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        ber_ep1 = 0  # for bit error rate
        interval = np.zeros(4)
        interval[int(ep1*4)] = 1.0
        for t in range(B):
          idx = np.random.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits à envoyer
          x = C[idx]  # bits encodés

          y_bac = bac.BAC_channel(x, ep0, ep1)  # symboles reçus

          start = time.time()
          yh = np.reshape(np.concatenate((y_bac,interval),axis=0), [1, N+4]) if channel == 'autoencoder'  else np.reshape(y_bac, [1, N])
          u_nn = U_k[np.argmax(model_decoder.predict(yh))]  # Detecteur NN
          end = time.time()
          # print('NN', end - start)

          ber_ep1 += bac.NbOfErrors(u, u_nn)  # Calcul de bit error rate avec NN
        ber_ep1 = ber_ep1 / (k * 1.0 * B)  # Calcul de bit error rate avec NN
        ber_ep0.append(ber_ep1)

    ber[ep0] = ber_ep0
    count+= 1
    print("{:.3f}".format(count/len(e0)*100),'% completed ')
  return ber

def BER_NN():
  e0 = np.logspace(-3, 0, 11)
  e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
  e1 = [t for t in e0 if t <= 0.5]
  nb_pkts = 1000
  BER = test.saved_results(N, k)
  C = NN_encoder(k, N)

  t = time.time()
  BER['auto'] = bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,'autoencoder')
  t = time.time()-t
  print(f"NN time = {t}s ========================")

  t = time.time()
  BER['MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1)
  t = time.time()-t
  print(f"MAP time = {t}s =======================")

  test.plot_ber(BER, N,k,e0)


BER_NN()
plt.show()
