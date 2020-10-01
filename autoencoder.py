# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf

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

def BSC_noise(x, epsilon):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon
  # K.print_tensor(x, 'x')
  # K.print_tensor(n,'n')
  aux = tf.math.floormod(tf.add(x, tf.cast(n, tf.float32)), tf.cast(2, tf.float32))
  # K.print_tensor(aux, 'noisy')
  return aux # Signal transmis + Bruit

def BAC_noise(x, epsilon0, batch_size):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles bruités + intervale crossover probability"""
  epsilon1_train_max = 0.25

  mini_batch = 1
  noise_size = int(batch_size/mini_batch)
  epsilon1 = np.random.uniform(low=0.0, high=0.1, size=(noise_size, 1))
  epsilon1 = np.reshape(np.repeat(epsilon1, mini_batch), (batch_size, 1))

  interval = np.eye(4)[[int(s * 3.99999/epsilon1_train_max) for s in epsilon1]]
  interval = tf.cast(interval, tf.float32)

  y = tf.cast(2, tf.float32)

  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon0,tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1,tf.float32)
  n = tf.math.floormod(n0*(tf.math.round(x)+1) + n1*tf.math.round(x), y)

  # K.print_tensor(x, 'x')
  # K.print_tensor(tf.math.round(x), 'x-rounded')
  # K.print_tensor(n0, 'n0')
  # K.print_tensor(n1, 'n1')
  # K.print_tensor(n,'n')

  X = tf.math.floormod(x+n,y) # Signal transmis + Bruit
  # K.print_tensor(X, 'X')
  return tf.concat([X,interval],1) # Signal transmis + Bruit + Intervale

def return_output_shape(input_shape):
  print('*****************************************************************', input_shape)
  return input_shape

def gradient_stopper(x):
  output = tf.stop_gradient(tf.math.round(x)-x)+x
  # K.print_tensor(x, 'x')
  # K.print_tensor(output, 'rounded')
  return output

def bler_metric(u_true,u_predict):
  # K.print_tensor(u_true,u_predict)
  # K.print_tensor(K.argmax(u_true, 1), K.argmax(u_predict, 1))
  # bler = K.mean(K.not_equal(K.argmax(u_true, 1),K.argmax(u_predict, 1)))
  # K.print_tensor(bler, 'BLER')
  return K.mean(K.not_equal(K.argmax(u_true, 1),K.argmax(u_predict, 1)))

###### \Python3\python.exe autoencoder.py BAC 16 4 300
###### \Python3\python.exe autoencoder.py BAC 16 4 300 BER 100
#Parameters
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
epoch = int(sys.argv[4])

rep = 500
train_epsilon = 0.07
S = 3

MAP_test = True

In = np.eye(2**k) # List of outputs of NN
In = np.tile(In,(rep,1))
# print(In)
batch_size = len(In)#int(len(In)/2)

########### Neural Network Generator ###################
optimizer = 'adam'
optimizer_enc = keras.optimizers.Adam(lr=0.001)
optimizer_dec = keras.optimizers.Adam(lr=0.001)
loss = 'categorical_crossentropy'                # or 'mse'

### Encoder Layers definitions
def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape = 2**k)
  x = Dense(units=S*2**(k), activation='relu')(inputs_encoder)
  # x = Dense(units=S * 2 ** (k), activation='relu')(x)
  # x = BatchNormalization()(x)
  outputs_encoder = Dense(units=N, activation='sigmoid')(x)
  ### Model Build
  model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder_model')
  return model_enc

###Meta model encoder
# inputs_meta = keras.Input(shape = 2**k)
# encoded_bits = model_encoder(inputs=inputs_meta)
# rounded_bits = Lambda(gradient_stopper)(encoded_bits)
# meta_model = keras.Model(inputs=inputs_meta, outputs=rounded_bits)

### Decoder Layers definitions
def decoder_generator(N,k,channel):
  # print(k,type(k))
  if channel == 'BSC':
    inputs_decoder = keras.Input(shape=N)
  elif channel == 'BAC':
    inputs_decoder = keras.Input(shape = N+4)
  x = Dense(units=S*2**(k), activation='relu')(inputs_decoder)
  # x = Dense(units=S * 2 ** (k), activation='relu')(x)
  # x = BatchNormalization()(x)
  outputs_decoder = Dense(units=2 ** k, activation='softmax')(x)
  ### Model Build
  model_dec = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder, name = 'decoder_model')
  return  model_dec



### Meta model Layers definitions
def meta_model_generator(k,channel,model_enc,model_dec):
  inputs_meta = keras.Input(shape = 2**k)
  encoded_bits = model_enc(inputs=inputs_meta)
  rounded_bits = Lambda(gradient_stopper)(encoded_bits)
  if channel == 'BSC':
    noisy_bits = Lambda(BSC_noise, arguments={'epsilon':train_epsilon}, output_shape=return_output_shape)(rounded_bits)
  elif channel == 'BAC':
    noisy_bits = Lambda(BAC_noise, arguments={'epsilon0':train_epsilon,'batch_size':batch_size})(encoded_bits)

  decoded_bits = model_dec(inputs=noisy_bits)
  meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits, name = 'meta_model')
  return meta_model

model_encoder = encoder_generator(N,k)
model_decoder = decoder_generator(N,k,channel)
meta_model = meta_model_generator(k,channel,model_encoder,model_decoder)

### Model print summary
# model_encoder.summary()
# model_decoder.summary()
# meta_model.summary()

### Compile our models
model_encoder.compile(loss=loss, optimizer=optimizer)
model_decoder.compile(loss=loss, optimizer=optimizer)
meta_model.compile(loss=loss, optimizer=optimizer, metrics=[bler_metric])

### Fit the model
history = meta_model.fit(In, In, epochs=epoch,verbose=2, shuffle=False, batch_size=batch_size)
# history = model_encoder.fit(In, c, epochs=epoch,verbose=1)
print("The model is ready to be used...")
# print(history.history)

### save Model
# model_decoder.save(f"./autoencoder/model_decoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
# model_encoder.save(f"./autoencoder/model_encoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
model_decoder.save(f"./autoencoder/model_decoder.h5")
model_encoder.save(f"./autoencoder/model_encoder.h5")

# Summarize history for loss

plt.semilogy(history.history['loss'],label='Loss (training data)')
plt.semilogy(history.history['bler_metric'],label='bler_metric (training data)')
# plt.semilogy(history.history['val_loss'],label='Loss (validation data)')
# plt.semilogy(history.history['binary_accuracy'],label='Accuracy (training data)')
# plt.semilogy(history.history['val_binary_accuracy'],label='Accuracy (validation data)')
plt.title('Loss function w.r.t. No. epoch')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.grid()

#####################################################
## TEST

C = utils.NN_encoder(k, N)
def BER_NN(nb_pkts=100):
  e0 = np.logspace(-3,0,15)
  e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
  e1 = [t for t in e0 if t <= 0.5]
  BER = test.saved_results(N, k)


  print("NN BER")
  t = time.time()
  BER['auto'] = utils.bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,channel)
  t = time.time()-t
  print(f"NN time = {t}s ========================")

  if MAP_test:
    print("MAP BER")
    t = time.time()
    BER['MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1)
    t = time.time()-t
    print(f"MAP time = {t}s =======================")

  test.plot_ber(BER, N,k,e0)
if len(sys.argv) > 5:
  if sys.argv[5] == 'BER':
    nb_pkts = int(sys.argv[6])
    BER_NN(nb_pkts)
plt.show()
