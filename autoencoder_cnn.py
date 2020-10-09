# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf
from mish import Mish as mish

import  ber_bler_calculator as test
import utils

import numpy as np
import matplotlib.pyplot as plt
import time

def sequence_generator(k):
  """
  parameter : number of bits to be sent
  return : all possible codewords
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
  """ parameter : Symboles à envoyer
      return : Symboles reçus, bruités """
  n = K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon
  # K.print_tensor(x, 'x')
  # K.print_tensor(n,'n')
  aux = tf.math.floormod(tf.add(x, tf.cast(n, tf.float32)), tf.cast(2, tf.float32))
  # K.print_tensor(aux, 'noisy')
  return aux # Signal transmis + Bruit

def BAC_noise(x, epsilon0, batch_size):
  """ parameter : Symboles à envoyer
      return : Symboles bruités + intervale crossover probability"""
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
  return output

def bler_metric(u_true,u_predict):
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
rounding = False

MAP_test = False

In = np.eye(2**k) # List of outputs of NN
In = np.tile(In,(rep,1))
# print(In)
batch_size = len(In)#int(len(In)/2)

########### Neural Network Generator ###################
optimizer = 'adam'
# optimizer_enc = keras.optimizers.Adam(lr=0.001)
# optimizer_dec = keras.optimizers.Adam(lr=0.001)
loss = 'categorical_crossentropy' #     'kl_divergence'            # or 'mse'

activation = 'softplus'
alpha=0.5 #only for leaky relu


### Encoder Layers definitions
def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape = 2**k)
  # x = Dense(units=S * 2 ** (k))(inputs_encoder)
  # x = LeakyReLU(alpha=alpha)(x)
  x = Dense(units=S*2**(k), activation=activation)(inputs_encoder)
  x = BatchNormalization(axis=1, momentum=0.0, center=False, scale=False)(x)
  outputs_encoder = Dense(units=N, activation='sigmoid')(x)
  ### Model Build
  model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder_model')
  return model_enc

### Decoder Layers definitions
def decoder_generator(N,k,channel):
  # print(k,type(k))
  if channel == 'BSC':
    inputs_decoder = keras.Input(shape=N)
  elif channel == 'BAC':
    inputs_decoder = keras.Input(shape = N+4)
  # x = Dense(units=S * 2 ** (k))(inputs_decoder)
  # x = LeakyReLU(alpha=alpha)(x)
  x = Dense(units=S * 2 ** (k), activation=activation)(inputs_decoder)
  x = BatchNormalization(axis=1, momentum=0.0, center=False, scale=False)(x)
  outputs_decoder = Dense(units=2 ** k, activation='softmax')(x)
  ### Model Build
  model_dec = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder, name = 'decoder_model')
  return  model_dec

### Meta model Layers definitions
def meta_model_generator(k,channel,model_enc,model_dec):
  inputs_meta = keras.Input(shape = 2**k)
  encoded_bits = model_enc(inputs=inputs_meta)
  if rounding:
    encoded_bits = Lambda(gradient_stopper)(encoded_bits)
  if channel == 'BSC':
    noisy_bits = Lambda(BSC_noise, arguments={'epsilon':train_epsilon}, output_shape=return_output_shape)(encoded_bits)
  elif channel == 'BAC':
    noisy_bits = Lambda(BAC_noise, arguments={'epsilon0':train_epsilon,'batch_size':batch_size})(encoded_bits)
  decoded_bits = model_dec(inputs=noisy_bits)
  meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits, name = 'meta_model')
  return meta_model

model_encoder = encoder_generator(N,k)
model_decoder = decoder_generator(N,k,channel)
meta_model = meta_model_generator(k,channel,model_encoder,model_decoder)

### Model print summary
model_encoder.summary()
model_decoder.summary()
# meta_model.summary()

### Compile our models
model_encoder.compile(loss=loss, optimizer=optimizer)
model_decoder.compile(loss=loss, optimizer=optimizer)
meta_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

### Fit the model
history = meta_model.fit(In, In, epochs=epoch,verbose=2, shuffle=False, batch_size=batch_size)
print("The model is ready to be used...")
# print(history.history)

### save Model
# model_decoder.save(f"./autoencoder/model_decoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
# model_encoder.save(f"./autoencoder/model_encoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
model_decoder.save(f"./autoencoder/model_decoder.h5")
model_encoder.save(f"./autoencoder/model_encoder.h5")

# Summarize history for loss

plt.semilogy(history.history['loss'],label='Loss (training data)')
# plt.semilogy(history.history['bler_metric'],label='bler_metric (training data)')
bler_accuracy = np.array(history.history['accuracy'])
plt.semilogy(1-bler_accuracy,label='BLER - metric (training data)')
# plt.semilogy(history.history['val_binary_accuracy'],label='Accuracy (validation data)')
plt.title('Loss function w.r.t. No. epoch')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.grid()

#####################################################
## TEST

def bit_error_rate_NN(N, k, C, N_iter_max, e0, e1, channel = 'BSC' ):
  print('*******************NN-Decoder********************************************')
  N_errors_mini = 100
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  count = 0
  for ep0 in e0:
    ber_row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        interval = np.zeros(4)
        interval[int(ep1*4)] = 1.0

        N_errors = 0
        N_iter = 0
        while N_iter < N_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits to be sent
          x = C[idx]  # coded bits
          y_bac = utils.BAC_channel(x, ep0, ep1)  # received symbols

          yh = np.reshape(np.concatenate((y_bac,interval),axis=0), [1, N+4]) if channel == 'BAC'  else np.reshape(y_bac, [1, N])
          u_nn = U_k[np.argmax(model_decoder(yh))]  #  NN Detector

          N_errors += utils.NbOfErrors(u, u_nn)  # bit error rate compute with NN
        ber_tmp = N_errors / (k * 1.0 * N_iter)  # bit error rate compute with NN
        ber_row.append(ber_tmp)

    ber[ep0] = ber_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    count+= 1
    print("{:.3f}".format(count / len(e0) * 100), '% completed ')
  return ber

def BER_NN(nb_pkts=100):
  e0 = np.logspace(-3, 0, 15)
  e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
  e1 = [t for t in e0 if t <= 0.5]

  one_hot = np.eye(2 ** k)
  C = np.round(model_encoder.predict(one_hot)).astype('int')
  aux = []
  for code in C.tolist():
    if code not in aux:
      aux.append(code)
  nb_repeated_codes = len(C) - len(aux)
  print('+++++++++++++++++++ Repeated Codes NN encoder = ', nb_repeated_codes)
  print('dist = ', sum([sum(codeword) for codeword in C]) * 1.0 / (N * 2 ** k))
  print('***************************************************************')

  if nb_repeated_codes ==0:
    metric = test.read_ber_file(N, k)
    BER = test.saved_results(metric, N, k)
    print("NN BER")
    t = time.time()
    BER['auto'] = bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,channel)
    t = time.time()-t
    print(f"NN time = {t}s ========================")
    print("metric['BKLC-NN'] = ", BER['auto'])

    if MAP_test:
      print("MAP BER")
      t = time.time()
      BER['MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1)
      t = time.time()-t
      print(f"MAP time = {t}s =======================")

    test.plot_ber(BER, N, k, e0)

  else:
    print('Bad codebook repeated codewords')


if len(sys.argv) > 5:
  if sys.argv[5] == 'BER':
    nb_pkts = int(sys.argv[6])
    BER_NN(nb_pkts)



plt.show()
