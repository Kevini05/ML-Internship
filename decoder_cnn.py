# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda

from keras.layers import Conv1D, MaxPooling1D, Flatten, Input, UpSampling1D, Reshape, AveragePooling1D
import keras.backend as K
import tensorflow as tf

import numpy as np
import time
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

def BSC_noise(x, epsilon_max,batch_size):
  """ parameter : Symboles à envoyer
      return : Symboles reçus, bruités """

  epsilon = np.random.uniform(low=0.0, high=epsilon_max, size=(batch_size, 1))
  two = tf.cast(2, tf.float32)
  n = tf.cast( K.random_uniform(shape=K.shape(x), minval=0.0, maxval=1.) < epsilon,tf.float32)
  y = tf.math.floormod(x+n,two)
  # K.print_tensor(epsilon, 'epsilon \n')
  # K.print_tensor(x, 'x \n')
  # K.print_tensor(n,'n \n')
  # K.print_tensor(y, 'y \n')
  return y # Signal transmis + Bruit

def BAC_noise(x, epsilon_1_max, batch_size):
  """ parameter : Symboles à envoyer
      return : Symboles bruités + intervale crossover probability"""
  epsilon_0_max = 0.07
  two = tf.cast(2, tf.float32)

  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon_0_max,tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon_1_max,tf.float32)
  n = n0*(x+1) + n1*x
  y = tf.math.floormod(x+n,two) # Signal transmis + Bruit

  # K.print_tensor(x, 'x\n')
  # # K.print_tensor(n0, 'n0\n')
  # # K.print_tensor(n1, 'n1\n')
  # K.print_tensor(n, 'n\n')
  # K.print_tensor(y, 'y\n')
  return y # Signal transmis + Bruit + Intervale


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

inputs_decoder = keras.Input(shape = N)
# x = Dense(units=S*2**(k), activation='relu')(inputs_decoder)
d1 = Dense(64)(inputs_decoder)
d2 = Reshape((16,4))(d1)
d3 = Conv1D(4,1,strides=1, activation='relu', padding='same')(d2)
d4 = AveragePooling1D()(d3)
flat = Flatten()(d4)
outputs_decoder = Dense(units=2**k, activation='softmax')(flat)
model_decoder = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)

### Meta model Layers definitions
inputs_meta = keras.Input(shape = N)
if channel == 'BSC':
  noisy_bits = Lambda(BSC_noise, arguments={'epsilon_max':train_epsilon}, output_shape=return_output_shape)(inputs_meta)
elif channel == 'BAC':
  noisy_bits = Lambda(BAC_noise, arguments={'epsilon_1_max':train_epsilon,'batch_size':batch_size})(inputs_meta)
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

def bit_error_rate_NN(N, k, C, Nb_sequences, e0, e1, channel = 'BSC' ):
  print('*******************NN-Decoder********************************************')
  # model_decoder = keras.models.load_model("./model/model_decoder_16_4_std.h5")
  print("Decoder Loaded from disk, ready to be used")
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(Nb_sequences/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        N_errors = 0
        N_errors_bler = 0
        N_iter = 0
        while N_iter < Nb_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1, size=(1, Nb_words)).tolist()[0]
          u = [U_k[a] for a in idx]
          x = [C[a] for a in idx]  # coded bits
          y_bac = [utils.BAC_channel(xi, ep0, ep1)  for xi in x]# received symbols

          yh = np.reshape(y_bac, [Nb_words, N]).astype(np.float64)
          u_nn = [U_k[idy] for idy in np.argmax(model_decoder.predict(yh),1) ]  #  NN Detector

          for i in range(len(u)):
            N_errors += np.sum(np.abs(np.array(u[i]) - np.array(u_nn[i])))  # bit error rate compute with NN
            N_errors_bler += np.sum(1.0*(u[i] != u_nn[i]))
        ber_row.append(N_errors / (k * 1.0 * Nb_sequences)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * Nb_sequences)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def BER_NN(nb_pkts=100):
  # e0 = np.logspace(-3, 0, 15)
  # e0 = np.linspace(0.001, 0.999, 11)
  e0 = np.concatenate((np.linspace(0.001, 0.2, 10, endpoint=False), np.linspace(0.2, 1, 8)), axis=0)
  e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
  e1 = [t for t in e0 if t <= 0.5]

  one_hot = np.eye(2 ** k)

  C = cn
  print('codebook\n', C)

  BER = test.read_ber_file(N, k, 'BER')
  BER = test.saved_results(BER, N, k)
  BLER = test.read_ber_file(N, k, 'BLER')
  BLER = test.saved_results(BLER, N, k,'BLER')
  print("NN BER")
  t = time.time()
  BER['decoder_cnn'],BLER['decoder_cnn'] = bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,channel)
  t = time.time()-t
  print(f"NN time = {t}s ========================")
  print("BER['auto-NN'] = ", BER['decoder_cnn'])
  print("BLER['auto-NN'] = ", BLER['decoder_cnn'])

  if MAP_test:
    print("MAP BER")
    t = time.time()
    BER['MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1)
    t = time.time()-t
    print(f"MAP time = {t}s =======================")
  utils.plot_BSC_BAC(f'BER Coding Mechanism N={N} k={k} - NN', BER, k / N)
  utils.plot_BSC_BAC(f'BLER Coding Mechanism N={N} k={k} - NN', BLER, k / N)

if len(sys.argv) > 5:
  if sys.argv[5] == 'BER':
    nb_pkts = int(sys.argv[6])
    BER_NN(nb_pkts)

plt.show()



