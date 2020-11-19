# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
import keras.backend as K

from keras.layers import Input, Dense, Layer, InputSpec
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, activations, initializers, constraints, Sequential
from keras import backend as K
from keras.constraints import UnitNorm, Constraint


import tensorflow as tf
from keras import regularizers
from keras.optimizers import Adam
from mish import Mish as mish

import  ber_bler_calculator as test
import utils

import numpy as np
import matplotlib.pyplot as plt
import time

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
  epsilon_0_max = 0.2
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

def gradient_stopper(x):
  output = tf.stop_gradient(tf.math.round(x)-x)+x
  return output

###### \Python3\python.exe autoencoder_optimal.py BAC 16 4 300
###### \Python3\python.exe autoencoder_optimal.py BAC 16 4 300 BER 100
#Parameters
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
epoch = int(sys.argv[4])

rep = 500
train_epsilon = 0.07
S = 3
rounding = True
MAP_test = False

In = np.eye(2**k) # List of outputs of NN
In = np.tile(In,(rep,1))
# print(In)
batch_size = int(len(In)/1)
####################################################################################################
########### Neural Network Generator ###################
# LearningRate = 0.001
# Decay1 = LearningRate / 100
# Decay2 = LearningRate / 1000
# optimizer =  Adam(lr=LearningRate)
# optimizer_enc = Adam(lr=LearningRate,decay=Decay1)
# optimizer_dec = Adam(lr=LearningRate,decay=Decay2)
optimizer = 'adam'
# optimizer = keras.optimizers.SGD(lr=0.03, nesterov=True)
# optimizer = keras.optimizers.Adagrad(learning_rate=0.01)

optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.001)

loss = 'categorical_crossentropy' #'categorical_crossentropy'  #'kl_divergence'          # 'mse'

activation = 'softplus'


class UncorrelatedFeaturesConstraint(Constraint):

  def __init__(self, encoding_dim, weightage=1.0):
    self.encoding_dim = encoding_dim
    self.weightage = weightage

  def get_covariance(self, x):
    x_centered_list = []

    for i in range(self.encoding_dim):
      x_centered_list.append(x[:, i] - K.mean(x[:, i]))

    x_centered = tf.stack(x_centered_list)
    covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)

    return covariance

  # Constraint penalty
  def uncorrelated_feature(self, x):
    if (self.encoding_dim <= 1):
      return 0.0
    else:
      output = K.sum(K.square(
        self.covariance - tf.math.multiply(self.covariance, K.eye(self.encoding_dim))))
      return output

  def __call__(self, x):
    self.covariance = self.get_covariance(x)
    return self.weightage * self.uncorrelated_feature(x)

### Encoder Layers definitions
inputs_encoder = keras.Input(shape = 2**k)
hidden_enc = Dense(units=S*2**(k), activation=activation, activity_regularizer=UncorrelatedFeaturesConstraint(S*2**(k), weightage = 1.))(inputs_encoder)
batch_enc = BatchNormalization()(hidden_enc)
outputs_encoder = Dense(units=N, activation='sigmoid')(batch_enc)
### Model Build
model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder_model')

### Decoder Layers definitions
inputs_decoder = keras.Input(shape = N)
hidden_dec = Dense(units=S * 2 ** (k), activation=activation, kernel_constraint=UnitNorm(axis=0))(inputs_decoder)
# hidden_dec = dense(inputs_decoder, transpose=False)
batch_dec = BatchNormalization()(hidden_dec)
outputs_decoder = Dense(units=2**k, activation='softmax')(batch_dec)
### Model Build
model_dec = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder, name = 'decoder_model')

### Meta model Layers definitions
inputs_meta = keras.Input(shape = 2**k)
encoded_bits = model_enc(inputs=inputs_meta)
x = Lambda(gradient_stopper,name='rounding_layer')(encoded_bits)
if channel == 'BSC':
  noisy_bits = Lambda(BSC_noise, arguments={'epsilon_max':train_epsilon,'batch_size':batch_size}, name='noise_layer')(x)
elif channel == 'BAC':
  noisy_bits = Lambda(BAC_noise, arguments={'epsilon_1_max':train_epsilon,'batch_size':batch_size}, name='noise_layer')(x)
decoded_bits = model_dec(inputs=noisy_bits)
meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits,name = 'meta_model')

### Model print summary
# model_encoder.summary()
# model_decoder.summary()
# meta_model.summary()

### Compile our models
model_enc.compile(loss='binary_crossentropy', optimizer=optimizer)
model_dec.compile(loss=loss, optimizer=optimizer)
meta_model.compile(loss=loss,loss_weights=1, optimizer=optimizer, metrics=['accuracy'])

### Fit the model
history = meta_model.fit(In, In, epochs=epoch,verbose=2, shuffle=True, batch_size=batch_size)
print("The model is ready to be used...")
# print(history.history)

### save Model
# model_decoder.save(f"./autoencoder/model_decoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
# model_encoder.save(f"./autoencoder/model_encoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
model_dec.save(f"./autoencoder/model_decoder.h5")
model_enc.save(f"./autoencoder/model_encoder.h5")

# Summarize history for loss

plt.semilogy(history.history['loss'],label='Crossentropy (training data)')
bler_accuracy = np.array(history.history['accuracy'])

plt.semilogy(1-bler_accuracy,label='BLER - metric (training data)')
# plt.semilogy(history.history['val_binary_accuracy'],label='Accuracy (validation data)')
plt.title('Loss function w.r.t. No. epoch')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.grid()


# tf.keras.utils.plot_model(meta_model, to_file="meta_model.png", show_shapes=False, show_layer_names=True)
# tf.keras.utils.model_to_dot(meta_model)
#####################################################
## TEST

def bit_error_rate_NN(N, k, C, Nb_sequences, e0, e1, channel = 'BSC' ):
  print('*******************NN-Decoder********************************************')
  # model_decoder = keras.models.load_model("autoencoder/model_decoder_bsc_16_8_array.h5")
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
          u_nn = [U_k[idy] for idy in np.argmax(model_dec.predict(yh),1) ]  #  NN Detector

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

  C = np.round(model_enc.predict(one_hot)).astype('int')
  print('codebook\n', C)
  aux = []
  for code in C.tolist():
    if code not in aux:
      aux.append(code)
  nb_repeated_codes = len(C) - len(aux)
  print('+++++++++++++++++++ Repeated Codes NN encoder = ', nb_repeated_codes)
  print('dist = ', sum([sum(codeword) for codeword in C]) * 1.0 / (N * 2 ** k))
  print('***************************************************************')

  if nb_repeated_codes ==0:
    BER = test.read_ber_file(N, k, 'BER')
    BER = test.saved_results(BER, N, k)
    BLER = test.read_ber_file(N, k, 'BLER')
    BLER = test.saved_results(BLER, N, k,'BLER')
    print("NN BER")
    t = time.time()
    BER['auto-non-inter'],BLER['auto-non-inter'] = bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,channel)
    t = time.time()-t
    print(f"NN time = {t}s ========================")
    print("BER['auto-NN'] = ", BER['auto-non-inter'])
    print("BLER['auto-NN'] = ", BLER['auto-non-inter'])

    if MAP_test:
      print("MAP BER")
      t = time.time()
      BER['MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1)
      t = time.time()-t
      print(f"MAP time = {t}s =======================")
    utils.plot_BSC_BAC(f'BER Coding Mechanism N={N} k={k} - NN', BER, k / N)
    utils.plot_BSC_BAC(f'BLER Coding Mechanism N={N} k={k} - NN', BLER, k / N)
  else:
    print('Bad codebook repeated codewords')

if len(sys.argv) > 5:
  if sys.argv[5] == 'BER':
    nb_pkts = int(sys.argv[6])
    BER_NN(nb_pkts)

plt.show()

