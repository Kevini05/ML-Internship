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
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from mish import Mish as mish

import  ber_bler_calculator as test
import utils

import numpy as np
import matplotlib.pyplot as plt
import time

def BSC_noise(inputs, train_epsilon , batch_size):
  """ parameter : Symboles à envoyer
      return : Symboles reçus, bruités """
  x = tf.cast(inputs[0],tf.float64)
  interval = inputs[1]


  epsilon = K.reshape(tf.cast(K.argmax(interval)/3*0.15+0.05, tf.float32),shape=(batch_size, 1))

  n = tf.cast( K.random_uniform(shape=K.shape(x), minval=0.0, maxval=1.0) < epsilon,tf.float64)
  two = tf.cast(2, tf.float64)
  y = tf.math.floormod(x+n,two)

  # K.print_tensor(epsilon, 'epsilon \n')
  # K.print_tensor(interval, 'interval \n')

  # print('x', K.shape(x))
  # print('epsilon', K.shape(epsilon))
  # print(K.shape(x))
  # K.print_tensor(x, 'x \n')
  # K.print_tensor(n,'n \n')
  # K.print_tensor(y, 'y \n')
  # return tf.concat([y,interval],1)
  # return tf.concat([x,interval],1)
  return y # Signal transmis + Bruit

def BAC_noise(inputs, epsilon_0_max, batch_size):
  """ parameter : Symboles à envoyer
      return : Symboles bruités + intervale crossover probability"""
  x = tf.cast(inputs[0], tf.float64)
  interval = inputs[1]
  epsilon_1_max = 0.002

  epsilon0 = K.reshape(tf.cast(K.argmax(interval)/3*0.15+0.05, tf.float32), shape=(batch_size, 1))

  two = tf.cast(2, tf.float64)

  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon0,tf.float64)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon_1_max,tf.float64)
  n = tf.math.floormod(n0*(x+1) + n1*x, two)
  y = tf.math.floormod(x+n,two) # Signal transmis + Bruit

  # K.print_tensor(x, 'x\n')
  # # K.print_tensor(tf.math.round(x), 'x-rounded\n')
  # # K.print_tensor(n0, 'n0\n')
  # # K.print_tensor(n1, 'n1\n')
  # K.print_tensor(n, 'n\n')
  # K.print_tensor(y, 'y\n')
  # return tf.concat([y,interval],1) # Signal transmis + Bruit + Intervale
  return y  # Signal transmis + Bruit

def return_output_shape(input_shape):
  print('*****************************************************************', input_shape)
  return input_shape

def gradient_stopper(x):
  output = tf.stop_gradient(tf.math.round(x)-x)+x
  return output

###### \Python3\python.exe autoencoder.py BAC 16 4 300
###### \Python3\python.exe autoencoder.py BAC 16 4 300 BER 100
#Parameters
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
epoch = int(sys.argv[4])

rep = 80
train_epsilon = 0.2
S = 5
rounding = True
MAP_test = False

In = np.eye(2**k) # List of outputs of NN
In = np.tile(In,(rep,1))
batch_size = 256
u_k = utils.symbols_generator(k)
# print(u_k)
U_k = np.tile(u_k,(rep,1))

# Interval = np.reshape(np.tile(np.eye(4),int(len(In)/4)), (len(In), 4))
Interval = []
idx =[0.80,0.10,0.08,0.02]
# idx =[0.25,0.25,0.25,0.25] # for proofs of BSC Noise layer
for i in range(4):
  for j in range(round(len(In)*idx[i])):
    Interval.append(np.eye(4)[i].tolist())
Interval = np.reshape(Interval, (len(In), 4))
# print('Interval \n',Interval)
####################################################################################################
########### Neural Network Generator ###################
# LearningRate = 0.001
# Decay1 = LearningRate / 100
# Decay2 = LearningRate / 1000
# optimizer =  Adam(lr=LearningRate)
# optimizer_enc = Adam(lr=LearningRate,decay=Decay1)
# optimizer_dec = Adam(lr=LearningRate,decay=Decay2)
optimizer = 'adam'
# optimizer = keras.optimizers.SGD(lr=0.01, nesterov=True)
# optimizer_dec = keras.optimizers.SGD(lr=0.05, nesterov=True)
# optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.001)
loss = 'categorical_crossentropy' #'categorical_crossentropy'  #'kl_divergence'          # 'mse'

activation = 'relu'

### Encoder Layers definitions
def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape=k, name='input_encoder')
  inputs_interval = keras.Input(shape=4, name='input_interval_encoder')
  merged_inputs = keras.layers.Concatenate(axis=1,name='merge')([inputs_encoder, inputs_interval])
  x = Dense(units=512, activation=activation)(merged_inputs)
  x = BatchNormalization()(x)
  x = Dense(units=256, activation=activation)(x)
  x = BatchNormalization()(x)
  outputs_encoder = Dense(units=N, activation='sigmoid')(x)
  ### Model Build
  model_enc = keras.Model(inputs=[inputs_encoder,inputs_interval], outputs=outputs_encoder, name = 'encoder_model')
  return model_enc

### Decoder Layers definitions
def decoder_generator(N,k):
  # print(k,type(k))

  inputs_decoder = keras.Input(shape = N, name='input_decoder')
  inputs_interval = keras.Input(shape=4, name='input_interval_decoder')
  merged_inputs = keras.layers.Concatenate(axis=1,name='merge')([inputs_decoder, inputs_interval])
  x = Dense(units=S*2**k, activation=activation)(merged_inputs)
  x = BatchNormalization()(x)
  outputs_decoder = Dense(units=2 ** k, activation='softmax')(x)
  ### Model Build
  model_dec = keras.Model(inputs=[inputs_decoder,inputs_interval], outputs=outputs_decoder, name = 'decoder_model')
  return  model_dec

### Meta model Layers definitions
def meta_model_generator(k,channel,model_enc,model_dec,round,epsilon_t):
  inputs_meta = keras.Input(shape=k, name='input_meta')
  inputs_interval = keras.Input(shape=4, name='input_interval_meta')

  encoded_bits = model_enc(inputs=[inputs_meta,inputs_interval])
  if round:
    x = Lambda(gradient_stopper, name='rounding_layer')(encoded_bits)
  else:
    x = encoded_bits
  if channel == 'BSC':
    noisy_bits = Lambda(BSC_noise, arguments={'train_epsilon':epsilon_t,'batch_size':batch_size}, name='noise_layer')([x,inputs_interval])
  elif channel == 'BAC':
    noisy_bits = Lambda(BAC_noise, arguments={'epsilon_0_max':epsilon_t,'batch_size':batch_size}, name='noise_layer')([x,inputs_interval])
  decoded_bits = model_dec(inputs=[noisy_bits,inputs_interval])
  meta_model = keras.Model(inputs=[inputs_meta,inputs_interval], outputs=decoded_bits,name = 'meta_model')
  return meta_model

epoch_pretrain = 1000
epoch_encoder = 200
epoch_decoder = 500
pretrain_epsilon = 0.03
encoder_epsilon = 0.03
decoder_epsilon = 0.03
pretraining = True

# # pretraining
if pretraining:
  model_encoder = encoder_generator(N,k)
  model_decoder = decoder_generator(N,k)
  meta_model = meta_model_generator(k,channel,model_encoder,model_decoder, False, pretrain_epsilon)
  ### Compile our models
  model_encoder.compile(loss='binary_crossentropy', optimizer=optimizer)
  model_decoder.compile(loss=loss, optimizer=optimizer)
  meta_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  ### Fit the model
  history = meta_model.fit([U_k, Interval], In, epochs=epoch_pretrain, verbose=2, shuffle=True, batch_size=batch_size, validation_data=([U_k, Interval], In))
  model_decoder.save(f"./autoencoder/model_decoder_array_non-round_interval_{channel}_rep-{rep}_epsilon-{pretrain_epsilon}_layerSize_{S}_epoch-{epoch_pretrain}_k_{k}_N-{N}.h5")
  model_encoder.save(f"./autoencoder/model_encoder_array_non-round_interval_{channel}_rep-{rep}_epsilon-{pretrain_epsilon}_layerSize_{S}_epoch-{epoch_pretrain}_k_{k}_N-{N}.h5")
  loss_values = history.history['loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
else:
  model_encoder = keras.models.load_model("autoencoder/model_encoder_non-round_interval_BSC_rep-1000_epsilon-0.002_layerSize_5_epoch-2000_k_4_N-16.h5")
  model_decoder = keras.models.load_model("autoencoder/model_decoder_non-round_interval_BSC_rep-1000_epsilon-0.002_layerSize_5_epoch-2000_k_4_N-16.h5")
  loss_values = []
  accuracy = []
  val_accuracy = []

# Fine tunning
for a in range(epoch):
  print(f"*****************Epoch  {a}/{epoch}*********************************")
  if a%2==0:
    model_decoder.trainable=False
    model_encoder.trainable=True
    epoch_int = epoch_encoder
    train_epsilon = encoder_epsilon
  else:
    model_decoder.trainable = True
    model_encoder.trainable = False
    epoch_int = epoch_decoder
    train_epsilon = decoder_epsilon
  ### Compile our models
  meta_model = meta_model_generator(k, channel, model_encoder, model_decoder, True, train_epsilon)
  model_encoder.compile(loss='binary_crossentropy', optimizer=optimizer)
  model_decoder.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-5))
  meta_model.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-5), metrics=['accuracy'])

  ### Fit the model
  history = meta_model.fit([U_k,Interval], In, epochs=epoch_int, verbose=2, shuffle=True, batch_size=batch_size,validation_data=([U_k, Interval], In))
  loss_values += history.history['loss']
  accuracy += history.history['accuracy']
  val_accuracy += history.history['val_accuracy']
print("The model is ready to be used...")

### Model print summary
model_encoder.summary()
model_decoder.summary()
meta_model.summary()


### save Model
# model_decoder.save(f"./autoencoder/model_decoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
# model_encoder.save(f"./autoencoder/model_encoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
model_decoder.save(f"./autoencoder/model_decoder.h5")
model_encoder.save(f"./autoencoder/model_encoder.h5")

# Summarize history for loss

plt.semilogy(loss_values,label='Crossentropy (training data)')
bler_accuracy = 1-np.array(accuracy)
bler_val_accuracy = 1-np.array(val_accuracy)
plt.semilogy(bler_accuracy,label='BLER - metric (training data)')
plt.semilogy(bler_val_accuracy,label='BLER - metric (validation data)')
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
    interval = np.zeros(4)
    interval[int(ep0/0.15*3-1) if ep0 < train_epsilon else 3] = 1.0
    # print('epsilon: ',ep0,' Interval: ',interval)
    inter_list = np.array(np.tile(interval, (2 ** k, 1)))
    C = np.round(model_encoder.predict([np.array(u_k), inter_list])).astype('int')

    inter_list = np.array(np.tile(interval, (Nb_words, 1)))
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

          y_bac = [utils.BAC_channel(xi, ep0, ep1) for xi in x]  # received symbols

          yh = np.reshape(y_bac, [Nb_words, N]).astype(np.float64)
          u_nn = [U_k[idy] for idy in np.argmax(model_decoder.predict([yh,inter_list]),1) ]  #  NN Detector

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
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
  e1 = [t for t in e0 if t <= 0.5]

  inter_list = np.array(np.tile([0,0,0,1], (2 ** k, 1)))
  C = np.round(model_encoder.predict([np.array(u_k), inter_list])).astype('int')
  print('codebook \n', C)
  print('codebook C is Linear? ', utils.isLinear(C))
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
    BER['auto-array-inter'],BLER['auto-array-inter'] = bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,channel)
    t = time.time()-t
    print(f"NN time = {t}s ========================")

    print("BER['auto-array-inter'] = ", BER['auto-array-inter'])
    print("BLER['auto-array-inter'] = ", BLER['auto-array-inter'])

    if MAP_test:
      print("MAP BER")
      t = time.time()
      BER['MAP'] = utils.bit_error_rate(k, C, 1000, e0, e1)
      t = time.time()-t
      print(f"MAP time = {t}s =======================")

      print("NN BLEP")
      t = time.time()
      BLER['auto_BLEP'] = utils.block_error_probability(N, k, C, e0, e1)
      t = time.time() - t
      print(f"NN time = {t}s ========================")

    utils.plot_BSC_BAC(f'BER Coding Mechanism N={N} k={k} - NN Interval', BER, k / N)
    utils.plot_BSC_BAC(f'BLER Coding Mechanism N={N} k={k} - NN Interval', BLER, k / N)
  else:
    print('Bad codebook repeated codewords')

if len(sys.argv) > 5:
  if sys.argv[5] == 'BER':
    nb_pkts = int(sys.argv[6])
    BER_NN(nb_pkts)

plt.show()


# \Python3\python.exe autoencoder_array_interval_alt-training.py BSC 16 8 10 BER 100000