# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.layers import Conv1D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf

import utils
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

###### \Python3\python.exe encoder_cnn.py BAC 16 4 200
#Parameters
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
epoch = int(sys.argv[4])
G,infoBits = polar.polar_generator_matrix(N, k, channel, 0.1)

k = len(G)      #Nombre de bits � envoyer
N = len(G[1])   #codeword length

epoch = int(sys.argv[4])

rep = 500
train_epsilon = 0.07
S = 3

################### Coding
U_k = utils.symbols_generator(k)  # all possible messages
cn = utils.matrix_codes(U_k, k, G, N)
# print('codebook',np.array(cn))
print('size C: ',len(cn), 'size Cn: ', len(cn[0]))
c = np.array(cn)
c = np.tile(c,(rep,1))
print(type(c[0]))

In = np.eye(2**k) # List of outputs of NN
In = np.tile(In,(rep,1))
# print(In)
batch_size = len(In)#int(len(In)/2)

########### Neural Network Generator ###################
optimizer = 'adam'
optimizer_enc = keras.optimizers.Adam(lr=0.001)
loss = "mse"                # or 'mse'

### Encoder Layers definitions
def encoder_generator(N,k):
  print(k)
  inputs_encoder = keras.Input(shape = 2**k)
  print(K.shape(inputs_encoder),'*******************************************')
  # x = Conv1D(32, 2, activation='relu', input_shape=(2**k,N))(inputs_encoder)
  x = Dense(units=S*2**(k), activation='selu')(inputs_encoder)
  # x = BatchNormalization(axis=1, momentum=0.0, center=False, scale=False)(x)
  outputs_encoder = Dense(units=N, activation='sigmoid')(x)
  ### Model Build
  model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder_model')
  return model_enc

###Meta model encoder
def meta_model_generator(k,model_enc):
  inputs_meta = keras.Input(shape = 2**k)
  encoded_bits = model_enc(inputs=inputs_meta)
  rounded_bits = Lambda(gradient_stopper)(encoded_bits)
  meta_model = keras.Model(inputs=inputs_meta, outputs=rounded_bits)
  return meta_model


model_encoder = encoder_generator(N,k)
meta_model = meta_model_generator(k,model_encoder)
meta_model.compile(loss=loss, optimizer=optimizer, metrics=[bler_metric])

### Model print summary
# model_encoder.summary()
# meta_model.summary()

### Compile our models
model_encoder.compile(loss=loss, optimizer=optimizer)

### Fit the model
history = meta_model.fit(In, c, epochs=epoch,verbose=2, shuffle=False, batch_size=batch_size)
print("The model is ready to be used...")
# print(history.history)

### save Model
model_encoder.save(f"./autoencoder/model_encoder.h5")

# Summarize history for loss

plt.semilogy(history.history['loss'],label='Loss (training data)')
plt.title('Loss function w.r.t. No. epoch')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.grid()

#####################################################
## TEST
one_hot = np.eye(2 ** k)
C_NN = np.round(model_encoder.predict(one_hot)).astype('int')
# print('BKLC',np.array(cn))
# print('NN-encoder',C_NN)
# print('dif',C_NN-np.array(cn))
plt.show()
