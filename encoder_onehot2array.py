# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy

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

def loss_fn(y_true, y_pred):
  binary_neck_loss = tf.abs(0.5 - tf.abs(0.5 -y_pred))
  return K.mean(binary_neck_loss, axis=-1)

def bler_metric(u_true,u_predict):
  return K.mean(K.not_equal(K.argmax(u_true, 1),K.argmax(u_predict, 1)))

def onehot2sequence(x, U_k):
  u_k = tf.constant(U_k)
  output = tf.gather(u_k,K.argmax(x, axis = 1))
  # K.print_tensor(u_k, 'u_k\n')
  # K.print_tensor(x, 'x\n')
  # K.print_tensor(output, 'output\n')
  return output
###### \Python3\python.exe encoder.py BAC 16 4 200
#Parameters
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
epoch = int(sys.argv[4])
G,infoBits = polar.polar_generator_matrix(N, k, channel, 0.1)

k = len(G)      #Nombre de bits � envoyer
N = len(G[1])   #codeword length

epoch = int(sys.argv[4])

rep = 4
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
batch_size = len(In)#int(len(In)/2)
Interval = np.reshape(np.tile(np.eye(4),int(len(In)/4)), (len(In), 4))

# print(In)
# print(Interval,len(Interval))

########### Neural Network Generator ###################
optimizer = 'adam'
optimizer_enc = keras.optimizers.Adam(lr=0.001)
loss = "mse"                # or 'mse'

### Encoder Layers definitions

inputs_encoder = keras.Input(shape=k, name='input_encoder')
x = Dense(units=S*(2**k), activation='selu',name='selu')(inputs_encoder)
x = BatchNormalization()(x)
outputs_encoder = Dense(units=N, activation='sigmoid',name='sigmoid')(x)
### Model Build
model_encoder = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder_model')

u_k = utils.symbols_generator(k)
###Meta model encoder
inputs_meta = keras.Input(shape = 2**k, name='input_meta')
inputs_sequence = Lambda(onehot2sequence,arguments={'U_k':u_k},name='one-hot_2_sequence')(inputs_meta)
encoded_bits = model_encoder(inputs=inputs_sequence)
rounded_bits = Lambda(gradient_stopper,name='Rounded')(encoded_bits)
meta_model = keras.Model(inputs=inputs_meta, outputs=[rounded_bits,encoded_bits])

meta_model.compile(loss=['mse',loss_fn], optimizer=optimizer, metrics=[bler_metric,'accuracy'])

### Model print summary
model_encoder.summary()
meta_model.summary()

### Compile our models
model_encoder.compile(loss=loss_fn, optimizer=optimizer)

### Fit the model
history = meta_model.fit(In, c, epochs=epoch,verbose=1, shuffle=False, batch_size=batch_size)
print("The model is ready to be used...")
# print(history.history)

### save Model
# model_encoder.save(f"./autoencoder/model_encoder_{channel}_rep-{rep}_epsilon-{train_epsilon}_layerSize_{S}_epoch-{epoch}_k_{k}_N-{N}.h5")
model_encoder.save(f"./autoencoder/model_encoder.h5")

# Summarize history for loss

plt.semilogy(history.history['loss'],label='Loss (training data)')
# plt.semilogy(history.history['bler_metric'],label='bler_metric (training data)')
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
one_hot = np.eye(2**k)
inter = np.zeros(4)
inter[1] = 1
inter_list = np.array(np.tile(inter,(2**k,1)))
# input = np.concatenate((one_hot,inter),axis=1)


C_NN = np.round(model_encoder.predict(u_k)).astype('int')
# C_NN = model_encoder.predict([one_hot,inter])

# for i in range(len(cn)):
  # print('BKLC',cn[i])
  # print('NN-encoder',C_NN[i])
print('dif \n',np.round(C_NN)-cn)
plt.show()
