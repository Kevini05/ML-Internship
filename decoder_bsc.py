# Import deep learning library
from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dense, Lambda
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import random as rd

#Parameters
k = 8 #Nombre de bits � envoyer
N = 16 #codeword length
G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) # Matrice g�n�ratrice

def FEC_encoder(b,G):
  """
  Entr�e : sequence de bits 
  Sortie : sequence g�n�r�e gr�ce � la matrice g�n�ratrice
  """ 
  c = np.dot(np.transpose(G),np.transpose(b))%2
  return c

def NbOfErrors(a,b):
  """
  Entr�e : 2 bit's arrays
  Sortie : numero de bit qui ne sont pas �gaux avec un distance de 1e-2
  """
  NbErrors = 0
  for i in range(len(a)):
    if np.abs(a[i]-b[i])>1e-2:
      NbErrors += 1
  return NbErrors
  
def codewords_generator(k):
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

def BAC_channel(x, epsilon0, epsilon1):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = [0] * len(x)
  # print('e0 ',epsilon0)
  # print('e1 ',epsilon1)
  for i in range(len(x)):
    rdnm = rd.randint(0, 1000) / (1000.0)
    n[i] = x[i] ^ 1 if (rdnm <= epsilon0 and x[i] == 0) or (rdnm <= epsilon1 and x[i] == 1) else x[i]
  return n  # Signal transmis + Bruit

messages = codewords_generator(k)

for i in range(len(messages)):
  c=FEC_encoder(messages[i],G) # The list of inputs of NN

def BSC_channel(x, epsilon):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = 1 if rd.randint(0, 1000) / (1000.0) <= epsilon  else 0
  return (n+x)%2  # Signal transmis + Bruit

def return_output_shape(input_shape):
  return input_shape

def compose_model(layers):
  model = Sequential()
  for layer in layers:
    model.add(layer)
  return model

messages = codewords_generator(k)
c = []
for i in range(len(messages)):
  c.append(FEC_encoder(messages[i],G)) # The list of inputs of NN

print('size C: ',len(c), 'size Cn: ', len(c[0]))

c=np.array(c)
In = np.eye(2**k) # List of outputs of NN


print(len(In))

########### Neural Network Generator ###################

optimizer = 'adam'           
loss = 'categorical_crossentropy'                # or 'mse'

train_epsilon = 0.1
# Declare the layers

noise_layers = [Lambda(BSC_channel, arguments={'epsilon':train_epsilon}, input_shape=(N,), output_shape=return_output_shape, name="noise")]
noise = compose_model(noise_layers)
noise.compile(optimizer=optimizer, loss=loss)

# meta_layer = Lambda(BSC_channel, arguments={'epsilon':train_epsilon})

decoder_layers = [Dense(units=2**(k+1), input_dim=len(G[0]), activation='relu'),
                  Dense(units=2**k, activation='softmax')]

# Creation of decoder model
model_decoder = compose_model(decoder_layers)
model_decoder.compile(loss=loss, optimizer=optimizer)

# Creation of meta model
meta_layers = noise_layers+decoder_layers
model_meta = compose_model(meta_layers)

# Configure an optimizer used to minimize the loss function
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile our model
model_meta.compile(loss=loss, optimizer=optimizer)

# Fit the model
history=model_meta.fit(c, In, epochs=9000,verbose=0)
model_meta.summary()
model_decoder.summary()
print("The model is ready to be used...")


### save the model
# model_decoder.save('C:/Users/ke.ibarra/Desktop/ML-Internship-master')
### serialize model to JSON

model_json = model_decoder.to_json()
with open("./model/model_decoder_bsc.json", "w") as json_file:
     json_file.write(model_json)
# serialize weights to HDF5
model_decoder.save_weights("./model/weights_decoder_bsc.h5")

# Summarize history for loss
plt.semilogy(history.history['loss'])
plt.title('Loss function w.r.t. Epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'])
plt.show()

# Make prediction using the model decoder
X = np.reshape(FEC_encoder([0,0,0,1,0,1,0,0.1],G),[1,16])
print(c[0:1].shape)
Y_hat = model_decoder.predict(X)
id = np.argmax(Y_hat)
print(messages[id])
