# Import deep learning library
from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dense
from keras.layers import GaussianNoise
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special as ss
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

def mod_BPSK(b):
  """
  Entr�e : sequence de bits 
  Sortie : Symboles � envoyer
  """
  c = [2*b[j]-1 for j in range(len(b))] #Symboles � envoyer
  return(c)

def dem_BPSK(y):
  """
  Entr�e : Symboles re�ues
  Sortie : s�quence de bits 
  """
  d = [(y[j]+1)/2 for j in range(len(y))] #Symboles � envoyer  #sequence de bits
  return(d)

def BEP(EbN0):
  """
  Entr�e : EbN0 value 
  Sortie : Bit error probability
  """
  return 0.5*ss.erfc(math.sqrt(EbN0))

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


messages = codewords_generator(k)
cn = [] # The list of inputs of NN
for i in range(len(messages)):
  c=FEC_encoder(messages[i],G)
  cn.append(mod_BPSK(c))

cn=np.array(cn)
In = np.eye(2**k) # List of outputs of NN

print(len(cn), len(cn[0]))
print(len(In))

########### Neural Network Generator ###################

optimizer = 'adam'           
loss = 'categorical_crossentropy'                # or 'mse'

# Build our model
model_meta = Sequential()
model_decoder = Sequential()

EbN0_dB = 1 #dB
train_SNR_Es = EbN0_dB + 10*np.log10(k/N)
Sigma_n = np.sqrt(1/(2*10**(train_SNR_Es/10))) # Noise Standard Deviation

# Declare the layers
meta_layer = GaussianNoise(Sigma_n)
layers = [Dense(units=2**(k+1), input_dim=len(G[0]), activation='relu'),
          Dense(units=2**k, activation='softmax')]

# Add the layers to the model
model_decoder.add(layers[0])
model_decoder.add(layers[1])

model_decoder.compile(loss=loss, optimizer=optimizer)

model_meta.add(meta_layer)
model_meta.add(layers[0])
model_meta.add(layers[1])

# Configure an optimizer used to minimize the loss function
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile our model
model_meta.compile(loss=loss, optimizer=optimizer)

# Fit the model
history=model_meta.fit(cn, In, epochs=2**5,verbose=0)
model_meta.summary()
model_decoder.summary()

print("The model is ready to be used...")

# # Mount drive folder as disk
# from google.colab import drive
# drive.mount("/content/gdrive", force_remount=True)
#
# # serialize model to JSON
# model_json = model_decoder.to_json()
# with open("model_ducuara_ibarra_rahmat.json", "w") as json_file:
#      json_file.write(model_json)
# # serialize weights to HDF5
# model_decoder.save_weights("weights_ducuara_ibarra_rahmat.h5")
#
# print("Saved model to disk")