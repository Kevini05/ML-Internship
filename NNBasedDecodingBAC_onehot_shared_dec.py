#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import time
"""
Created on Wed Jun 17 12:10:27 2020

@author: meryem
"""

"""
Created on Mon Oct 21 17:43:35 2019

@author: meryem
""" 
import numpy as np 
from keras.layers.core import Dense, Lambda 
import matplotlib.pyplot as plt 
import file_definitions_BAC as d_f
from keras import Model, Input
from keras.utils import plot_model
import tensorflow as tf
import keras.backend as K
from keras.layers.normalization import BatchNormalization  
####################################################################
#              General parameters 
####################################################################

new_save_enabled = 1 # Need to save the MAP if it is newly computed  
MAP_activ = 0        # Need to run with the MAP decoder
quick_load = 0      # Need to run with the previous MAP curve
plot_losses = 1      # Whether to plot or not the losses   
plot_ber = 1         # plot / no plot BER
plot_fer = 0         # plot / no plot FER
our_polar  = 1       # which polar code to use (Malab (our) or Hoydis's)
diff_q_p = 1         # Use different p and q for validation 
plot_3D = 0          # Plot the whole 3D curve over p and q 
plot_diagonal = 1    # plot the case where p = q
test_ber = 1         # test the NN on a real communication chain 
 
k = 4                # number of information bits
N = 16               # code length 
  
alpha = 0.00
train_q = 0.07

train_p_1 = 0.01 
train_p_3 = 0.4     # Training crossover probability q
train_p_2 = 0.2
nb_p = 2             # Number of training crossovers probabilities


build_p = 0.01       # Probability to build the frozen bits sets 

nb_epoch = 2**10     # Number of learning epochs
design = [64]        # Each list entry defines the number of nodes in a layer
batch_size_norm = 100; # Number of copies of the dataset per regular batch
batch_size_exp = batch_size_norm*(2**k);   # Number of sequences per regular batch
batch_size = (2**k)*batch_size_norm;  # Total number of batches after copying 

optimizer = 'adam'   # Optimizer used        
shuffle_var = True
 
# Testing parameters    
num_words = 1000    # Number of transmitted frames of k bits
N_errors_mini = 100  # Minimal number of errors required before switching
N_iter_max = 50     # Number of iterations in order to compute errors

# P crossover probability
p_start = 10**-4    
p_stop= 1 -  train_q -0.001       
p_points = 10

# Q crossover probability
q_start = 0.07
q_stop = 0.07
q_points = 1
  

####################################################################
#                   Generate the neural network  
####################################################################
# Define noise
noise_1= Lambda(d_f.addBACNoise, arguments={'p':train_p_1, 'q': train_q}, 
                      input_shape=(N,), output_shape=d_f.return_output_shape, name="noise1")
noise_2= Lambda(d_f.addBACNoise, arguments={'p':train_p_2, 'q': train_q}, 
                      input_shape=(N,), output_shape=d_f.return_output_shape, name="noise2")
# Normalzation layer 
norm_layer = BatchNormalization()

# Decoder 
inputs_dec = Input(shape=(N,))
layer_dec = Dense(design[0], activation='relu')(inputs_dec) 
predictions = Dense(2**k, activation='softmax')(layer_dec)
decoder = Model(inputs=inputs_dec, outputs=predictions) 

# Overall model 
inputs_overall = Input(shape=(N,))
x_noisy_1 = noise_1(inputs_overall)
x_noisy_2 = noise_2(inputs_overall)
x_dec_1 = decoder(x_noisy_1)
x_dec_2 = decoder(x_noisy_2) 

model = Model(inputs=inputs_overall, outputs=[x_dec_1 , x_dec_2 ], name = 'overall_model')
model.compile(optimizer=optimizer, loss='categorical_crossentropy',loss_weights=[1.0,0.08]) 
   

##################################################################
#                 Data generation and training 
#################################################################

# Create all possible information words
d = np.zeros((2**k,k),dtype=bool)
for i in range(1,2**k):
    d[i]= d_f.inc_bool(d[i-1])

# Create sets of all possible codewords (codebook) 
if our_polar==1: 
    if float(k)/N == 1.0/2: # Rate 1/2 polar code
        A = np.array([ False, False, False, True, False, True, False, True, 
               False, True,  False, True, False, True, True,  True ])
    if float(k)/N == 1.0/4: # Rate 1/4 polar code
        A = np.array([False, False, False, False, False, False, False,  True,
                  False, False, False, True, False, True, False, True,])
    if float(k)/N == 3.0/4:
        A = np.array([False,True, False, True, False, True, True, True,
         False, True, True, True, True, True, True, True,]) 
else: 
    A = d_f.polar_design_bsc(N,k)
    
if N==8:
    A = np.array([ False, False, False, True, False, True, True, True])    
# Logical vector indicating the nonfrozen bit locations 
x = np.zeros((2**k, N),dtype=bool)
u = np.zeros((2**k, N),dtype=bool)
u[:,A] = d

for i in range(0,2**k):
    x[i] = d_f.polar_transform_iter(u[i]) 

# One hot training vector
x_train = np.tile( x, (batch_size_norm ,1))
d_train = np.tile( np.eye(2**k) , (batch_size_norm,1))
  
# Training the neural net
history = model.fit(x_train, [d_train,d_train] , batch_size=batch_size, epochs=nb_epoch, verbose=2, shuffle=shuffle_var)
decoder.save_weights('decoder_lm_'+str(k)+'_'+str(N)+'_ours_q_'+str(100*train_q)+'.h5')  
 
#if plot_losses==1:
plt.figure()
plt.semilogy(history.history['loss'])
#    plt.plot(np.log(history.history['model_16_loss_2']))
#    plt.plot(np.log(history.history['model_16_loss_1']))
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['loss', 'loss_1', 'loss_2'], loc='upper left')
#    plt.show()
#    plt.yscale('log')
    
#####################################################################
#                            Testing 
##################################################################### 
p_vect = np.linspace(p_start, p_stop, p_points)
q_vect = np.linspace(q_start, q_stop, q_points)

# Choose different p and q or the same 
q_vect = p_vect 

p_vect_MAP =  p_vect
q_vect_MAP =  q_vect

# Initialization    
nb_errors_NN_f = np.zeros((len(p_vect), len(q_vect)),dtype=int)
nb_errors_NN_b = np.zeros((len(p_vect), len(q_vect)),dtype=int)

nb_errors_MAP_f = np.zeros((len(p_vect), len(q_vect)),dtype=int)
nb_errors_MAP_b = np.zeros((len(p_vect), len(q_vect)),dtype=int)

nb_bits_NN = np.zeros((len(p_vect), len(q_vect)),dtype=int)
nb_bits_MAP = np.zeros((len(p_vect), len(q_vect)),dtype=int)
nb_sequences_NN = np.zeros((len(p_vect), len(q_vect)),dtype=int)
nb_sequences_MAP = np.zeros((len(p_vect), len(q_vect)),dtype=int)
 
if test_ber == 1:
    for i_p in range(0,len(p_vect)):
        p = p_vect[i_p]
        
        for i_q in range(0, len(q_vect)):
            q = q_vect[i_q]
            
            # Initialize the errors counter
            N_errors = 0 
            N_iter = 0
            
            while N_iter < N_iter_max and N_errors < N_errors_mini:
                N_iter += 1
                
                # Source
                np.random.seed(0)
                d_test = np.random.randint(0,2,size=(num_words,k))
                # print('d_test',d_test)
                # d_test_onehot = np.zeros((num_words,2**k))
                # ind_test_dec = reduce(lambda a,b: 2*a+b, np.transpose(d_test))
                
                # Encoder 
                x_test = np.zeros((num_words, N),dtype=bool)
                u_test = np.zeros((num_words, N),dtype=bool)
                u_test[:,A] = d_test
            
                for iii in range(0,num_words):
                    x_test[iii] = d_f.polar_transform_iter(u_test[iii])
                    # d_test_onehot[iii,ind_test_dec[iii]]= 1
         
                # Channel (BA)
                w_soft_p = np.random.random(x_test.shape)
                np.random.seed(1)
                w_soft_q = np.random.random(x_test.shape)
                w_hard_p = (w_soft_p < p)*1
                w_hard_q = (w_soft_q < q)*1
                
                y_test = np.mod(x_test + x_test*w_hard_q + (1- x_test)*w_hard_p ,2)
                y_test = y_test.astype('float')
                # print('SHAPE',y_test.shape)
                # Decoder
                te = time.time()
                output_seq = np.argmax(decoder.predict(y_test),1)
                te = time.time() - te
                # print(f"A prediction time = {te}s ========================")

                # nb_errors_NN_f[i_p,i_q] += np.sum(1.0* (output_seq != ind_test_dec))
             
                # NN Decoder BER
                output_seq_b = 1*d_f.int2bin(output_seq,k)
                nb_errors_NN_b[i_p,i_q] += np.sum(1.0*(output_seq_b != d_test))
                nb_bits_NN[i_p,i_q] += k*num_words
                N_errors += np.sum(1.0*(output_seq_b != d_test)) 
        
                # MAP Decoder
                if MAP_activ == 1: 
                
                    for i_test in range(0, num_words):
                        
                        d_test_MAP = d_test[i_test,:]
                        y_test_MAP = y_test[i_test,:]
                        # ind_test_MAP = ind_test_dec[i_test]
                
                        log_APP = np.zeros((2**k,1),dtype= float)
                        for i_info in range(0, len(log_APP)):
                            I_0 = ( x[i_info,:]*1 < 10**-6)
                            I_1 = (x[i_info,:]*1 >  1 - 10**-6)
                            diff_count_0 = np.sum(np.mod(y_test_MAP[I_0]+ x[i_info,I_0]*1,2))  
                            diff_count_1 = np.sum(np.mod(y_test_MAP[I_1]+ x[i_info,I_1]*1,2))  
                            log_APP[i_info] = ( diff_count_0*np.log(np.clip(p/(1- p),10**-12,1)) +  
                                    diff_count_1*np.log(np.clip(q/(1- q),10**-12,1)) + np.sum(1*I_0) *np.log(1- p) + 
                                     np.sum(1*I_1) *np.log(1-q) ) 
                             
                        # Find the right codeword and information word
                        ind_inf = np.argmax(log_APP)  
                        output_seq_MAP = d[ind_inf,:]
                        
                        # Output sequence
                        # nb_errors_MAP_f[i_p,i_q] += 1.0*(ind_test_MAP != ind_inf)
                        nb_errors_MAP_b[i_p,i_q] += np.sum(np.mod(output_seq_MAP*1 + d_test_MAP ,2))
                    
            
            nb_sequences_NN[i_p,i_q] = num_words*N_iter
    
    nb_bits_MAP = nb_bits_NN
    nb_sequences_MAP = nb_sequences_NN    
    
    if plot_ber == 1: 
        legend = []
        plt.figure()
        p_ref = np.linspace(p_start,p_stop, 100)   
        plt.plot(p_vect_MAP, np.float32(nb_errors_MAP_b[:,i_q])/(nb_bits_MAP[:,i_q]) )
        legend.append('MAP')
        plt.plot(p_vect, np.float32(nb_errors_NN_b[:,i_q])/(nb_bits_NN[:,i_q]))
        legend.append('NN') 
        plt.legend(legend, loc=3)
        plt.yscale('log')
        plt.xlabel('$p$')
        plt.ylabel('BER')    
        plt.grid(True)
        plt.title('q = '+ str(q_vect[i_q] ))
        plt.show() 
    