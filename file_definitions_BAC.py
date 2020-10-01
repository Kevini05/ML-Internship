#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:32:34 2019

@author: meryem
"""
import numpy as np 
from keras.models import Sequential 
from keras import backend as K    
import tensorflow as tf 
from keras import layers

alpha_renyi = 2;          # Renyi divergence parameter 
eps_clip = 10**-40          # clipping for probabilities 
 
def BAC_marginal_pmf(Matrix_x, y, p_m, p,q):
    Num_words = Matrix_x.shape[0] 
    p_y_m = np.zeros((Num_words,1)) 
    for i in range( 0, Num_words):
        x = np.array([Matrix_x[i,:]])  
        p_y_m[i] =  BAC_conditional_pmf_v(x, y, p,q);
    
    return  np.dot(np.transpose(p_y_m) ,p_m)

def BAC_conditional_pmf_v(x_m, y, p,q):   
    x_int = x_m 
    I_0 = x_int < 10**-6  
    I_1 = x_int > 1 - 10**-6   
    
    P_Y_X_p = (p/(1-p) )** np.sum(np.mod(y[I_0].transpose()+ x_int[I_0]*1,2))
    P_Y_X_q = (q/(1-q) )** np.sum(np.mod(y[I_1].transpose()+ x_int[I_1]*1,2))
    p_y_m = (1-p)**(np.sum(1*I_0)) *(1-q)**(np.sum(1*I_1))  * P_Y_X_p* P_Y_X_q;
    return p_y_m

def BAC_conditional_pmf_m(X_m, y, p,q):  
    N_words = X_m.shape[1]
    p_y_m = np.zeros((N_words, 1),dtype = float ) 
    
    for i_m in range(0, N_words): 
        x_m = X_m[:,i_m]
        p_y_m[i_m]=  BAC_conditional_pmf_v(x_m, y, p,q)
    return p_y_m

def addBSCNoise(x, p): 
    w_soft = srng.uniform(K.shape(x))
#   w_soft = np.random.random(N)
    w_hard = (w_soft < p)*1
    
    return  np.mod(x + w_hard ,2) 
 

def addBACNoise(x, p,q): 
    w_soft_p = K.random_uniform(K.shape(x), 0, 1 )
    w_hard_p =  K.cast((w_soft_p < p) ,tf.float32) 
    
    w_soft_q = K.random_uniform(K.shape(x), 0, 1 )
    w_hard_q =  K.cast((w_soft_q< q) ,tf.float32)
    y = tf.cast(2, tf.float32)
    return  tf.math.floormod(x + x*w_hard_q + (1-x)*w_hard_p , y)
 

def addBACNoiseMI(x, p,q): 
    w_soft_p = K.random_uniform(K.shape(x), 0, 1 )
    w_hard_p =  K.cast((w_soft_p < p) ,tf.float32) 
    
    w_soft_q = K.random_uniform(K.shape(x), 0, 1 )
    w_hard_q =  K.cast((w_soft_q< q) ,tf.float32)  
    
    w = np.mod( x*w_hard_q + (1-x)*w_hard_p ,2)
    wx = np.mod( x + x*w_hard_q + (1-x)*w_hard_p ,2)
    return  [wx, w]
    
def addBACNoise_compound_q(x, p, q_1,q_2,batch_size_exp, nb_q):  
    w_soft_p = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_q_1 = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_q_2 = K.random_uniform(K.shape(x), 0, 1 )
    w_hard_p = K.cast((w_soft_p < p) ,tf.float32) 
    w_hard_q_1 = K.cast((w_soft_q_1 < q_1) ,tf.float32) 
    w_hard_q_2 = K.cast((w_soft_q_2 < q_2) ,tf.float32) 
     
    x_int_1 = np.mod(x + x*w_hard_q_1 + (1-x)*w_hard_p ,2)
    x_int_2 = np.mod(x + x*w_hard_q_2 + (1-x)*w_hard_p ,2)
    
    return K.concatenate([x_int_1[0:batch_size_exp,:], x_int_2[ batch_size_exp:nb_q*batch_size_exp,:]], axis =0 )

def addBACNoise_compound_p(x, p_1, p_2,q,batch_size_exp, nb_p):  
    w_soft_q = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_p_1 = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_p_2 = K.random_uniform(K.shape(x), 0, 1 )
    w_hard_q = K.cast((w_soft_q < q) ,tf.float32) 
    w_hard_p_1 = K.cast((w_soft_p_1 < p_1) ,tf.float32) 
    w_hard_p_2 = K.cast((w_soft_p_2 < p_2) ,tf.float32) 
     
    x_int_1 = np.mod(x + x*w_hard_q+ (1-x)*w_hard_p_1 ,2)
    x_int_2 = np.mod(x + x*w_hard_q + (1-x)*w_hard_p_2 ,2)
    
    return K.concatenate([x_int_1[0:batch_size_exp,:], x_int_2[ batch_size_exp:nb_p*batch_size_exp,:]], axis =0 )

def addBACNoise_compound_p123(x, p_1, p_2,p_3,q,batch_size_exp, nb_p):  
    w_soft_q = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_p_1 = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_p_2 = K.random_uniform(K.shape(x), 0, 1 ) 
    w_soft_p_3 = K.random_uniform(K.shape(x), 0, 1 )
    w_hard_q = K.cast((w_soft_q < q) ,tf.float32) 
    w_hard_p_1 = K.cast((w_soft_p_1 < p_1) ,tf.float32) 
    w_hard_p_2 = K.cast((w_soft_p_2 < p_2) ,tf.float32) 
    w_hard_p_3 = K.cast((w_soft_p_3 < p_3) ,tf.float32)  
    x_int_1 = np.mod(x + x*w_hard_q+ (1-x)*w_hard_p_1 ,2)
    x_int_2 = np.mod(x + x*w_hard_q + (1-x)*w_hard_p_2 ,2)
    x_int_3 = np.mod(x + x*w_hard_q + (1-x)*w_hard_p_3 ,2)
    
    return K.concatenate([x_int_1[0:batch_size_exp,:], x_int_2[ batch_size_exp:2*batch_size_exp,:], x_int_3[2*batch_size_exp:3*batch_size_exp,:]], axis =0 )


def addBACNoise_compound_p_ratio(x, p_1, p_2,q,batch_size_exp, nb_p, ratio):
    cursor = np.round(batch_size_exp*ratio*nb_p).astype(int) 
    w_soft_q = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_p_1 = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_p_2 = K.random_uniform(K.shape(x), 0, 1 )
    w_hard_q = K.cast((w_soft_q < q) ,tf.float32) 
    w_hard_p_1 = K.cast((w_soft_p_1 < p_1) ,tf.float32) 
    w_hard_p_2 = K.cast((w_soft_p_2 < p_2) ,tf.float32) 
     
    x_int_1 = np.mod(x + x*w_hard_q+ (1-x)*w_hard_p_1 ,2)
    x_int_2 = np.mod(x + x*w_hard_q + (1-x)*w_hard_p_2 ,2)
    
    return K.concatenate([x_int_1[0:cursor,:], x_int_2[ cursor:nb_p*batch_size_exp,:]], axis =0 )

def addBACNoise_compound_p_q(x, p_1, p_2,q_1,q_2,batch_size_exp, nb_p):  
    w_soft_q_1 = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_q_2 = K.random_uniform(K.shape(x), 0, 1 )
    
    w_soft_p_1 = K.random_uniform(K.shape(x), 0, 1 )
    w_soft_p_2 = K.random_uniform(K.shape(x), 0, 1 )
    
    w_hard_q_1 = K.cast((w_soft_q_1 < q_1) ,tf.float32) 
    w_hard_q_2 = K.cast((w_soft_q_2 < q_2) ,tf.float32)
    
    w_hard_p_1 = K.cast((w_soft_p_1 < p_1) ,tf.float32) 
    w_hard_p_2 = K.cast((w_soft_p_2 < p_2) ,tf.float32) 
     
    x_int_1 = np.mod(x + x*w_hard_q_1+ (1-x)*w_hard_p_1 ,2)
    x_int_2 = np.mod(x + x*w_hard_q_2 + (1-x)*w_hard_p_2 ,2)
    
    return K.concatenate([x_int_1[0:batch_size_exp,:], x_int_2[ batch_size_exp:nb_p*batch_size_exp,:]], axis =0 )

#def addNoise_compound(x,sigma_1,sigma_2,batch_size_exp, nb_snr): 
##    shape_x = list(K.shape(x))
##    assert len(shape_x) == 2    
#    w_1 = K.random_normal( K.shape(x) , mean=0.0, stddev=sigma_1)
#    w_2 = K.random_normal( K.shape(x) , mean=0.0, stddev=sigma_2)
#    x_int_1 = x + w_1 
#    x_int_2 = x + w_2 
#    return K.concatenate([x_int_1[0:batch_size_exp,:], x_int_2[ batch_size_exp:nb_snr*batch_size_exp,:]], axis =0 )
# 
def KL_divergence(y_true,y_pred):  
    return K.sum(y_true*(K.log(K.clip(y_true,eps_clip,10))-
                                      K.log(K.clip(y_pred,eps_clip,10))),1) 

def KL_divergence_loss(y_true,y_pred):  
    return K.mean(K.sum(y_true*(K.log(K.clip(y_true,eps_clip,10))-
                                      K.log(K.clip(y_pred,eps_clip,10))),1)) 

def Renyi_divergence(y_true,y_pred):
    div = 1/(alpha_renyi-1)*K.log(K.mean((K.sum(y_pred*y_true,1))**(-alpha_renyi) ,0))
    return div

def crossentropy_divergence(y_true,y_pred):  
    return  K.mean(K.categorical_crossentropy(y_true,y_pred),0) + 10*KL_divergence_marginal(y_true,y_pred)

def entropy_loss(y_true,y_pred):  
    return K.sum(y_pred*(K.log(K.clip(y_pred,eps_clip,10)),1))

def KL_divergence_marginal(y_true,y_pred):
    marginal_pred = K.mean(y_pred,0)
    marginal_true = K.mean(y_true,0)
    div_marg = K.sum(marginal_true*(K.log(K.clip(marginal_true,eps_clip,10))-
                                      K.log(K.clip(marginal_pred,eps_clip,10))))
    return div_marg 

def h_2(w):
    return -w*np.log2(w) -(1-w)*np.log2(1-w)
def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))

def fer(y_true, y_pred):
    return  K.not_equal(K.argmax(y_true), K.argmax(y_pred)) 

def return_output_shape(input_shape):  
    return input_shape
 
def return_output_shapeMI(N):  
    return (N,N)

def compose_model(layers):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    return model

def log_likelihood_ratio(x, sigma):
    return 2*x/np.float32(sigma**2)

def errors(y_true, y_pred):
#     K.sum(K.not_equal(y_true, K.round(y_pred))) # if theano used
    return K.sum(K.cast(K.not_equal(y_true, K.round(y_pred)),tf.int32)) # if tensorflow used

def half_adder(a,b):
    s = a ^ b
    c = a & b
    return s,c

def full_adder(a,b,c):
    s = (a ^ b) ^ c
    c = (a & b) | (c & (a ^ b))
    return s,c

def add_bool(a,b):
    if len(a) != len(b):
        raise ValueError('arrays with different length')
    k = len(a)
    s = np.zeros(k,dtype=bool)
    c = False
    for i in reversed(range(0,k)):
        s[i], c = full_adder(a[i],b[i],c)    
    if c:
        warnings.warn("Addition overflow!")
    return s

def inc_bool(a):
    k = len(a)
    increment = np.hstack((np.zeros(k-1,dtype=bool), np.ones(1,dtype=bool)))
    a = add_bool(a,increment)
    return a

def bitrevorder(x):
    m = np.amax(x)
    n = np.ceil(np.log2(m)).astype(int)
    for i in range(0,len(x)):
        x[i] = int('{:0{n}b}'.format(x[i],n=n)[::-1],2)  
    return x

def int2bin(x,N):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        binary = np.zeros((len(x),N),dtype='bool')
        for i in range(0,len(x)):
            binary[i] = np.array([int(j) for j in bin(x[i])[2:].zfill(N)])
    else:
        binary = np.array([int(j) for j in bin(x)[2:].zfill(N)],dtype=bool)
    
    return binary

def bin2int(b):
    if isinstance(b[0], list):
        integer = np.zeros((len(b),),dtype=int)
        for i in range(0,len(b)):
            out = 0
            for bit in b[i]:
                out = (out << 1) | bit
            integer[i] = out
    elif isinstance(b, np.ndarray):
        if len(b.shape) == 1:
            out = 0
            for bit in b:
                out = (out << 1) | bit
            integer = out     
        else:
            integer = np.zeros((b.shape[0],),dtype=int)
            for i in range(0,b.shape[0]):
                out = 0
                for bit in b[i]:
                    out = (out << 1) | bit
                integer[i] = out
        
    return integer

def polar_design_ours(N,k,build_p):  
         
    z0 = np.zeros(N) 
    z0[0] = np.log(2)+0.5*np.log(build_p)+0.5*np.log(1-build_p)
    
    for j in range(1,int(np.log2(N))+1):
        u = 2**j
        for t in range(0,int(u/2)):
            T = z0[t]
            z0[t] = 2*T - T**2     # upper channel
            z0[int(u/2)+t] = T**2  # lower channel
        
    # sort into increasing order
    idx = np.argsort(z0)
        
    # select k best channels
    idx = np.sort((idx[0:k]))
    
    A = np.zeros(N, dtype=bool)
    A[idx] = True
        
    return A

def polar_design_bsc(N,k,build_p):  
         
    z0 = np.zeros(N) 
    z0[0] = np.log(2)+0.5*np.log(build_p)+0.5*np.log(1-build_p)
    
    for j in range(1,int(np.log2(N))+1):
        u = 2**j
        for t in range(0,int(u/2)):
            T = z0[t]
            z0[t] = 2*T - T**2     # upper channel
            z0[int(u/2)+t] = T**2  # lower channel
        
    # sort into increasing order
    idx = np.argsort(z0)
        
    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))
    
    A = np.zeros(N, dtype=bool)
    A[idx] = True
        
    return A

def polar_transform_iter(u):

    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0,stages):
        i = 0
        while i < N:
            for j in range(0,n):
                idx = i+j
                x[idx] = x[idx] ^ x[idx+n]
            i=i+2*n
        n=2*n
    return x



 #class addBACNoiseMI_compound(Layer):
#    def __init__(self, **kwargs):
#        super(addBACNoiseMI_compound, self).__init__(**kwargs)
#
#    def build(self, input_shape):
#        super(addBACNoiseMI_compound, self).build(input_shape) 
#
#    def call(self, x):
#        p_1 = train_p_1
#        p_2 = train_p_2
#        p_3 = train_p_3
#        q = train_q 
#        
#        w_soft_q = K.random_uniform(K.shape(x), 0, 1 )
#        w_soft_p_1 = K.random_uniform(K.shape(x), 0, 1 )
#        w_soft_p_2 = K.random_uniform(K.shape(x), 0, 1 )
#        w_soft_p_3 = K.random_uniform(K.shape(x), 0, 1 )
#        
#        w_hard_q = K.cast((w_soft_q < q) ,tf.float32) 
#        w_hard_p_1 = K.cast((w_soft_p_1 < p_1) ,tf.float32) 
#        w_hard_p_2 = K.cast((w_soft_p_2 < p_2) ,tf.float32) 
#        w_hard_p_3 = K.cast((w_soft_p_3 < p_3) ,tf.float32) 
#        
#        x_int_1 = np.mod(x + x*w_hard_q+ (1-x)*w_hard_p_1 ,2)
#        x_int_2 = np.mod(x + x*w_hard_q + (1-x)*w_hard_p_2 ,2) 
#        x_int_3 = np.mod(x + x*w_hard_q + (1-x)*w_hard_p_3 ,2)
#        
#        p_vect_1 = tf.zeros(K.shape(x))
#        p_vect_2 = tf.ones(K.shape(x))
#        p_vect_3 = 2*tf.ones(K.shape(x)) 
#        
#        wx = K.concatenate([x_int_1[0:batch_size_exp,:], x_int_2[ batch_size_exp:nb_p*batch_size_exp,:],x_int_3[2*batch_size_exp:nb_p*batch_size_exp,:]], axis =0 )
#        p_vect = K.concatenate([p_vect_1[0:batch_size_exp,:], p_vect_2[batch_size_exp:nb_p*batch_size_exp,:] ,p_vect_3[2*batch_size_exp:nb_p*batch_size_exp,:]], axis =0 )
#        return  K.concatenate([wx,p_vect], axis= 1)
#    
#    def compute_output_shape(self, input_shape): 
#        return [input_shape, input_shape]  
 
 
##############################################################################################################################################
# Experimental stuff : test a function (tensorflow)
# sess = tf.InteractiveSession()
#def hard_decision(x):
#    n_r, n_c = x.get_shape().as_list()
#    x_max = tf.keras.backend.max(x, axis = 1, keepdims= True )
#    x_max_rep = K.repeat_elements(x_max, n_c, 1)
#    x_bool = ((x - x_max_rep) >= 0 )*1.0
#    return x_bool 
#
#n_ro = 100  
#n_co = 5
#x = tf.zeros((n_ro,n_co)) 
#w_soft_p = K.softmax(K.random_uniform(K.shape(x), 0, 1 ))
#w_soft_p.eval()


##summa = K.sum(w_soft_p,axis = 1)
##summa.eval()
#
#z = hard_decision(w_soft_p)
#z.eval()
#
#
## 
#  
