#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import matplotlib.pyplot as plt
import itertools
import keras
from mish import Mish as mish
# import random as rd
from polarcodes import *
import time

def plot_BSC_BAC(title, error_probability,R):
  """
  :param title: Figure title
  :param e0: linspace of all epsilon0
  :param error_probability: error probability dictionary (BER or BLER)
  :param R: Coding rate R=k/N
  :return: plot
  """"""
  """

  fig = plt.figure(figsize=(7, 3.5), dpi=180, facecolor='w', edgecolor='k')
  fig.subplots_adjust(wspace=0.4, top=0.8)
  fig.suptitle(title, fontsize=14)
  ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
  ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
  marker = itertools.cycle(('h', 'p', '*', '.', '+', 'o', 'h', 's', ','))
  linestyle = itertools.cycle(('-', '--', '-.', ':'))
  legends = []
  for keys in error_probability:
    bac_fer = []
    bsc_fer = []
    legends.append(keys)
    # print(keys,error_probability,e0)
    e0_bac = []
    for ep0 in error_probability[keys]:
      e0_bac.append(ep0)
    for ep0 in e0_bac:
      bac_fer.append(error_probability[keys][ep0][0])
      if ep0 <= 0.5:
        bsc_fer.append(error_probability[keys][ep0][-1])
    # print(keys)
    # print('BAC', ["{:.4f}".format(a) for a in bac_fer])
    # print('BSC', ["{:.4f}".format(a) for a in bsc_fer])

    e0_bsc = [x for x in e0_bac if x <= 0.5]
    m = next(marker)
    # l = next(linestyle)
    l='-'
    ax1.semilogy(e0_bac, [bac_fer[a] for a in range(len(bac_fer))], linestyle=l, marker=m, ms=0.5, linewidth=0.5)
    ax2.semilogy(e0_bsc, [bsc_fer[a] for a in range(len(bsc_fer))], linestyle=l, marker=m, ms=0.5, linewidth=0.5)

  E0 = np.linspace(0.0001, 0.99999, 901)
  ax1.semilogy(E0,cut_off_epsilon(E0, e0_bac[0], R,'BAC'),'k', linestyle='-', ms=0.1, linewidth=0.15)
  E0 = np.linspace(0.0001, 0.49999, 451)
  ax2.semilogy(E0, cut_off_epsilon(E0, 0, R, 'BSC'), 'k', linestyle='-', ms=0.1, linewidth=0.15)

  ax1.legend(legends,prop={'size': 5},loc="lower right")
  ax1.set_title(f"BAC($\epsilon_1$={e0_bac[0]},$\epsilon_0$)", fontsize=8)
  ax1.set_xlabel('$\epsilon_0$', fontsize=8)
  ax1.set_ylabel('Error Probability', fontsize=8)
  # ax1.set_xticklabels(np.arange(0, 1, step=0.2))
  ax1.grid(which='both', linewidth=0.2)

  ax2.legend(legends,prop={'size': 5},loc="lower right")
  ax2.set_title('BSC($\epsilon$)', fontsize=8)
  ax2.set_xlabel('$\epsilon$', fontsize=8)
  ax2.grid(which='both', linewidth=0.2)


def h2(x):
  return -(1-x)*np.log2(1-x)-x*np.log2(x)

def cut_off_epsilon(E0,e1,R,channel):
  c = []
  if channel == 'BAC':
    for e0 in E0:
      z = 2**((h2(e0)-h2(e1))/(1-e0-e1))
      c.append(np.log2(z+1) - (1-e1)*h2(e0)/(1-e0-e1) + e0*h2(e1)/(1-e0-e1))
  elif channel == 'BSC':
    for e0 in E0:
      c.append(h2(0.5)-h2(e0))
  index = np.argmin(np.abs(np.array(c) - R))
  cut_off = []
  for i in range(len(E0)):
    cut_off.append(0) if i < index else cut_off.append(0.5)
  return cut_off

def NN_encoder(k,N):
  print('*******************codebook********************************************')
  one_hot = np.eye(2 ** k)
  model_encoder = keras.models.load_model("autoencoder/model_encoder.h5")
  print("Encoder Loaded from disk, ready to be used")

  codebook = np.round(model_encoder.predict(one_hot)).astype('int')
  # print(codebook)

  return codebook

def block_error_probability(N, k, C, e0, e1):
  """ :param N: coded message size
      :param k: message size
      :param C: Codebook
      :return: error probability for all combinations of e0 and e1"""
  U_k = symbols_generator(k)  # all possible messages
  Y_n = symbols_generator(N)  # all possible symbol sequences

  # print("0.00", '|', ["{:.4f}".format(ep1) for ep1 in e1])
  # print('------------------------------------------------------------------')
  error_probability = {}
  for ep0 in e0:
    row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        a = succes_probability(Y_n, C, U_k, ep0, ep1)
        row.append(1 - a)
    error_probability[ep0] = row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in row])
  return error_probability

def bit_error_rate(k, C, N_iter_max, e0, e1, coded = True):
  N_errors_mini = 100
  U_k = symbols_generator(k)  # all possible messages
  ber = {}
  count = 0
  for ep0 in e0:
    ber_row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]:
        ber_tmp = 0  # for bit error rate

        N_errors = 0
        N_iter = 0
        while N_iter < N_iter_max:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits to be sent
          x = C[idx]  # coded bits
          y_bac = BAC_channel(x, ep0, ep1)  # received symbols

          te = time.time()
          u_map_bac = U_k[MAP_BAC(y_bac, k, C, ep0, ep1)] if coded else MAP_BAC_uncoded(y_bac, ep0, ep1)  # MAP Detector
          te = time.time() - te
          # print(f"A MAP time = {te}s ========================")

          N_errors += NbOfErrors(u, u_map_bac)  # bit error rate compute with MAPs
        ber_tmp = N_errors / (k * 1.0 * N_iter)  # bit error rate compute with MAP
        ber_row.append(ber_tmp)

    ber[ep0] = ber_row
    count += 1
    print("{:.3f}".format(count / len(e0) * 100), '% completed ')
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
  return ber

def bit_error_rate_NN(N, k, C, N_iter_max, e0, e1, channel = 'BSC' ):
  print('*******************NN-Decoder********************************************')
  model_decoder = keras.models.load_model("autoencoder/model_decoder.h5")
  # model_decoder = keras.models.load_model("./model/model_decoder_16_4_std.h5")
  print("Decoder Loaded from disk, ready to be used")
  U_k = symbols_generator(k)  # all possible messages
  ber = {}
  count = 0
  for ep0 in e0:
    ber_row = []
    interval = np.zeros(4)
    interval[int(ep0*4) if ep0 < 0.25 else 3] = 1.0
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        N_errors = 0
        N_iter = 0
        while N_iter < N_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits to be sent
          x = C[idx]  # coded bits
          y_bac = BAC_channel(x, ep0, ep1)  # received symbols

          yh = np.reshape(np.concatenate((y_bac,interval),axis=0), [1, N+4]) if channel == 'BAC'  else np.reshape(y_bac, [1, N]).astype(np.float64)
          u_nn = U_k[np.argmax(model_decoder(yh))]  #  NN Detector

          N_errors += NbOfErrors(u, u_nn)  # bit error rate compute with NN
        ber_tmp = N_errors / (k * 1.0 * N_iter)  # bit error rate compute with NN
        ber_row.append(ber_tmp)

    ber[ep0] = ber_row
    print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber

def bit_error_rate_NN_predict(N, k, C, Nb_sequences, e0, e1, inter=False):
  print('*******************NN-Decoder********************************************')
  model_decoder = keras.models.load_model("autoencoder/model_decoder.h5")
  # model_decoder = keras.models.load_model("./model/model_decoder_16_4_std.h5")
  print("Decoder Loaded from disk, ready to be used")
  U_k = symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(Nb_sequences/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    interval = np.zeros(4)
    interval[int(ep0*4) if ep0 < 0.25 else 3] = 1.0
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
          if inter:
            y_bac = [np.concatenate((BAC_channel(xi, ep0, ep1), interval), axis=0) for xi in x]  # received symbols
            dec_input_size = N+4
          else:
            y_bac = [BAC_channel(xi, ep0, ep1)  for xi in x]# received symbols
            dec_input_size = N

          yh = np.reshape(y_bac, [Nb_words, dec_input_size]).astype(np.float64)
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

def mapping(C, X, t, nx):
  codes = []
  count = 0
  if len(C[1]) % t == 0:
    for c in C:
      # print(c)
      row = []
      for i in range(0, len(c), t):
        idx = X.index(c[i:i + t])
        # print(idx)
        row.append(1) if idx < nx else row.append(0)
      count += sum(row)
      codes.append(row)
    print(f"dist = {count * 1.00 / (len(row) * len(codes)):.3f} after mapping")
    aux = []
    a = 0
    for code in codes:
      if code in aux:
        a+=1
    print('++++++++++++++++++Repeated Codes = ', a)
    return codes
  else:
    raise IOError('ERROR t is not multiple of big N')

def mapping2(C, X, t, nx):
  codes = []
  count = 0
  idx_list = list(range(len(C[1])))
  np.random.shuffle(idx_list)
  # idx_list = [27, 25, 7, 34, 40, 43, 50, 9, 6, 30, 24, 39, 4, 49, 1, 17, 10, 5, 58, 12, 23, 33, 36, 20, 2, 29, 15, 48, 3, 60, 11, 53, 59, 51, 8, 47, 37, 54, 61, 56, 35, 14, 0, 38, 21, 22, 44, 46, 31, 55, 13, 32, 26, 57, 62, 28, 18, 63, 19, 42, 45, 52, 16, 41]
  print(idx_list)
  if len(C[1]) % t == 0:
    for c in C:
      row = []
      # print(c)
      for i in range(0,int(len(C[1])),t):
        aux = [c[a] for a in idx_list[i:i+t]]
        # print(aux)
        idx = X.index(aux)
        # print(idx)
        row.append(1) if idx <= nx else row.append(0)
      count += sum(row)
      codes.append(row)
    print(f"dist = {count * 1.00 / (len(row) * len(codes)):.3f} after mapping")
    aux = []
    for code in codes:
      if code in aux:
        # print('****repeated code******')
        a=1
      else:
        aux.append(code)
    print('+++++++++++++++++++Repeated Codes = ',len(C)-len(aux))
    return codes
  else:
    raise IOError('ERROR t is not multiple of big N')

def mapping3(C, X, t, nx):
  codes = []
  count = 0
  if len(C[1]) % t == 0:
    for c in C:
      # print(c)
      row = []
      for i in range(0, len(c), t):
        # print(c[i:i + t])
        s =sum(c[i:i + t])
        # print(s)
        row.append(0) if s <= t*nx else row.append(1)
      count += sum(row)
      codes.append(row)
    print(f"dist = {count * 1.00 / (len(row) * len(codes)):.3f} after mapping")
    aux = []
    for code in codes:
      if code in aux:
        # print('****repeated code******')
        a=1
      else:
        aux.append(code)
    print('+++++++++++++++++++++Repeated Codes = ', len(C) - len(aux))
    return codes
  else:
    raise IOError('ERROR t is not multiple of big N')

def integrated_function(infoBits, msm, k, N, threshold):
  T = np.transpose(arikan_gen(int(np.log2(N))))
  V = []
  for i in range(len(msm)):
    row = []
    count = 0
    count_frozen = 0
    frozen = [(1 if np.random.randint(0, 100) > threshold else 0) for x in range(N - k)]
    # print(frozen)
    for a in range(N):
      if a in infoBits:
        row.append(msm[i][count])
        count += 1
      else:
        row.append(frozen[count_frozen])
        # row.append(1)
        count_frozen += 1
    V.append(row)
  codebook = matrix_codes2(V, k, T, N)
  # Validation of codewords
  aux = []
  for code in codebook:
    if code in aux:
      print('****repeated codeword Integrated Scheme******')
    else:
      aux.append(code)
  return codebook

################################## BAC Functions #####################################################

def linspace(a, b, n=100):
  """ :param a: start point
      :param b: stop point
      :param n: number of points
      :return: linspace """
  if n < 2:
      return b
  diff = (float(b) - a)/(n - 1)
  return [diff * i + a  for i in range(n)]

def MAP_BAC(symbols,k,codes,e0,e1):
  """ :param symbols: Received Symbols
      :param k: message size
      :param codes: codebook (all codewords of N-length)
      :param e0 et e1: Crossover probabilities
      :return: index of decoded message among every possible messages """
  g = [0 for i in range(2**k)]
  for j in range(2**k):
    d11 = 0
    d01 = 0
    d10 = 0
    for i in range(len(symbols)):
      d11 += int(codes[j][i]) & int(symbols[i])
      d01 += ~int(codes[j][i]) & int(symbols[i])
      d10 += int(codes[j][i]) & ~int(symbols[i])
    g[j] = (e0/(1-e0))**d01*(e1/(1-e0))**d10*((1-e1)/(1-e0))**d11
  return g.index(max(g))

def MAP_BAC_uncoded(code,e0,e1):
  """ :param codes: codebook (all codewords of N-length)
      :param e0 et e1: Crossover probabilities
      :return: index of decoded message among every possible messages """
  if e1+e0==1.0 or e1==0.0 or e0==0:
    y = 0.5
  else:
    y = np.log(e1 / (1 - e0)) / (np.log((e1 * e0) / ((1 - e0) * (1 - e1))))
  decoded_message = []
  for u in code:
    decoded_message.append(1) if u > y else decoded_message.append(0)
  return decoded_message

def symbols_generator(N):
  """ :param N: symbols size (number of bits)
      :return: all possible bit combinations of length N """
  messages = []
  for i in range(2**N):
     messages.append([0 for a in range(N)])
     nb = bin(i)[2:].zfill(N)
     for j in range(N):
        messages[i][j] = int(nb[j])
  return messages

def succes_probability(symbols,codes,msm,e0,e1):
  """ :param symbols: recei
      :param
      :param e0 et e1: Crossover probabilities
      :return: succes probabilities  """
  Pc = 0
  for y in symbols:
    # print('y',y,'g(y)')
    id = MAP_BAC(y,len(msm[1]),codes,e0,e1)
    u = msm[id]
    d11 = 0
    d01 = 0
    d10 = 0
    for i in range(len(y)):
      d11 += int(codes[id][i]) & int(y[i])
      d01 += ~int(codes[id][i]) & int(y[i])
      d10 += int(codes[id][i]) & ~int(y[i])

    Pc += (e0/(1-e0))**d01*(e1/(1-e0))**d10*((1-e1)/(1-e0))**d11
    # print('u',u,'f(u)',codes[id])
  return (1-e0)**len(y)/(2**len(u))*Pc

def matrix_codes(msm, k, G, N):
  codes = []
  g = []
  for i in range(N):
    g.append([G[j][i] for j in range(k)])
  # print('G',G,'g',g)
  for a in range(2**k):
    row = [sum([i * j for (i, j) in zip(g[b], msm[a])])%2 for b in range(N)]
    codes.append(row)
  print('dist = ', sum([sum(codes[h]) for h in range(len(codes))])*1.0/(N*2**k))
  return codes

def matrix_codes2(msm, k, G, N):
  codes = []
  g = []
  for i in range(N):
    g.append([G[j][i] for j in range(N)])
  # print('G',G,'g',g)
  for a in range(2**k):
    row = [sum([i * j for (i, j) in zip(g[b], msm[a])])%2 for b in range(N)]
    codes.append(row)
  print('dist = ', sum([sum(codes[h]) for h in range(len(codes))])*1.000/(N*2**k))
  return codes

def optimal_distribution(e0,e1):
	if e0+e1<1:
		he0	= -e0*np.math.log(e0,2)-(1-e0)*np.math.log(1-e0,2)
		he1 = -e1*np.math.log(e1,2)-(1-e1)*np.math.log(1-e1,2)
		z = 2.0**((he0-he1)/(1.0-e0-e1))
		q = (z-e0*(1+z))/((1+z)*(1-e0-e1))
	else:
		q = 0.5
	return q

def FEC_encoder(b, G):
  """ :param b: bit sequence and Generator Matrix
      :return: sequence générée grâce à la matrice génératrice """
  return np.dot(np.transpose(G), np.transpose(b)) % 2

def BAC_channel(x, epsilon0, epsilon1):
  """ input : Symbols to be sent
      :return: Symboles reçus, bruités """
  # print('e0 ',epsilon0)
  # print('e1 ',epsilon1)
  x = np.array(x)
  n0 = np.array([int(b0<epsilon0) for b0 in np.random.uniform(0.0, 1.0, len(x))])
  n1 = np.array([int(b1<epsilon1) for b1 in np.random.uniform(0.0, 1.0, len(x))])
  n = n0*(x+1)+n1*x
  return np.mod(n+x,2) # Signal transmis + Bruit

def NbOfErrors(a, b):
  """ :param a,b: 2 bit's arrays
      :return: number of bits that are not equals (maximal distance 1e-2)"""
  # print('sent',a,'rec',b,'dif',np.sum(1.0*(a != b)))
  return np.sum(np.abs(np.array(a) - np.array(b)))
  # return np.sum(1.0*(a != b))  para el BLER