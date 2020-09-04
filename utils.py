#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import itertools
import BACchannel as bac
import random as rd
from polarcodes import *
import time



def plot_BSC_BAC(title, e0, error_probability,R):
  """
  :param title: Figure title
  :param e0: linspace of all epsilon0
  :param error_probability: error probability dictionary
  :param design_parameter: linspace with all design parameters
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

    for ep0 in e0:
      bac_fer.append(error_probability[keys][ep0][0])
      if ep0 <= 0.5:
        bsc_fer.append(error_probability[keys][ep0][-1])
    # print(keys)
    # print('BAC', ["{:.4f}".format(a) for a in bac_fer])
    # print('BSC', ["{:.4f}".format(a) for a in bsc_fer])

    ep0 = [x for x in e0 if x <= 0.5]
    m = next(marker)
    l = next(linestyle)
    ax1.semilogy(e0, [bac_fer[a] for a in range(len(bac_fer))], linestyle=l, marker=m, ms=1, linewidth=0.7)
    ax2.semilogy(ep0, [bsc_fer[a] for a in range(len(bsc_fer))], linestyle=l, marker=m, ms=1, linewidth=1)

  E0 = np.linspace(0.0001, 0.9999, 901)
  ax1.semilogy(E0,cut_off_epsilon(E0, e0[0], R,'BAC'),'k', linestyle='-', ms=0.1, linewidth=0.8)
  E0 = np.linspace(0.0001, 0.24999, 451)
  ax2.semilogy(E0, cut_off_epsilon(E0, e0[0], R, 'BSC'), 'k', linestyle='-', ms=0.1, linewidth=0.8)

  ax1.legend(legends,prop={'size': 5})
  ax1.set_title(f"BAC($\epsilon_1$={e0[0]},$\epsilon_0$)", fontsize=8)
  ax1.set_xlabel('$\epsilon_0$', fontsize=8)
  ax1.set_ylabel('Probabilité d`érreur', fontsize=8)
  ax1.grid()

  ax2.legend(legends,prop={'size': 5})
  ax2.set_title('BSC($\epsilon$)', fontsize=8)
  ax2.set_xlabel('$\epsilon$', fontsize=8)
  ax2.grid()

  plt.show()

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
    cut_off.append(0) if i < index else cut_off.append(1)
  return cut_off





def block_error_probability(N, k, C, e0, e1):
  """
  :param N: coded message size
  :param k: message size
  :param C: Codebook
  :return: error probability for all combinations of e0 and e1
  """
  U_k = bac.symbols_generator(k)  # all possible messages
  Y_n = bac.symbols_generator(N)  # all possible symbol sequences

  # e0 = np.linspace(0.1, 0.9, 9)
  # e1 = np.linspace(0.1, 0.5, 5)

  # print("0.00", '|', ["{:.4f}".format(ep1) for ep1 in e1])
  # print('------------------------------------------------------------------')
  error_probability = {}
  for ep0 in e0:
    row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        # if ep1 == 0.1:
        a = bac.succes_probability(Y_n, C, U_k, ep0, ep1)
        row.append(1 - a)
    error_probability[ep0] = row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in row])
  return error_probability

def block_error_rate(N, k, C, e0, e1):
  """
  :param N: coded message size
  :param k: message size
  :param C: Codebook
  :return: error probability for all combinations of e0 and e1
  """
  U_k = bac.symbols_generator(k)  # all possible messages
  Y_n = bac.symbols_generator(N)  # all possible symbol sequences

  # e0 = np.linspace(0.1, 0.9, 9)
  # e1 = np.linspace(0.1, 0.5, 5)

  # print("0.00", '|', ["{:.4f}".format(ep1) for ep1 in e1])
  # print('------------------------------------------------------------------')
  error_probability = {}
  for ep0 in e0:
    row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        # if ep1 == 0.1:
        a = bac.succes_probability(Y_n, C, U_k, ep0, ep1)
        row.append(1 - a)
    error_probability[ep0] = row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in row])
  return error_probability


def bit_error_rate(k, C, B, e0, e1, coded = True):
  # print(np.array(C))
  U_k = bac.symbols_generator(k)  # all possible messages
  # e0 = np.linspace(0.1, 0.9, 9)
  # e1 = np.linspace(0.1, 0.5, 5)
  print(B)
  ber = {}
  for ep0 in e0:
    ber_row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        # if ep1 == 0.1:
        ber_tmp = 0  # for bit error rate
        # ser_tmp = 0  # for symbol error rate
        for t in range(B):
          idx = rd.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits à envoyer
          x = C[idx]  # bits encodés

          y_bac = bac.BAC_channel(x, ep0, ep1)  # symboles reçus
          # ser_tmp += bac.NbOfErrors(x, y_bac)
          start = time.time()
          u_map_bac = U_k[bac.MAP_BAC(y_bac, k, C, ep0, ep1) ] if coded else bac.MAP_BAC_uncoded(y_bac, ep0, ep1) # Detecteur MAP
          end = time.time()
          print(end - start)
          ber_tmp += bac.NbOfErrors(u, u_map_bac)  # Calcul de bit error rate avec MAP
          # print(u,u_map_bac,ber_tmp)
        ber_tmp = ber_tmp / (k * 1.0 * B)  # Calcul de bit error rate avec MAP
        # ser_tmp = ser_tmp / (N * 1.0 * B)  # Calcul de symbol error rate avec MAP
        ber_row.append(ber_tmp)

    ber[ep0] = ber_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
  return ber


def tensorflow_shutup():
  """
  Make Tensorflow less verbose
  """
  try:
    # noinspection PyPackageRequirements
    import os
    from tensorflow import logging
    logging.set_verbosity(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
    # noinspection PyUnusedLocal
    def deprecated(date, instructions, warn_once=True):
      def deprecated_wrapper(func):
        return func

      return deprecated_wrapper

    from tensorflow.python.util import deprecation
    deprecation.deprecated = deprecated

  except ImportError:
    pass


def NN_decoder(y, codes, model,N):
  yh = np.reshape(np.array(y), [1, N])
  vector_prob = model.predict(yh)
  id = np.argmax(vector_prob)
  return codes[id]

def bit_error_rate_NN(N, k, C, B, e0, e1):
  print(B)
  U_k = bac.symbols_generator(k)  # all possible messages

  from keras.models import model_from_json
  # load json and create model
  tensorflow_shutup()
  json_file = open('./model_decoder_bsc.json', 'r')
  model_decoder_json = json_file.read()
  json_file.close()
  model_decoder = model_from_json(model_decoder_json)
  # load weights into new model
  model_decoder.load_weights("./weights_decoder_bsc.h5")
  print("Loaded model from disk")

  ber = {}
  for ep0 in e0:
    ber_row = []

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        # if ep1 == 0.1:
        # print('e0',ep0,'e1',ep1)
        ber_tmp = 0  # for bit error rate
        # ser_tmp = 0  # for symbol error rate
        for t in range(B):
          idx = rd.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits à envoyer
          x = C[idx]  # bits encodés

          y_bac = bac.BAC_channel(x, ep0, ep1)  # symboles reçus
          # ser_tmp += bac.NbOfErrors(x, y_bac)
          start = time.time()
          u_map_bac = NN_decoder(y_bac, U_k, model_decoder,N) # Detecteur MAP
          end = time.time()
          print(end - start)
          # print('yn', y_bac,'ukt',u_map_bac)
          ber_tmp += bac.NbOfErrors(u, u_map_bac)  # Calcul de bit error rate avec MAP
          # print(u,u_map_bac,ber_tmp)
        ber_tmp = ber_tmp / (k * 1.0 * B)  # Calcul de bit error rate avec MAP
        # ser_tmp = ser_tmp / (N * 1.0 * B)  # Calcul de symbol error rate avec MAP
        ber_row.append(ber_tmp)

    ber[ep0] = ber_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
  return ber


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

def mapping2(C, X, t, nx):
  codes = []
  count = 0
  idx_list = list(range(len(C[1])))
  rd.shuffle(idx_list)
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


def integrated_function(infoBits, msm, k, N, threshold):
  # print(infoBits)
  # threshold = 20
  T = np.transpose(arikan_gen(int(np.log2(N))))
  V = []
  for i in range(len(msm)):
    row = []
    count = 0
    count_frozen = 0
    frozen = [(1 if rd.randint(0, 100) > threshold else 0) for x in range(N - k)]
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

  codebook = bac.matrix_codes2(V, k, T, N)
  # print(np.array(codebook))

  # Validation of codewords
  aux = []
  for code in codebook:
    if code in aux:
      print('****repeated codeword Integrated Scheme******')
    else:
      aux.append(code)

  return codebook
