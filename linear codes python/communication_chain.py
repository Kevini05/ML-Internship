#!/usr/bin/python
import sys
import numpy as np
import BACchannel as bac
import polar_codes_generator as polar
import random as rd
import utils

cont = 5
if sys.argv[3] == 'AWGN':
  design_parameter = np.linspace(0.0,10,cont)
else:
  design_parameter = np.linspace(0.1, 0.5,cont)
legends = []
BER = {}

for e_design in design_parameter:
  print('===============================', e_design)
  legends.append(f"p = {e_design:.2f}")
  G,infoBits = polar.polar_generator_matrix(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], e_design)
  # print(G)
  k = len(G)
  N = len(G[1])
  U_k = bac.symbols_generator(k)  # all possible messages
  Y_n = bac.symbols_generator(N)  # all possible symbol sequences
  C = bac.matrix_codes(U_k, k, G, N)
  B = 1000

  e0 = np.linspace(0.1, 0.9, 9)
  e1 = np.linspace(0.1, 0.5, 5)

  bep ={}
  ber = {}
  for ep0 in e0:
    row = []
    ber_row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1==ep0 or ep1 == 0.1:
      # if ep1 == 0.1:
        # a = bac.succes_probability(Y_n, C, U_k, ep0, ep1)
        # row.append(1 - a)

        ber_tmp = 0 # for bit error rate
        ser_tmp = 0  #for symbol error rate
        for t in range(B):
          u = [rd.randint(0, 1) for i in range(k)]  # Bits à envoyer
          x = bac.FEC_encoder(u, G)  # bits encodés

          y_bac = bac.BAC_channel(x, ep0, ep1)  # symboles reçus
          ser_tmp+= bac.NbOfErrors(x, y_bac)

          u_map_bac = U_k[bac.MAP_BAC(y_bac, k, C, ep0, ep1)] # Detecteur MAP
          ber_tmp+= bac.NbOfErrors(u, u_map_bac)  # Calcul de bit error rate avec MAP

        ber_tmp = ber_tmp/(N*1.0*B)  # Calcul de bit error rate avec MAP
        ser_tmp = ser_tmp/(k*1.0*B)  # Calcul de symbol error rate avec MAP
        ber_row.append(ber_tmp)

    bep[ep0] = row
    ber[ep0] = ber_row

    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in row])
    print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
  BER[e_design] = ber

utils.plot_BSC_BAC(f'BER Polar Codes - Com chain N={N} k={k}',e0,BER, legends)
