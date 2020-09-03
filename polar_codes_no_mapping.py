#!/usr/bin/python
import sys
import numpy as np
import BACchannel as bac
import utils
import polar_codes_generator as polar

################################################################################python polar_codes_no_mapping.py 8 4 BSC
#### Uncomment these lines and comment the def polar_no_mapping(): to use directly (remember the return at the end)
channel_type = sys.argv[3]
k = int(sys.argv[2])
N = int(sys.argv[1])
cont = 5
if channel_type == 'AWGN':
  design_parameter = np.linspace(0.0, 10, cont)
else:
  design_parameter = np.linspace(0.1, 0.5, cont)
legends = []
FER = {}
for e_design in design_parameter:
  print('===============================', e_design)
  legends.append(f"p = {e_design:.2f}")

  G, infoBits = polar.polar_generator_matrix(N, k, channel_type, e_design)
  # print(G)
  k = len(G)
  N = len(G[1])
  U_k = bac.symbols_generator(k)  # all possible messages
  Y_n = bac.symbols_generator(N)  # all possible symbol sequences
  C = bac.matrix_codes(U_k, k, G, N)

  e0 = np.linspace(0.1, 0.9, 9)
  e1 = np.linspace(0.1, 0.5, 5)

  # print("0.00", '|', ["{:.4f}".format(ep1) for ep1 in e1])
  # print('------------------------------------------------------------------')
  error_probability = {}
  for ep0 in e0:
    row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1==ep0 or ep1 == 0.1:
      # if ep1 == 0.1:
        a = bac.succes_probability(Y_n, C, U_k, ep0, ep1)
        row.append(1 - a)
    error_probability[ep0] = row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in row])
  FER[e_design] = error_probability
utils.plot_BSC_BAC(f'FER Polar Codes N={N} k={k}',e0,FER, legends)
