# #!/usr/bin/python
# import sys
# import numpy as np
# import BACchannel as bac
# import polar_codes_generator as polar
# import utils
# import matrix_codes as mat_gen
# import keras
#
# def NN_encoder(k,N):
#   codebook = []
#   one_hot = np.eye(2 ** k)
#   if N == 8 and k == 4:
#     model_encoder = keras.models.load_model("autoencoder/model_encoder.h5")
#   elif N == 16 and k == 8:
#     model_encoder = keras.models.load_model("autoencoder/model_encoder_BSC_rep-1000_epsilon-0.07_layerSize_4_epoch-100_k_8_N-16.h5")
#   elif N == 16 and k == 4:
#     model_encoder = keras.models.load_model("autoencoder/model_encoder.h5")
#     # model_encoder = keras.models.load_model("autoencoder/model_encoder_BAC_rep-100_epsilon-0.07_layerSize_5_epoch-300_k_4_N-16.h5")
#   print("Encoder Loaded model from disk, ready to be used")
#
#   for X in one_hot:
#     X = np.reshape(X, [1, 2**k])
#     c = [int(round(x)) for x in model_encoder.predict(X)[0]]
#     codebook.append(c)
#   aux = []
#   for code in codebook:
#     if code not in aux:
#       aux.append(code)
#   print('+++++++++++++++++++Repeated Codes = ', len(codebook) - len(aux))
#   return codebook
#
# def bit_error_rate_NN(N, k, C, B, e0, e1, channel = 'decoder' ):
#   # load weights into new model
#   if channel == 'decoder':
#     if N == 8 and k == 4:
#       model_decoder = keras.models.load_model("model/model_decoder_BAC_rep-1000_epsilon-0.07_layerSize_5_epoch-1000_k_4_N-8.h5")
#     elif N == 16 and k == 8:
#       model_decoder = keras.models.load_model("model/model_decoder_BAC_rep-500_epsilon-0.07_layerSize_5_epoch-100_k_8_N-16.h5")
#   elif channel == 'autoencoder':
#     print("autoencoder-decoder")
#     if N == 8 and k == 4:
#       model_decoder = keras.models.load_model("autoencoder/model_decoder_16_4_std.h5")
#     elif N == 16 and k == 4:
#       # model_decoder = keras.models.load_model("autoencoder/model_decoder_BAC_rep-100_epsilon-0.07_layerSize_5_epoch-300_k_4_N-16.h5")
#       model_decoder = keras.models.load_model("autoencoder/model_decoder_16_4_std.h5")
#     elif N == 16 and k == 8:
#       model_decoder = keras.models.load_model("autoencoder/model_decoder_16_4_std.h5")
#   print("Decoder Loaded model from disk, ready to be used")
#
#   U_k = bac.symbols_generator(k)  # all possible messages
#   ber = {}
#   count = 0
#   for ep0 in e0:
#     ber_row = []
#     for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
#       if ep1 == ep0 or ep1 == e0[0]:
#         ber_tmp = 0  # for bit error rate
#         interval = np.zeros(4)
#         interval[int(ep1*4)] = 1.0
#         for t in range(B):
#           idx = np.random.randint(0, len(U_k) - 1)
#           u = U_k[idx]  # Bits à envoyer
#           x = C[idx]  # bits encodés
#
#           y_bac = bac.BAC_channel(x, ep0, ep1)  # symboles reçus
#
#           start = time.time()
#           yh = np.reshape(np.concatenate((y_bac,interval),axis=0), [1, N+4]) if channel == 'autoencoder'  else np.reshape(y_bac, [1, N])
#           u_nn = U_k[np.argmax(model_decoder.predict(yh))]  # Detecteur NN
#           end = time.time()
#           # print('NN', end - start)
#
#           ber_tmp += bac.NbOfErrors(u, u_nn)  # Calcul de bit error rate avec NN
#         ber_tmp = ber_tmp / (k * 1.0 * B)  # Calcul de bit error rate avec NN
#         ber_row.append(ber_tmp)
#
#     ber[ep0] = ber_row
#     count+= 1
#     print(count/len(e0)*100,'% completed ')
#   return ber
#
# ################################################################################
#
# def polar_codes_NN(N=8, k=4, nb_pkts = 100, graph = 'BER',channel='BSC'):
#   print('-------------------Polar Code + NN decoder -----------------------------')
#   key = 'NN_dec'
#   G,infoBits = polar.polar_generator_matrix(N, k, channel, 0.1)
#   # print('G = ', np.array(G))
#   k = len(G)
#   N = len(G[1])
#   U_k = bac.symbols_generator(k)  # all possible messages
#   C = bac.matrix_codes(U_k, k, G, N)
#   print('k ', k, 'N ', N)
#   if graph == 'BLER':
#     return utils.block_error_probability(N, k, C, e0, e1)
#   else:
#     return bit_error_rate_NN(N, k, C, nb_pkts, e0, e1, 'decoder')
#     # print('key',key,BER[key])
#
# ### ============ Linear codes======================================
# def linear_codes_NN(N=8, k=4, nb_pkts = 100, graph = 'BER'):
#   print('-------------------Linear Code + NN-----------------------------',graph,N,k)
#   for key in ['BLKC_NN']:
#     # print(key)
#     G = mat_gen.matrix_codes(N, k, 'BKLC')
#     print(G)
#     if G != []:
#       # print('G = ', np.array(G))
#       k = len(G)
#       N = len(G[1])
#       U_k = bac.symbols_generator(k)  # all possible messages
#       C = bac.matrix_codes(U_k, k, G, N)
#       # print(np.array(C))
#       print('k ',k,'N ',N)
#       if graph == 'BLER':
#         return utils.block_error_probability(N,k,C,e0,e1)
#       else:
#         return bit_error_rate_NN(N, k, C, nb_pkts, e0, e1, 'decoder')
#         # print('key',key,BER[key])
#
# ### ============ NN-autoencoder ======================================
# def autoencoder_NN(N=8, k=4, nb_pkts = 100, graph = 'BER',channel='BSC'):
#   print('------------------- NN-autoencoder -----------------------------')
#   key = 'NN_auto'
#   C = NN_encoder(k,N)
#   # print(np.array(C))
#   print('k ', k, 'N ', N)
#   if graph == 'BLER':
#     return utils.block_error_probability(N, k, C, e0, e1)
#   else:
#     return bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,'autoencoder')
#     # print('key',key,BER[key])
# ### ============ Figures ======================================
#
# def saved_results(N=8, k=4, graph = 'BER'):
#   ber = {}
#   bler = {}
#   if graph == 'BER':
#     if N == 16:
#       if k == 4:
#         ber = {'BKLC': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [0.0, 0.0], 0.015848931924611134: [0.0, 0.0], 0.03162277660168379: [0.0, 0.000125], 0.0630957344480193: [0.000175, 0.00275], 0.12589254117941676: [0.0012, 0.03105], 0.25118864315095796: [0.009825, 0.20215], 0.501187233627272: [0.1149], 0.999: [0.499175]}}
#       elif k == 8:
#         ber = {'BKLC': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [0.0, 0.0], 0.015848931924611134: [0.0, 0.000375], 0.03162277660168379: [0.0, 0.004125], 0.0630957344480193: [0.00225, 0.024875], 0.12589254117941676: [0.011, 0.104], 0.25118864315095796: [0.05775, 0.290375], 0.501187233627272: [0.234], 0.999: [0.50075]}}
#     elif N == 8:
#       if k == 4:
#         ber = {'BKLC': {0.001: [2.5e-05], 0.001995262314968879: [7.5e-05, 5e-05], 0.003981071705534973: [2.5e-05, 5e-05], 0.007943282347242814: [5e-05, 0.000725], 0.015848931924611134: [0.000875, 0.0036], 0.03162277660168379: [0.002, 0.00975], 0.0630957344480193: [0.0085, 0.036975], 0.12589254117941676: [0.029175, 0.1164], 0.25118864315095796: [0.089625, 0.285975], 0.501187233627272: [0.263], 0.999: [0.507125]}}
#     return ber
#
#
#
# def plot_ber(N=8,k=4,e0=[],graph='BER'):
#   # print(e0)
#   if graph == 'BLER':
#     utils.plot_BSC_BAC(f'BLER Coding Mechanism N={N} k={k}',e0,BLER,k/N)
#     # print(BLER)
#   else:
#     utils.plot_BSC_BAC(f'BER Coding Mechanism N={N} k={k}',e0,BER,k/N)
#     print(BER)
#
#
# N = int(sys.argv[1])
# k = int(sys.argv[2])
#
# if sys.argv[3] == 'BLER':
#  graph = sys.argv[3]
#  nb_pkts = 0
# else:
#   graph = 'BER'
#   nb_pkts = int(sys.argv[3])
#
# e0 = np.logspace(-3,0,11)
# e0[len(e0)-1] = e0[len(e0)-1]-0.001
# e1 = [t for t in e0 if t<=0.5]
#
# BLER = {}
# BER = {}
# if sys.argv[4] =='saved':
#   aux = saved_results(N, k, graph)
#   if sys.argv[3] == 'BLER':
#     BLER = aux
#   else:
#     BER = aux
#
# ################## \Python3\python.exe test_NN_ber_calculator.py 8 4 1000 saved
#
#
#
# # uncoded(k, nb_pkts, graph)
# import time
#
#
#
# # start = time.time()
# # polar_codes_NN(N, k, nb_pkts, graph)
# # end = time.time()
# # print('NN - total ',end - start)
#
# # linear_codes_NN(N, k, nb_pkts, graph)
# autoencoder_NN(N, k, nb_pkts, graph)
#
# plot_ber(N,k,e0,graph)
#
#
#
