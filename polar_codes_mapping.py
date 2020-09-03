#!/usr/bin/python
import sys
import numpy as np
import BACchannel as bac
import polar_codes_generator as polar
import flip_codes as flip
import utils
import matrix_codes as mat_gen

################################################################################ python polar_codes_mapping.py 8 4 BER 2000

### ============ Uncoded ======================================
def uncoded(k=4, nb_pkts = 100, graph = 'BER'):
  print('-------------------Uncoded-----------------------------')
  key = f"Uncode"
  N = k
  U_k = bac.symbols_generator(k)  # all possible messages
  if graph == 'BLER':
    BLER[key] = utils.block_error_probability(N,k,U_k,e0,e1)
  else:
    BER[key] = utils.bit_error_probability(N,k,U_k,nb_pkts,e0,e1,False)
    # print(BER[key])

### ============ Polar codes======================================
def polar_codes(N=8, k=4, nb_pkts = 100, graph = 'BER',channel='BSC'):
  print('-------------------Polar Code-----------------------------')
  for key in [0.1]:
    G, infoBits = polar.polar_generator_matrix(N,k, channel, key)
    k = len(G)
    N = len(G[1])
    U_k = bac.symbols_generator(k)  # all possible messages
    C = bac.matrix_codes(U_k, k, G, N)
    # print('Polar codebook', np.array(C))
    if graph == 'BLER':
      BLER[f"Polar({key})"] = utils.block_error_probability(N,k,C,e0,e1)
    else:
      BER[f"Polar({key})"] = utils.bit_error_probability(N,k,C,nb_pkts,e0,e1)
      # print(BER[f"Polar({key})"])

### ============ Linear codes + Mapping======================================
def linear_codes_mapping(N=8, k=4, nb_pkts = 100, graph = 'BER'):
  print('-------------------Linear Code + Mapping-----------------------------')
  G = mat_gen.matrix_codes(64,k,'linear')
  if G!= []:
    for key in [0.55]:
      k = len(G)
      Nt = len(G[1])
      t = int(Nt/N)
      U_k= bac.symbols_generator(k)  # all possible messages
      X_m = bac.symbols_generator(t)  # all possible symbol sequences
      C = bac.matrix_codes(U_k, k, G, Nt)
      nx = 2**t*key
      # print('nx', nx, 't', t)
      x = utils.mapping(C, X_m, t, nx) #codebook after mapping
      N = len(x[1])

      if graph == 'BLER':
        BLER[f"L+M({key})"] = utils.block_error_probability(N,k,x,e0,e1)
      else:
        BER[f"L+M({key})"] = utils.bit_error_probability(N,k,x,nb_pkts,e0,e1)
        # print(BER[f"L+M({key})"])

### ============ Polar codes + Mapping ======================================
def polar_codes_mapping(N=8, k=4, nb_pkts = 100, graph = 'BER',channel='BSC'):
  print('-------------------Polar Codes + Mapping-----------------------------')
  cont = 2
  if channel == 'AWGN':
    design_parameter = np.linspace(0.0, 10, cont)
  else:
    design_parameter = np.linspace(0.0001, 0.1, cont)

  for key in [0.5]:
    e_design = 0.1
    # print('===============Design================',key)
    G,infoBits = polar.polar_generator_matrix(64, k, channel, e_design)

    k = len(G)
    Nt = len(G[1])
    t = int(Nt /N)
    U_k = bac.symbols_generator(k)  # all possible messages
    X_m = bac.symbols_generator(t)  # all possible symbol sequences
    C = bac.matrix_codes(U_k, k, G, Nt)

    nx = 2**t*key
    # print('nx', nx, 't', t)
    x = utils.mapping2(C, X_m, t, nx)
    N = len(x[1])
    if graph == 'BLER':
      BLER[f"P({e_design})+M({key})"] = utils.block_error_probability(N,k,x,e0,e1)
    else:
      BER[f"P({e_design})+M({key})"] = utils.bit_error_probability(N,k,x,nb_pkts,e0,e1)
      # print(BER[f"P({e_design})+M({key})"])

### ============ Polar codes Integrated Scheme ======================================
def integrated_scheme(N=8, k=4, nb_pkts = 100, graph = 'BER',channel='BSC'):
  print('-------------------Integrated Scheme Code-----------------------------')
  for key in [0.5]:
    G, infoBits = polar.polar_generator_matrix(64, k, channel, 0.1)
    k = len(G)
    Nt = len(G[1])
    t = int(Nt / N)

    U_k = bac.symbols_generator(k)  # all possible messages
    C = utils.integrated_function(infoBits,U_k,k,Nt,-1)

    X_m = bac.symbols_generator(t)  # all possible symbol sequences
    nx = 2 ** t * key
    # print('nx', nx, 't', t)
    x = utils.mapping(C, X_m, t, nx)
    N = len(x[1])
    if graph == 'BLER':
      BLER[f"Int_P({key})"] = utils.block_error_probability(N, k, C,e0,e1)
    else:
      BER[f"Int_P({key})"] = utils.bit_error_probability(N, k, x, nb_pkts,e0,e1)
      # print(BER[f"Int_P({key})"])

### ============ BCH ======================================
def bch_codes(N=8, k=4, nb_pkts = 100, graph = 'BER'):
  print('-------------------BCH Code-----------------------------')
  G = mat_gen.matrix_codes(N, k, 'bch')
  if G != []:
    for key in [0]:
      # print('G = ', np.array(G))
      k = len(G)
      N = len(G[1])
      U_k = bac.symbols_generator(k)  # all possible messages
      C = bac.matrix_codes(U_k, k, G, N)
      print('k ',k,'N ',N)
      if graph == 'BLER':
        BLER[f"BCH({key})"] = utils.block_error_probability(N,k,C,e0,e1)
      else:
        BER[f"BCH({key})"] = utils.bit_error_probability(N,k,C, nb_pkts,e0,e1)
        # print(BER[f"BCH({key})"])

### ============ Linear codes======================================
def linear_codes(N=8, k=4, nb_pkts = 100, graph = 'BER'):
  print('-------------------Linear Code-----------------------------')
  for key in ['BKLC']:
    print(key)
    G = mat_gen.matrix_codes(N, k, key)
    if G != []:
      # print('G = ', np.array(G))
      k = len(G)
      N = len(G[1])
      U_k = bac.symbols_generator(k)  # all possible messages
      C = bac.matrix_codes(U_k, k, G, N)
      print('k ',k,'N ',N)
      if graph == 'BLER':
        BLER[key] = utils.block_error_probability(N,k,C,e0,e1)
      else:
        BER[key] = utils.bit_error_probability(N,k,C, nb_pkts,e0,e1)
        # print(BER[key])


### ============ Figures ======================================

def saved_results(N=8, k=4, graph = 'BER'):
  ber = {}
  bler = {}
  if graph == 'BER':
    if N == 16:
      if k == 4:
        ber = {'Uncode': {0.001: [0.002225], 0.001995262314968879: [0.0018, 0.002075], 0.003981071705534973: [0.0033, 0.00495], 0.007943282347242814: [0.004875, 0.00755], 0.015848931924611134: [0.008675, 0.01695], 0.03162277660168379: [0.017475, 0.032075], 0.0630957344480193: [0.033475, 0.06445], 0.12589254117941676: [0.0637, 0.128375], 0.25118864315095796: [0.12605, 0.25245], 0.501187233627272: [0.247025], 0.999: [0.5046]},
              'Polar(0.1) ': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [0.0, 0.0], 0.015848931924611134: [0.0, 0.0], 0.03162277660168379: [0.0, 5e-05], 0.0630957344480193: [0.0001, 0.002725], 0.12589254117941676: [0.000725, 0.03065], 0.25118864315095796: [0.00775, 0.204375], 0.501187233627272: [0.102425], 0.999: [0.5037]},
              'BCH(0)': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [0.0, 0.0], 0.015848931924611134: [0.0, 0.0], 0.03162277660168379: [0.0, 0.000125], 0.0630957344480193: [7.5e-05, 0.00255], 0.12589254117941676: [0.000975, 0.028375], 0.25118864315095796: [0.00935, 0.197725], 0.501187233627272: [0.115025], 0.999: [0.501475]},
              'BKLC': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [0.0, 0.0], 0.015848931924611134: [0.0, 0.0], 0.03162277660168379: [0.0, 0.000125], 0.0630957344480193: [0.000175, 0.00275], 0.12589254117941676: [0.0012, 0.03105], 0.25118864315095796: [0.009825, 0.20215], 0.501187233627272: [0.1149], 0.999: [0.499175]},
              'L+M(0.55)': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 2.5e-05], 0.007943282347242814: [0.0, 0.0001], 0.015848931924611134: [5e-05, 0.000225], 0.03162277660168379: [0.0001, 0.0006], 0.0630957344480193: [0.000375, 0.003775], 0.12589254117941676: [0.002075, 0.026675], 0.25118864315095796: [0.00995, 0.175575], 0.501187233627272: [0.0828], 0.999: [0.5001]},
              'P(0.1)+M(0.5)': {0.001: [0.0], 0.001995262314968879: [0.0, 5e-05], 0.003981071705534973: [2.5e-05, 5e-05], 0.007943282347242814: [0.0, 0.0001], 0.015848931924611134: [0.0001, 0.000125], 0.03162277660168379: [0.00015, 0.00175], 0.0630957344480193: [0.00055, 0.007375], 0.12589254117941676: [0.0017, 0.04555], 0.25118864315095796: [0.011475, 0.2178], 0.501187233627272: [0.104675], 0.999: [0.502275]},
              'Int_P(0.5)': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [0.0, 0.0], 0.015848931924611134: [0.0, 0.0], 0.03162277660168379: [0.0, 0.00015], 0.0630957344480193: [5e-05, 0.0023], 0.12589254117941676: [0.000575, 0.03225], 0.25118864315095796: [0.007125, 0.207425], 0.501187233627272: [0.0857], 0.999: [0.5022]}}


      elif k == 8:
        ber = {'Uncode': {0.001: [0.002625], 0.001995262314968879: [0.000875, 0.00225], 0.003981071705534973: [0.0045, 0.003875], 0.007943282347242814: [0.004125, 0.00925], 0.015848931924611134: [0.008625, 0.018], 0.03162277660168379: [0.0185, 0.03375], 0.0630957344480193: [0.038125, 0.05975], 0.12589254117941676: [0.063875, 0.120625], 0.25118864315095796: [0.131125, 0.246875], 0.501187233627272: [0.259375], 0.999: [0.497125]},
              'Polar(0.1)': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.00025], 0.007943282347242814: [0.000625, 0.0], 0.015848931924611134: [0.000625, 0.006625], 0.03162277660168379: [0.00625, 0.022875], 0.0630957344480193: [0.01375, 0.07925], 0.12589254117941676: [0.0475, 0.18], 0.25118864315095796: [0.13975, 0.3805], 0.501187233627272: [0.315375], 0.999: [0.502125]},
              'BCH(0)': {0.001: [0.000375], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.00025], 0.007943282347242814: [0.000125, 0.001375], 0.015848931924611134: [0.000625, 0.003625], 0.03162277660168379: [0.002125, 0.011875], 0.0630957344480193:[0.005375, 0.045875], 0.12589254117941676: [0.028375, 0.141875], 0.25118864315095796: [0.092375, 0.29925], 0.501187233627272: [0.268], 0.999: [0.499875]},
              'BKLC': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [0.0, 0.0], 0.015848931924611134: [0.0, 0.000375], 0.03162277660168379: [0.0, 0.004125], 0.0630957344480193: [0.00225, 0.024875], 0.12589254117941676: [0.011, 0.104], 0.25118864315095796: [0.05775, 0.290375], 0.501187233627272: [0.234], 0.999: [0.50075]},
              'L+M(0.55)': {0.001: [0.0], 0.001995262314968879: [0.000375, 0.0005], 0.003981071705534973: [0.00025, 0.001375], 0.007943282347242814: [0.0, 0.0005], 0.015848931924611134: [0.002875, 0.003125], 0.03162277660168379: [0.002, 0.00975], 0.0630957344480193: [0.007875, 0.049875], 0.12589254117941676: [0.029125, 0.1485], 0.25118864315095796: [0.0955, 0.374875], 0.501187233627272: [0.289], 0.999: [0.495375]},
              'P(0.1)+M(0.5)': {0.001: [0.00875], 0.001995262314968879: [0.00625, 0.00725], 0.003981071705534973: [0.0065, 0.006625], 0.007943282347242814: [0.01025, 0.009625], 0.015848931924611134: [0.00875, 0.016125], 0.03162277660168379: [0.01325, 0.02875], 0.0630957344480193: [0.020375, 0.064875], 0.12589254117941676: [0.042375, 0.1735], 0.25118864315095796: [0.11625, 0.36175], 0.501187233627272: [0.30425], 0.999: [0.4965]},
              'Int_P(0.5)': {0.001: [0.12375], 0.001995262314968879: [0.13375, 0.128], 0.003981071705534973: [0.1315, 0.1265], 0.007943282347242814: [0.1265, 0.12625], 0.015848931924611134: [0.12125, 0.12575], 0.03162277660168379: [0.121, 0.128375], 0.0630957344480193: [0.127625, 0.165125], 0.12589254117941676: [0.144, 0.225375], 0.25118864315095796: [0.193, 0.383125], 0.501187233627272: [0.326], 0.999: [0.4925]}}
    elif N == 8:
      if k == 4:
        ber = {'Uncode': {0.001: [0.001875], 0.001995262314968879: [0.002075, 0.002175], 0.003981071705534973: [0.003625, 0.0033], 0.007943282347242814: [0.004875, 0.00905], 0.015848931924611134: [0.009025, 0.0168], 0.03162277660168379: [0.0182, 0.032125], 0.0630957344480193: [0.035075, 0.0627], 0.12589254117941676: [0.063325, 0.126325], 0.25118864315095796: [0.12725, 0.25295], 0.501187233627272: [0.250975], 0.999: [0.50225]},
              'Polar(0.1)': {0.001: [0.0001], 0.001995262314968879: [5e-05, 0.0], 0.003981071705534973: [0.0, 0.0001], 0.007943282347242814: [0.0001, 0.000575], 0.015848931924611134: [0.000825, 0.003125], 0.03162277660168379: [0.00135,0.010825], 0.0630957344480193: [0.0082, 0.0382], 0.12589254117941676: [0.0333, 0.119075], 0.25118864315095796: [0.102225, 0.314875], 0.501187233627272: [0.29455], 0.999: [0.5023]},
              'BCH(0)': {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0001, 0.000325], 0.007943282347242814: [0.000125, 0.00075], 0.015848931924611134: [0.001125, 0.002075], 0.03162277660168379: [0.002075, 0.008025], 0.0630957344480193: [0.00935, 0.032175], 0.12589254117941676: [0.025925, 0.100325], 0.25118864315095796: [0.085, 0.266325], 0.501187233627272: [0.24555], 0.999: [0.50235]},
              'BKLC': {0.001: [2.5e-05], 0.001995262314968879: [7.5e-05, 5e-05], 0.003981071705534973: [2.5e-05, 5e-05], 0.007943282347242814: [5e-05, 0.000725], 0.015848931924611134: [0.000875, 0.0036], 0.03162277660168379: [0.002, 0.00975], 0.0630957344480193: [0.0085, 0.036975], 0.12589254117941676: [0.029175, 0.1164], 0.25118864315095796: [0.089625, 0.285975], 0.501187233627272: [0.263], 0.999: [0.507125]},
              'L+M(0.55)': {0.001: [0.0019], 0.001995262314968879: [0.00195, 0.002025], 0.003981071705534973: [0.001775, 0.00325], 0.007943282347242814: [0.003425, 0.005375], 0.015848931924611134: [0.00475, 0.0145], 0.03162277660168379: [0.008175, 0.02685], 0.0630957344480193: [0.01685, 0.06045], 0.12589254117941676: [0.035375, 0.13545], 0.25118864315095796: [0.090625, 0.3], 0.501187233627272: [0.220875], 0.999: [0.502775]},
              'P(0.1)+M(0.5)': {0.001: [0.00105], 0.001995262314968879: [0.00125, 0.001], 0.003981071705534973: [0.001325, 0.0025], 0.007943282347242814: [0.00185, 0.0048], 0.015848931924611134: [0.00285, 0.01065], 0.03162277660168379: [0.00685, 0.022475], 0.0630957344480193: [0.01505, 0.05655], 0.12589254117941676: [0.034325, 0.14195], 0.25118864315095796: [0.095075, 0.320575], 0.501187233627272: [0.256175], 0.999: [0.497575]},
              'Int_P(0.5)': {0.001: [0.0], 0.001995262314968879: [0.0001, 5e-05], 0.003981071705534973: [0.000175, 2.5e-05], 0.007943282347242814: [0.0002, 0.00065], 0.015848931924611134: [0.000525, 0.00275], 0.03162277660168379: [0.0022, 0.01095], 0.0630957344480193: [0.00795, 0.0345], 0.12589254117941676: [0.02565, 0.1225], 0.25118864315095796: [0.08135, 0.306625], 0.501187233627272: [0.24775], 0.999: [0.49835]}}

    return ber
  elif graph == 'BLER':
    if N == 16:
      if k == 4:
        bler = {'Uncode': {0.001: [0.003994003998999962], 0.001995262314968879: [0.005977080666546231, 0.007957194586922567], 0.003981071705534973: [0.009924988552464376, 0.01582944536234132], 0.007943282347242814: [0.01776694849809124, 0.031396555750146105], 0.015848931924611134: [0.033274420643965574, 0.061904457030626414], 0.03162277660168379: [0.06366647340736609, 0.1206165975131418], 0.0630957344480193: [0.12215968034298319, 0.22948541319954663], 0.12589254117941676: [0.23063789666190093, 0.4162064337867246], 0.25118864315095796: [0.4167453026090392, 0.6855948219086256], 0.501187233627272: [0.6854351958031142], 0.999: [0.9375]},
                'Polar(0.1)': {0.001: [4.663024411044603e-10], 0.001995262314968879: [1.7091809079161635e-09, 7.350769215541675e-09], 0.003981071705534973: [9.438784642767928e-09, 1.1522459630874948e-07], 0.007943282347242814: [6.951947639777245e-08, 1.7860678352965564e-06], 0.015848931924611134: [6.66037556995569e-07, 2.7069244676325432e-05], 0.03162277660168379: [6.242399077871497e-06, 0.0003917722801248802], 0.0630957344480193: [7.59253863791276e-05, 0.00514875982608598], 0.12589254117941676: [0.0010230996422875283, 0.05493238665145861], 0.25118864315095796: [0.01404410487288088, 0.3679267771663388], 0.501187233627272: [0.17241850492532151], 0.999: [0.9375]},
                'BCH(0)': {0.001: [4.873073056188559e-10], 0.001995262314968879: [2.255736708534073e-09, 7.673217394810194e-09], 0.003981071705534973: [1.571178764869785e-08, 1.2017159656263487e-07], 0.007943282347242814: [1.0079732470913427e-07, 1.8595979613955649e-06], 0.015848931924611134: [7.784285169787353e-07, 2.8092172501659185e-05], 0.03162277660168379: [7.991371429483252e-06, 0.0004041636357504652], 0.0630957344480193: [0.00010192214992688076, 0.005259002462390527], 0.12589254117941676: [0.0012593338957352929, 0.05535283225745713], 0.25118864315095796: [0.01738150257685478, 0.36695327549905676], 0.501187233627272: [0.19896145469122506], 0.999: [0.9375]},
                'BKLC': {0.001: [4.870062131345776e-10], 0.001995262314968879: [1.9987611565852603e-09, 7.673194857282795e-09], 0.003981071705534973: [1.1444556635709091e-08, 1.2017225348159855e-07], 0.007943282347242814: [8.425259179212219e-08, 1.8595988041658629e-06], 0.015848931924611134: [7.777258022034772e-07, 2.8092171812210687e-05], 0.03162277660168379: [7.703228494082559e-06, 0.00040416363593032134], 0.0630957344480193: [9.362370801957454e-05, 0.005259002462257412], 0.12589254117941676: [0.0012523783397200283, 0.05535283225738796], 0.25118864315095796: [0.016975059456702746, 0.366953275499072], 0.501187233627272: [0.19789749211702912], 0.999: [0.9375]},
                'L+M(0.55)': {0.001: [2.6246691446907278e-06], 0.001995262314968879: [4.804419578396946e-06, 1.0449011468849356e-05], 0.003981071705534973: [1.1028153132919272e-05, 4.16140838627177e-05], 0.007943282347242814: [3.100911487796676e-05, 0.0001660399719476402], 0.015848931924611134: [0.00010181792993968486, 0.0006677530334779913], 0.03162277660168379: [0.00037324974073627004, 0.0027677607704662543], 0.0630957344480193: [0.0014983587572872104, 0.012609721529144524], 0.12589254117941676: [0.006346006573917817, 0.06842031429361506], 0.25118864315095796: [0.02999478148492607, 0.3662827625965799], 0.501187233627272: [0.19850518359097646], 0.999: [0.9375]},
                'P(0.1)+M(0.5)': {0.001: [1.5287956189879992e-06], 0.001995262314968879: [2.5750097986110276e-06, 6.200693034696947e-06], 0.003981071705534973: [5.210557184054387e-06, 2.5598520945857572e-05], 0.007943282347242814: [1.321196431935956e-05, 0.00010922160595705499], 0.015848931924611134: [3.6361041869326094e-05, 0.0004936812836560112], 0.03162277660168379: [0.00012159278219425751, 0.0024404756123660443], 0.0630957344480193: [0.0005035907812881435, 0.01346199658525049], 0.12589254117941676: [0.002743351151984985, 0.07905293240305267], 0.25118864315095796: [0.0176670628846054, 0.3919357253480511], 0.501187233627272: [0.1511403402786069], 0.999: [0.9375]},
                'Int_P(0.5)': {0.001: [0.7500000000175159], 0.001995262314968879: [0.7500000000179664, 0.7500000002760664], 0.003981071705534973: [0.7500000000313118, 0.7500000043541152], 0.007943282347242814: [0.7500000004513712, 0.750000068349369], 0.015848931924611134: [0.7500000134701702, 0.7500010627263365], 0.03162277660168379: [0.7500000205733874, 0.750016206264416], 0.0630957344480193: [0.7500004091634982, 0.7502374684307448], 0.12589254117941676: [0.7500078639474461, 0.7531917176614641], 0.25118864315095796: [0.7501054167754733, 0.7845456541073789], 0.501187233627272: [0.7529793520801775], 0.999: [0.9375]},
                'Int_P(0.9)': {0.001: [0.7500000000175159], 0.001995262314968879: [0.7500000000179664, 0.7500000002760664], 0.003981071705534973: [0.7500000000313118, 0.7500000043541152], 0.007943282347242814: [0.7500000004513712, 0.750000068349369], 0.015848931924611134: [0.7500000134701702, 0.7500010627263365], 0.03162277660168379: [0.7500000205733874, 0.750016206264416], 0.0630957344480193: [0.7500004091634982, 0.7502374684307448], 0.12589254117941676: [0.7500078639474461, 0.7531917176614641], 0.25118864315095796: [0.7501054167754733, 0.7845456541073789], 0.501187233627272: [0.7529793520801775], 0.999:[0.9375]}}

      elif k == 8:
        bler = {'Uncode': {0.001: [0.0018375], 0.001995262314968879: [0.00208125, 0.00203125], 0.003981071705534973: [0.0029375, 0.00398125], 0.007943282347242814: [0.0049375, 0.00785], 0.015848931924611134: [0.0091, 0.015825], 0.03162277660168379: [0.0171375, 0.032675], 0.0630957344480193: [0.03314375, 0.0641625], 0.12589254117941676: [0.06451875, 0.12606875], 0.25118864315095796: [0.12760625, 0.25244375], 0.501187233627272: [0.251975], 0.999: [0.50184375]},
                'Polar(0.1)': {0.001: [5.625e-05], 0.001995262314968879: [0.00013125, 0.000125], 0.003981071705534973: [0.000175, 0.00043125], 0.007943282347242814: [0.00063125, 0.0016625], 0.015848931924611134: [0.00165, 0.005325], 0.03162277660168379: [0.00366875, 0.021475], 0.0630957344480193: [0.0136625, 0.0685], 0.12589254117941676: [0.04769375, 0.1856], 0.25118864315095796: [0.1312375, 0.37719375], 0.501187233627272: [0.32815], 0.999: [0.49995625]},
                'BCH(0)': {0.001: [8.75e-05], 0.001995262314968879: [6.25e-05, 5e-05], 0.003981071705534973: [0.00013125, 0.00026875], 0.007943282347242814: [0.00023125, 0.00080625], 0.015848931924611134: [0.00044375, 0.00365], 0.03162277660168379: [0.0018125, 0.01293125], 0.0630957344480193: [0.007025, 0.0442875], 0.12589254117941676: [0.0258375, 0.1336625], 0.25118864315095796: [0.09119375, 0.2986625], 0.501187233627272: [0.264575], 0.999: [0.50145]},
                'BKLC': {0.001: [1.875e-05], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [2.5e-05, 0.00011875], 0.015848931924611134: [8.125e-05, 0.00055], 0.03162277660168379: [0.00045625, 0.00354375], 0.0630957344480193: [0.001875, 0.02281875], 0.12589254117941676: [0.010375, 0.10095], 0.25118864315095796: [0.05811875, 0.2904875], 0.501187233627272: [0.241625], 0.999: [0.49931875]},
                'L+M(0.55)': {0.001: [0.00028125], 0.001995262314968879: [0.0002875, 0.0004875], 0.003981071705534973: [0.00031875, 0.00074375], 0.007943282347242814: [0.00074375, 0.00180625], 0.015848931924611134: [0.0014875, 0.0053125], 0.03162277660168379: [0.00279375, 0.01525625], 0.0630957344480193: [0.00936875, 0.0501125], 0.12589254117941676: [0.0276875, 0.16263125], 0.25118864315095796: [0.09575, 0.36948125], 0.501187233627272: [0.30030625], 0.999: [0.50085]},
                'P(0.1)+M(0.5)': {0.001: [0.02591875], 0.001995262314968879: [0.026875, 0.0267875], 0.003981071705534973: [0.02643125, 0.02705625], 0.007943282347242814: [0.029975, 0.03229375], 0.015848931924611134: [0.0315, 0.040325], 0.03162277660168379: [0.0385125, 0.05834375], 0.0630957344480193: [0.0501875, 0.10281875], 0.12589254117941676: [0.0769625, 0.20686875], 0.25118864315095796: [0.15371875, 0.39544375], 0.501187233627272: [0.32949375], 0.999: [0.49814375]},
                'Int_P(0.5)': {0.001: [0.12483125], 0.001995262314968879: [0.1248375, 0.1267125], 0.003981071705534973: [0.124425, 0.1256], 0.007943282347242814: [0.1253, 0.126575], 0.015848931924611134: [0.1259125, 0.1279125], 0.03162277660168379: [0.1276875, 0.13408125], 0.0630957344480193: [0.1288375, 0.155725], 0.12589254117941676: [0.14586875, 0.2274], 0.25118864315095796: [0.19374375, 0.3874875], 0.501187233627272: [0.337825], 0.999: [0.50081875]}}
    elif N == 8:
      if k == 4:
        bler = {'Uncode': {0.001: [0.003994003998999962], 0.001995262314968879: [0.005977080666546231, 0.007957194586922567], 0.003981071705534973: [0.009924988552464376, 0.01582944536234132], 0.007943282347242814: [0.01776694849809124, 0.031396555750146105], 0.015848931924611134: [0.033274420643965574, 0.061904457030626414], 0.03162277660168379: [0.06366647340736609, 0.1206165975131418], 0.0630957344480193: [0.12215968034298319, 0.22948541319954663], 0.12589254117941676: [0.23063789666190093, 0.4162064337867246], 0.25118864315095796: [0.4167453026090392, 0.6855948219086256], 0.501187233627272: [0.6854351958031142], 0.999: [0.9375]},
                'Polar(0.1)':  {0.001: [2.0930104921101922e-05], 0.001995262314968879: [4.170985086604517e-05, 8.304813753134965e-05], 0.003981071705534973: [0.00010374382967748286, 0.0003284371599558966], 0.007943282347242814: [0.0003093936845051104, 0.0012903426820463082], 0.015848931924611134: [0.0010452374886334992, 0.005002828092101952], 0.03162277660168379: [0.0038025282255498283, 0.01888878413491335], 0.0630957344480193: [0.014334037680204914,0.06760162311202733], 0.12589254117941676: [0.05184532653776541, 0.21701403305161693], 0.25118864315095796: [0.16909150572799436, 0.5580107458703938], 0.501187233627272: [0.4907785921994958], 0.999: [0.9375]},
                'BCH(0)': {0.001: [2.0930104917771253e-05], 0.001995262314968879: [4.686627713423874e-05, 8.304813753312601e-05], 0.003981071705534973: [0.0001291809046302106, 0.00032843715995367617], 0.007943282347242814: [0.00041369298568405544, 0.0012903426820485286], 0.015848931924611134: [0.0014491383091672638, 0.005002828092105838], 0.03162277660168379: [0.005291917896825038, 0.018888784134911463], 0.0630957344480193: [0.015436204839802015, 0.06760162311202789], 0.12589254117941676: [0.05219256548064566, 0.2170140330516156], 0.25118864315095796: [0.17326993904123578, 0.5580107458703947], 0.501187233627272: [0.5030267527221204], 0.999: [0.9375]},
                'BKLC': {0.001: [2.0930104921101922e-05], 0.001995262314968879: [4.170985086604517e-05, 8.304813753134965e-05], 0.003981071705534973: [0.00010374382967748286, 0.0003284371599558966], 0.007943282347242814: [0.0003093936845051104, 0.0012903426820463082], 0.015848931924611134: [0.0010452374886334992, 0.005002828092101952], 0.03162277660168379: [0.0038025282255498283, 0.01888878413491335], 0.0630957344480193: [0.014334037680204914, 0.06760162311202733], 0.12589254117941676: [0.05184532653776541, 0.21701403305161693], 0.25118864315095796: [0.16909150572799436, 0.5580107458703938], 0.501187233627272: [0.4907785921994958], 0.999: [0.9375]},
                'L+M(0.55)': {0.001: [0.0019431064481476579], 0.001995262314968879: [0.0024495295763085068, 0.0038880669854024497], 0.003981071705534973: [0.0034717521059396406, 0.007801308144549934], 0.007943282347242814: [0.005558110549595985, 0.015735762651908458], 0.015848931924611134: [0.009904627910614971, 0.03204703178603696], 0.03162277660168379: [0.019294792547481454, 0.06631832071323585], 0.0630957344480193: [0.03868852476446627, 0.14016410664955914], 0.12589254117941676: [0.08406121988508519, 0.2991765639906231], 0.25118864315095796: [0.197107344422361, 0.6016296899477842], 0.501187233627272: [0.4806917493801961], 0.999: [0.9375]},
                'P(0.1)+M(0.5)': {0.001: [0.0020049840050305745], 0.001995262314968879: [0.002637158674338891, 0.004010302976008795], 0.003981071705534973: [0.003918263756833129, 0.008040379820742016], 0.007943282347242814: [0.0065522255590596545, 0.016194045089280462], 0.015848931924611134: [0.012111059426292314, 0.03289045081495423], 0.03162277660168379: [0.024359202379581224, 0.06774537459839003], 0.0630957344480193: [0.04632106116108725, 0.1421807978674262], 0.12589254117941676: [0.09469098741272519, 0.30104615205841967], 0.25118864315095796: [0.20929448821277952, 0.6020829454405856], 0.501187233627272: [0.48730806763295853], 0.999: [0.9375]},
                'Int_P(0.5)': {0.001: [0.8750000000043652], 0.001995262314968879: [0.8750000000044716, 0.8750000000690084], 0.003981071705534973: [0.8750000000078266, 0.8750000010884859], 0.007943282347242814: [0.8750000001128574, 0.8750000170873421], 0.015848931924611134: [0.8750000033675578, 0.875000265681888], 0.03162277660168379: [0.8750000051433401, 0.8750040516317886], 0.0630957344480193: [0.8750001022909015, 0.8750593812121678],0.12589254117941676: [0.8750019660023436, 0.8758004925688065], 0.25118864315095796: [0.8750263569726182, 0.8839573500031664], 0.501187233627272: [0.8757470704772243], 0.999: [0.9375]}}

    return bler


def plot_ber(N=8,k=4,e0=[],graph='BER'):
  # print(e0)
  if graph == 'BLER':
    utils.plot_BSC_BAC(f'BLER Coding Mechanism N={N} k={k}',e0,BLER,k/N)
    print(BLER)
  else:
    utils.plot_BSC_BAC(f'BER Coding Mechanism N={N} k={k}',e0,BER,k/N)
    print(BER)


N = int(sys.argv[1])
k = int(sys.argv[2])

if sys.argv[3] == 'BLER':
 graph = sys.argv[3]
 nb_pkts = 0
else:
  graph = 'BER'
  nb_pkts = int(sys.argv[3])

# e0 = np.linspace(0.1, 0.9, 9)
# e1 = e0
e0 = np.logspace(-3,0,11)
e0[len(e0)-1] = e0[len(e0)-1]-0.001
e1 = [t for t in e0 if t<=0.5]

################## python polar_codes_mapping.py 8 4 1000
################## python polar_codes_mapping.py 8 4 BLER
BLER = {}
BER = {}
aux = saved_results(N, k, graph)

if sys.argv[3] == 'BLER':
  BLER = aux
else:
  BER = aux

# uncoded(k, nb_pkts, graph)
# polar_codes(N, k, nb_pkts, graph)
# bch_codes(N, k, nb_pkts, graph)
# linear_codes(N, k, nb_pkts, graph)
#
# linear_codes_mapping(N, k, nb_pkts, graph)
# polar_codes_mapping(N, k, nb_pkts, graph)
# integrated_scheme(N, k, nb_pkts, graph)

plot_ber(N,k,e0,graph)



