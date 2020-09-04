def matrix_codes(N, k, type):
  G = []
  if type == 'linear':
    if N == 64:
      if k == 4:
        G = [ [1,0,0,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1],
              [0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1],
              [0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
              [0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]]
      elif k == 8:
        G = [ [1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,0,1,1,0,0,1,1,1],
              [0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,1,0,0,1,1,1,0,0,1,0,1,1,0,0],
              [0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1],
              [0,0,0,0,0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1,1,0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1],
              [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,0,1,1,1],
              [0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]]
    # G = [ [1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0],
    #       [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
    #       [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
    #       [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
    #       [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    #       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
  elif type == 'bch':
    if N == 16:
        if k == 4:
          G = [	[1,0,0,0,0,1,1,1,0,1,1,0,0,1,0],
                [0,1,0,0,0,0,1,1,1,0,1,1,0,0,1],
                [0,0,1,0,0,1,1,0,1,0,1,1,1,1,0],
                [0,0,0,1,0,0,1,1,0,1,0,1,1,1,1],
                [0,0,0,0,1,1,1,0,1,1,0,0,1,0,1]]
        elif k == 8:
          G = [ [1,0,0,0,0,0,0,0,0,1,0,1,1,1,0],
                [0,1,0,0,0,0,0,0,0,0,1,0,1,1,1],
                [0,0,1,0,0,0,0,0,0,1,0,0,1,0,1],
                [0,0,0,1,0,0,0,0,0,1,1,1,1,0,0],
                [0,0,0,0,1,0,0,0,0,0,1,1,1,1,0],
                [0,0,0,0,0,1,0,0,0,0,0,1,1,1,1],
                [0,0,0,0,0,0,1,0,0,1,0,1,0,0,1],
                [0,0,0,0,0,0,0,1,0,1,1,1,0,1,0],
                [0,0,0,0,0,0,0,0,1,0,1,1,1,0,1]]
        G.pop(0)
    if N == 8:
      if k == 4:
        G = [[1, 0, 0, 0, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 1],
             [0, 0, 1, 0, 1, 1, 1],
             [0, 0, 0, 1, 1, 0, 1]]

  elif type == 'BKLC':
    if N == 16:
      if k == 4:
        G = [[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
             [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]
      elif k == 8:
        G = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1]]
    if N == 8:
      if k==4:
        G = [ [1, 0, 0, 1, 0, 1, 1, 0],
              [0, 1, 0, 1, 0, 1, 0, 1],
              [0, 0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1]]
  elif type == 'Other':
    if N==16:
      if k == 4:
        G = [[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
             [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
              0]]  # This code has been created as a R = ReedMuller(16,5) and then shorted using ShotenCode(R,15) in MAGMA
      if k == 8:
        G = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1,
              1]]  # This code has been created as a C = QRCode(GF(2), 17) and then shorted using ShotenCode(C,13) in MAGMA

  return G

 # BER[f"Polar"] = {0.1: [0.01505], 0.2: [0.049025, 0.1219], 0.30000000000000004: [0.099925, 0.295], 0.4: [0.162175, 0.433375], 0.5: [0.250925, 0.5014], 0.6: [0.3333], 0.7000000000000001: [0.40295], 0.8: [0.467775], 0.9: [0.501625]}
      # BER[f"BCH"] = {0.1: [0.01505], 0.2: [0.04635, 0.112275], 0.30000000000000004: [0.10325, 0.2791], 0.4: [0.16735, 0.4252], 0.5: [0.24625, 0.50225], 0.6: [0.324525], 0.7000000000000001: [0.406675], 0.8: [0.463675], 0.9: [0.498925]}
      # BER[f"BKLC"] = {0.1: [0.01425], 0.2: [0.051825, 0.116275], 0.30000000000000004: [0.102625, 0.2775], 0.4: [0.16905, 0.427325], 0.5: [0.2568, 0.50065], 0.6: [0.333425], 0.7000000000000001: [0.41], 0.8: [0.472325], 0.9: [0.4994]}
      # BER[f"L+M1"] = {0.1: [0.0127], 0.2: [0.039025, 0.0967], 0.30000000000000004: [0.083275, 0.2544], 0.4: [0.145725, 0.40565], 0.5: [0.216925, 0.50245], 0.6: [0.29635], 0.7000000000000001: [0.3701], 0.8: [0.44605], 0.9: [0.498625]}
      # BER[f"Uncoded"] = {0.1: [0.10085], 0.2: [0.14945, 0.200325], 0.30000000000000004: [0.20095, 0.300875], 0.4: [0.24825, 0.40135], 0.5: [0.3036, 0.503775], 0.6: [0.3498], 0.7000000000000001: [0.402725], 0.8: [0.44725], 0.9: [0.497375]}
      # BER[f"L+M2"] = {0.1: [0.0156], 0.2: [0.0424, 0.10665], 0.30000000000000004: [0.08255, 0.253625], 0.4: [0.14585, 0.401075], 0.5: [0.216425, 0.4993], 0.6: [0.29165], 0.7000000000000001: [0.366575], 0.8: [0.4358], 0.9: [0.5023]}