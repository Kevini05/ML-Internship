#!/usr/bin/python
import numpy as np
from polarcodes import *


def polar_generator_matrix(N=8,k=4,channel_type='AWGN', design_parameter=0.1):
    # initialise polar code
    myPC = PolarCode(N, k)
    myPC.construction_type = 'bb'

    # mothercode construction
    # Construct(myPC, design_parameter, channel_type)
    Construct(myPC, design_parameter)
    # print('Frozen Bits : ',myPC.frozen,'Reliabilities Bits : ',myPC.reliabilities, "\n")

    # # set message
    # my_message = np.random.randint(2, size=myPC.K)
    # myPC.set_message(my_message)
    # print("The message is:", my_message)
    #
    # # encode message
    # Encode(myPC)
    # print("The coded message is:", myPC.u)

    T = arikan_gen(int(np.log2(N)))
    # print(myPC.reliabilities)
    infoBits = [myPC.reliabilities[a] for a in range(len(myPC.reliabilities)-1, len(myPC.reliabilities)-k-1, -1)]
    infoBits.sort()
    G = []

    for j in range(len(T)):
        G.append([T[j][i] for i in infoBits])

    # print(infoBits)
    G = np.array(G)

    return np.transpose(G),infoBits
    # x = np.dot(G,np.transpose(my_message))%2
    # print('Coded with G', x)