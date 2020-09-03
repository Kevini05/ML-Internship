#!/usr/bin/python
import sys
import time
import BACchannel as bac

def matrix_codes(msm,k,G,N):
	codes = []
	g = []
	for i in range(N):
		g.append([G[j][i] for j in range(k)])
	#print(g)	
	for a in range(2**k):
		row = [sum([i*j for (i, j) in zip(g[b], msm[a])])%2 for b in range(N)]
		codes.append(row)
	return codes

G = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]
k = len(G)
N = len(G[1])
e0 = 0.2
e1 = 0.2
U_k = bac.symbols_generator(k) #all possible messages
Y_n = bac.symbols_generator(N) #all possible symbol sequences
C = matrix_codes(U_k,k,G,N)

# print(C)

for y in Y_n:
	id = bac.MAP_BAC(y,len(U_k[1]),C,e0,e1)
	print('y',y,'g(y)',U_k[id])
a = bac.succes_probability(Y_n,C,U_k,e0,e1)
print(a)	
