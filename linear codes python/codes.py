#!/usr/bin/python
import sys
import time
import numpy as np
import BACchannel as bac

def hadamard_codes(msm,k):
	codes = []	
	for a in range(2**k):
		row = [sum([i*j for (i, j) in zip(msm[b], msm[a])])%2 for b in range(2**k)]
		codes.append(row)
	return codes

def convolutive_codes(msm,k):
	codes=[]	
	for m in msm:
		m.append(0)
		m.append(0)
		d1 = 0
		d2 = 0
		row = []
		for a in m:
			c1 = a^d1^d2
			c2 = a^d2
			d2 = d1
			d1 = a
			row.append(c1)
			row.append(c2)
		codes.append(row)
	return codes

################################################################################
print(sys.argv[1])
if sys.argv[1] == 'hadamard':
	t = 'hadamard' 
	k = int(sys.argv[2])
	N = 2**k
	U_k = bac.symbols_generator(k) #all possible messages
	Y_n = bac.symbols_generator(N) #all possible symbol sequences
	C = hadamard_codes(U_k,k)
elif sys.argv[1] == 'convolutive':
	t = 'convolutive' 
	k = 6
	N = 16
	U_k = bac.symbols_generator(k) #all possible messages
	Y_n = bac.symbols_generator(N) #all possible symbol sequences
	C = convolutive_codes(U_k,k)
	U_k = bac.symbols_generator(k) #all possible messages
else: 
	if sys.argv[1] == 'polar':
		t = 'polar'
		if int(sys.argv[2])==16:
			G = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
				[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
				[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
				[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
				[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] 
		else:
			G = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
					[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
		
	elif sys.argv[1] == 'hamming':
		t = 'hamming' 
		G = [[1, 0, 0, 0, 1, 1, 0],
		[0, 1, 0, 0, 0, 1, 1],
		[0, 0, 1, 0, 1, 1, 1],
		[0, 0, 0, 1, 1, 0, 1]]
	elif sys.argv[1] == 'bch':
		t = 'bch'
		if int(sys.argv[2])==7:
			G = [[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
			[0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
			[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
			[0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]]
		elif int(sys.argv[2])==5:
			G = [[1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
			[0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1],
			[0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
			[0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
			[0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]]
		else:
			G = [[1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
			[0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
			[0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
			[0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]]
	elif sys.argv[1] == 'reedmuller':
		t = 'reedmuller'
		if int(sys.argv[2])==5:
			G = [	[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
			[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
			[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
			[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
			[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]
		else:
			G = [[1, 0, 0, 1, 0, 1, 1, 0],
			[0, 1, 0, 1, 0, 1, 0, 1],
			[0, 0, 1, 1, 0, 0, 1, 1],
			[0, 0, 0, 0, 1, 1, 1, 1]]
	elif sys.argv[1] == 'linear':
		t = 'linear'
		G = [[1, 0, 0, 1, 0, 1, 0, 0],
		[0, 1, 0, 1, 0, 1, 0, 0],
		[0, 0, 1, 0, 1, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 0, 1]]
	elif sys.argv[1] == 'cyclic':
		t = 'cyclic'
		G = [[1, 0, 0, 0, 1, 0, 1],
		[0, 1, 0, 0, 1, 1, 1],
		[0, 0, 1, 0, 1, 1, 0],
		[0, 0, 0, 1, 0, 1, 1]]
	elif sys.argv[1] == 'best':
		t = 'best'
		if int(sys.argv[2])==16:
			#G = [[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
			#[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
			#[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
			#[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]
			G = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
			[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
			[0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
			[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1]]
		elif int(sys.argv[2])==15:
			G = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
			[0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
			[0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
			[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1]]
		else:
			G = [[1, 0, 0, 1, 0, 1, 1, 0],
			[0, 1, 0, 1, 0, 1, 0, 1],
			[0, 0, 1, 1, 0, 0, 1, 1],
			[0, 0, 0, 0, 1, 1, 1, 1]]
	elif sys.argv[1] == 'alternant':
		t = 'alternant'
		G = [[1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
		[0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
		[0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
		[0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]]		
	else:
		raise IOError('ERROR selected code does not exists')
	k = len(G)
	N = len(G[1])
	U_k = bac.symbols_generator(k) #all possible messages
	Y_n = bac.symbols_generator(N) #all possible symbol sequences
	C = bac.matrix_codes(U_k,k,G,N)


e0 = np.linspace(0.1,0.9,9)
e1 = np.linspace(0.1,0.1,1)

print('Type',t,'k=',k,' N=',N)
print("0.00",'|',["{:.4f}".format(ep1) for ep1 in e1])
print('------------------------------------------------------------------')
for ep0 in e0:
	row = []
	for ep1 in (ep1 for ep1 in e1 if ep1+ep0<=1 and ep1<=ep0):
		a = bac.succes_probability(Y_n,C,U_k,ep0,ep1)
		row.append("{:.4f}".format(a))
	print("{:.2f}".format(ep0), '|',row)
  
