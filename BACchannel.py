#!/usr/bin/python
import sys
import time
import random as rd
import math as m
import numpy as np

def linspace(a, b, n=100):
    if n < 2:
        return b
    diff = (float(b) - a)/(n - 1)
    return [diff * i + a  for i in range(n)]

def MAP_BAC(symbols,k,codes,e0,e1):
  """
  :param symbols: Symbols reçus
  :param k: taille du message
  :param codes: codebook (tous les mot-codes de taille N)
  :param e0 et e1: Parametres du canal
  :return: l'index du message décodé sur la matrice de tous les messages possibles
  """
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
  """
  :param symbols: Symbols reçus
  :param k: taille du message
  :param codes: codebook (tous les mot-codes de taille N)
  :param e0 et e1: Parametres du canal
  :return: l'index du message décodé sur la matrice de tous les messages possibles
  """
  if e1+e0==1.0 or e1==0.0 or e0==0:
    y = 0.5
  else:
    y = np.log(e1 / (1 - e0)) / (np.log((e1 * e0) / ((1 - e0) * (1 - e1))))
  # print('e1 =',e1,' e0 =',e0,' y =',y)
  # print('s',code)
  decoded_message = []
  for u in code:
    decoded_message.append(1) if u > y else decoded_message.append(0)
  # print('r',decoded_message)
  return decoded_message

def symbols_generator(N):
  """
  :param N: taille de symbols (numéro de bits)
  :return: toutes les combinaisons de bits possibles de longueur N
  """
  messages = []	
  for i in range(2**N):
     messages.append([0 for a in range(N)])
     nb = bin(i)[2:].zfill(N)  
     for j in range(N):
        messages[i][j] = int(nb[j])
  return messages

def succes_probability(symbols,codes,msm,e0,e1): 
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
		he0	= -e0*m.log(e0,2)-(1-e0)*m.log(1-e0,2)
		he1 = -e1*m.log(e1,2)-(1-e1)*m.log(1-e1,2)
		z = 2.0**((he0-he1)/(1.0-e0-e1))
		q = (z-e0*(1+z))/((1+z)*(1-e0-e1))
	else:
		q = 0.5
	return q


def FEC_encoder(b, G):
  """ Entrée : sequence de bits
      Sortie : sequence générée grâce à la matrice génératrice """
  return np.dot(np.transpose(G), np.transpose(b)) % 2


def BAC_channel(x, epsilon0, epsilon1):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles reçus, bruités """
  n = [0] * len(x)
  # print('e0 ',epsilon0)
  # print('e1 ',epsilon1)
  for i in range(len(x)):
    rdnm = rd.randint(0, 1000) / (1000.0)
    n[i] = x[i] ^ 1 if (rdnm <= epsilon0 and x[i] == 0) or (rdnm <= epsilon1 and x[i] == 1) else x[i]
  return n  # Signal transmis + Bruit


def NbOfErrors(a, b):
  """ Entrée : 2 bit's arrays
      Sortie : numero de bit qui ne sont pas égaux avec un distance de 1e-2 """
  NbErrors = 0
  for i in range(len(a)):
    if np.abs(a[i] - b[i]) > 1e-2:
      NbErrors += 1
  return NbErrors