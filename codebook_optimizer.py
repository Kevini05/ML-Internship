#!/usr/bin/python
import sys
import time

def linspace(a, b, n=100):
    if n < 2:
        return b
    diff = (float(b) - a)/(n - 1)
    return [diff * i + a  for i in range(n)]

def MAP_BAC(symbols,k,codes,e0,e1): 
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
  
def symbols_generator(N):
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
    id = MAP_BAC(y,k,codes,e0,e1)
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

################################################################################
def codebook_generator_k4(N):
  codebooks = {}
  for t1 in range(1,N):
    for t2 in (t2 for t2 in range(N) if t2>t1): 
      for t3 in (t3 for t3 in range(N) if t3>t2):
        # print('t1=',t1, 't2=',t2, 't3=',t3)
        x1 = [0 for s in range(N-t1)] + [1 for s in range(t1)]
        x2 = [0 for s in range(N-t2)] + [1 for s in range(t2)]
        x3 = [0 for s in range(N-t3)] + [1 for s in range(t3)]
        nx1 = [x1[s]^1 for s in range(N)]
        nx2 = [x2[s]^1 for s in range(N)]
        nx3 = [x3[s]^1 for s in range(N)]
        x1x2 = [x1[s]^x2[s] for s in range(N)]
        nx1x2 = [x1[s]^x2[s]^1 for s in range(N)]
        x1x3 = [x1[s]^x3[s] for s in range(N)]
        nx1x3 = [x1[s]^x3[s]^1 for s in range(N)]
        x2x3 = [x2[s]^x3[s] for s in range(N)]
        nx2x3 = [x2[s]^x3[s]^1 for s in range(N)]
        x1x2x3 = [x1[s]^x2[s]^x3[s] for s in range(N)]
        nx1x2x3 = [x1[s]^x2[s]^x3[s]^1 for s in range(N)]

        code = [[0 for s in range(N)],
              x1,x1x2,x2,x2x3,x1x2x3,x1x3,x3,nx3,nx1x3,nx2x3,nx1x2x3,nx2,nx1x2,nx1,
              [1 for s in range(N)]]
        codebooks[t1,t2,t3] = code
  return codebooks


k = 2
N = 3 #int(sys.argv[1])
b = 1
n = N-b

e0 = linspace(0.1,0.9,9)
e1 = 0.2

print('k=',k,' N=',N, 'e1',e1)

C = codebook_generator_k4(N)
U_k = symbols_generator(k) #all possible messages
Y_n = symbols_generator(N) #all possible symbol sequences

P_success = {}


for ep0 in e0:
	start_time = time.time()
	P_success[ep0]=[0.0,0.0,0.0,0.0]
	for t1 in range(b,n):
		for t2 in (t2 for t2 in range(n) if t2>t1): 
		  for t3 in (t3 for t3 in range(n) if t3>t2):
		    a = succes_probability(Y_n,C[t1,t2,t3],U_k,ep0,e1)
		    # print('t1=',t1, 't2=',t2, 't3=',t3,'a=',"{:.8f}".format(a))
		    if a > P_success[ep0][0]:
		      P_success[ep0] = [a,t1,t2,t3]
	print('e0 =',"{:.2f}".format(ep0),'t1 =',P_success[ep0][1],'t2 =',P_success[ep0][2],'t3=',P_success[ep0][3],'P_c =',"{:.4f}".format(P_success[ep0][0]))
	print("--- %s seconds ---" % (time.time() - start_time))
