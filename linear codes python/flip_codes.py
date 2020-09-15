#!/usr/bin/python
import sys
import time
import BACchannel as bac

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

        # code = [[0 for s in range(N)],
        #       x1,x1x2,x2,x2x3,x1x2x3,x1x3,x3,nx3,nx1x3,nx2x3,nx1x2x3,nx2,nx1x2,nx1,
        #       [1 for s in range(N)]]
        code = [[0 for s in range(N)],
                x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3, nx1, nx2, nx3, nx1x2, nx1x3, nx2x3, nx1x2x3,
                [1 for s in range(N)]]
        codebooks[t1,t2,t3] = code
  return codebooks


# k = 4
# N = int(sys.argv[1])
# b = 1
# n = N-b
# t = 'Flip'
# e0 = bac.linspace(0.6,0.9,4)
# e1 = bac.linspace(0.1,0.7,3)
#
# print('Type',t,'k=',k,' N=',N)
#
# C = codebook_generator_k4(N)
# U_k = bac.symbols_generator(k) #all possible messages
# Y_n = bac.symbols_generator(N) #all possible symbol sequences
#
# P_success = {}
#
# print("0.00",'|',["{:.4f}".format(ep1) for ep1 in e1])
# print('-------------------------------------------------------------------')
# for ep0 in e0:
#   row = []
#   for ep1 in (ep1 for ep1 in e1 if ep1+ep0<=1 and ep1<=ep0):
#     # if ep1 == 0.1:
#     start_time = time.time()
#     P_success[ep0]=[0.0,0.0,0.0,0.0]
#     for t1 in range(b,n):
#       for t2 in (t2 for t2 in range(n) if t2>t1):
#         for t3 in (t3 for t3 in range(n) if t3>t2):
#           a = bac.succes_probability(Y_n,C[t1,t2,t3],U_k,ep0,ep1)
#           # print('t1=',t1, 't2=',t2, 't3=',t3,'a=',"{:.8f}".format(a))
#           if a > P_success[ep0][0]:
#             P_success[ep0] = [a,t1,t2,t3]
#     row.append("{:.4f}".format(P_success[ep0][0]))
#   print("{:.2f}".format(ep0), '|',row)
#     #print('e0 =',"{:.2f}".format(ep0),'t1 =',P_success[ep0][1],'t2 =',P_success[ep0][2],'t3=',P_success[ep0][3],'P_c =',"{:.4f}".format(P_success[ep0][0]))
#   #print("--- %s seconds ---" % (time.time() - start_time))
