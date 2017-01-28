'''
Python Multiwinner Package

Converts 2D points to preference orders (by Euclidean distances).
'''


from sys import *
from itertools import *


'''
Functions for preparing the preference profile.
'''


def dist( x, y ):
  '''
  Computes Euclidean distance between two 2D points.
  '''  
  return (sum( [ (x[i]-y[i])**2 for i in range(len(x))] ))**(0.5)


def computeDist( v, C ):
  '''
  Computes the distances of voter v from the candidates in set C.
  Outputs a list of the format (i,d) where i is the candidate name and d is the distance.
  '''
  m = len(C)
  d = [ (j, dist(v, C[j])) for j in range(m) ]
  return d


def second( x ):
  '''
  Returns the second element in an array.
  '''    
  return x[1]


def preferenceOrders( C, V ):
  '''
  Returns preference orders.
  '''
  P = []
  for v in V:
    v_dist = computeDist( v, C )
    v_sorted = sorted( v_dist, key = second )
    v_order = [ cand for (cand, dis) in v_sorted]
    P += [v_order]
  return P


def printPrefOrders( C, V, P ):
  '''
  Prints preference orders
  m n (number of candidates and voters)
  m lines with candidate names (number position)
  n lines with preference orders (followed by positions)
  '''
  m = len(C)
  n = len(V)
  print m,n
  
  for i in range(len(C)):
    print i, "  ", C[i][0], C[i][1], C[i][2]

  for i in range(len(P)):
    print " ".join( [str(p) for p in P[i]]), "  ", V[i][0], V[i][1]


def readData( f ):
  '''
  Reads-in the data in our format
  m n  (number of candidates and voters)
  x  y (m candidates in m lines)
  ...
  x  y (n voters in n lines)
  ...

  Returns (n,k,d,F,X)
  '''
  P = []
  C = []
  lines = f.readlines()
  (m, n) = lines[0].split();
  m = int(m)
  n = int(n)

  for l in lines[1:m+1]:
    (x,y,p) = l.split()
    C += [(float(x), float(y), p)]

  for l in lines[m+1:m+n+1]:
    (x,y, ignored) = l.split()
    P += [(float(x), float(y))]
    
  return (m,n,C,P)


'''
Main.
'''


data_in  = stdin
data_out = stdout

(m,n,C,V) = readData( data_in )

P = preferenceOrders( C, V )
printPrefOrders( C, V, P )