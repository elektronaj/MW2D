################################
# 2d2pref --- converts 2D points to preference orders
#

from sys       import *
from itertools import *


#############################################################
#
# functions for preparing the preference profile


def dist( x, y ):
  return (sum( [ (x[i]-y[i])**2 for i in range(len(x))] ))**(0.5)



# Compute the distances of voter v from the candidates in set C
# outputs a list of the format (i,d) where i is the candidate
# name and d is the distance
#
def computeDist( v, C ):
  m = len(C)
  d = [ (j, dist(v, C[j])) for j in range(m) ]
  return d


def second( x ):
  return x[1]

def preferenceOrders( C, V ):
  P = []
#  print C
  for v in V:
#    print v
    v_dist = computeDist( v, C )
    v_sorted = sorted( v_dist, key = second )
#    print v_sorted
    v_order = [ cand for (cand, dis) in v_sorted]
#    print v_order
    P += [v_order]
  return P




# Print pref orders
# m n (number of candidates and voters)
# m lines with candidate names (number position)
# n lines with preference orders (followed by positions)

def printPrefOrders( C, V, P ):
  m = len(C)
  n = len(V)
  print m,n

  for i in range(len(C)):
    print i, "  ", C[i][0], C[i][1], C[i][2]

  for i in range(len(P)):
    print " ".join( [str(p) for p in P[i]]), "  ", V[i][0], V[i][1]




# read in the data in our format
# m n  (number of candidates and voters)
# x  y (m candidates in m lines)
# ...
# x  y (n voters in n lines)
# ...

# return (n,k,d,F,X)
def readData( f ):
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





# MAIN

if __name__ == "__main__":


  if( len(argv) > 1 ):
    print "This script converts an election in the 2D Euclidean format to a preference-order based one"
    print
    print "Invocation:"
    print "  python 2d2pref.py  <2d_point.in >election.out"
    exit()



  data_in  = stdin
  data_out = stdout

  (m,n,C,V) = readData( data_in )


  P = preferenceOrders( C, V )
  printPrefOrders( C, V, P )


