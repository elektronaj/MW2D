################################
# winner.py -- Winner Computation
#

from random    	     import *
from sys             import *
from itertools       import *
from math            import *
from copy            import copy
from random          import choice
from sets            import Set
import sys
import os

from core import *





# import rule packages
for file in os.listdir("."):
    if file.startswith("rule_") and file.endswith(".py"):
        debug("RULES: %s" % file[:-3])
        exec("from %s import *" % (file[:-3]) )


# irrespectively, import core rule packages
from rule_weakly_separable import *
from rule_proportional     import *


try:
  import numpy as np
  import ilp
except:
  debug( "No numeric libraries! Do not use ILP" )



# read in the data in our format
# m n  (number of candidates and voters)
# m candidate names
# ...
# pref1  (n preference orders)
# ...

# return (m,n,V)
def readData( f, k ):
  V = []
  C = []
  lines = f.readlines()
  (m, n) = lines[0].split();
  m = int(m)
  n = int(n)

  print m, n, k

  for l in lines[1:m+1]:
    s = l.rstrip()
    C += [s]
    print s


  for l in lines[m+1:m+n+1]:
    print l.rstrip()
    s = l.split()[0:m]
    s = [int(x) for x in s]
    V += [s]

  return (m,n,C,V)


#
# print winners
#

def printWinners( W, C, k ):
  debug( "printwinners" )
  for i in W:
    print C[i]















if __name__ == "__main__":

  data_in  = stdin
  data_out = stdout

  seed()
 
  R = kborda
  k = 1

  if( len(argv) >= 2 and argv[1].endswith("help")):
    print "This script computes election results"
    print
    print "Invocation:"
    print "  python winner.py rule k <ordinal_election.out"
    print
    print "Available rules:"
    for (rule, description) in RULES:
      l = 10
      print "%s - %s" % (rule+" "*(l-len(rule)), description)
    exit()


  if( len(argv) >= 2 ):
      R = eval(argv[1])

  if( len(argv) >= 3 ):
      k = int(argv[2])

  (m,n,C,V) = readData( data_in, k )

  W = R(V,k)

  printWinners( W, C, k )


