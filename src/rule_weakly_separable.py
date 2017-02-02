#
# weakly separable rules
#

from core import *
from random import random


##############################################################################

 #    #  ######    ##    #    #  #        #   #
 #    #  #        #  #   #   #   #         # #
 #    #  #####   #    #  ####    #          #
 # ## #  #       ######  #  #    #          #
 ##  ##  #       #    #  #   #   #          #
 #    #  ######  #    #  #    #  ######     #


  ####   ######  #####     ##    #####     ##    #####   #       ######
 #       #       #    #   #  #   #    #   #  #   #    #  #       #
  ####   #####   #    #  #    #  #    #  #    #  #####   #       #####
      #  #       #####   ######  #####   ######  #    #  #       #
 #    #  #       #       #    #  #   #   #    #  #    #  #       #
  ####   ######  #       #    #  #    #  #    #  #####   ######  ######




def weaklySeparable( V, k, scoring_vector ):
  m = len( V[0] )
  n = len( V )

  score = [[i,0.1*random()] for i in range(m)]
  for v in V:
    for i in range(m):
      score[v[i]][1] += scoring_vector[i]

  score = sorted(score, key = negsecond )[0:k]
  debug( "print score" )
  debug( score )
  winner = [ s[0] for s in score ]
  return winner


RULES += [("bloc", "the Bloc rule (k-Approval)")]
def bloc( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, ([1]*k)+([0]*(m-k)) )

RULES += [("kborda", "the k-Borda rule")]
def kborda( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, [m-i-1 for i in range(m)] )

def concave_kborda( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, [sqrt(m-i-1) for i in range(m)] )

def convex_kborda( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, [(m-i-1)**2 for i in range(m)] )


RULES += [("sntv", "the SNTV rule (k-Plurality)")]
def sntv( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, [1]+[0]*(m-1) )




def tapproval( V, k, t ):
  m = len( V[0] )
  return weaklySeparable( V, k, ([1]*t)+([0]*(m-t)) )

for i in range(1,201):
  text = "approval_%d = lambda x,y: tapproval( x, y, %d )" % (i,i)
  exec( text )
