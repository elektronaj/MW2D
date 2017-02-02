import sys
from random    	     import *
from sys             import *
from itertools       import *
from math            import *
from copy            import copy
from random          import choice
from sets            import Set


try:
  import numpy as np
  import ilp
except:
  debug( "No numeric libraries! Do not use ILP" )



# core components, to be imported by all rule implementations


# directory of rules (each entry is (fucntion_name, description)
# when implementig a rule, precede the function by adding an
# entry to the RULES list
RULES = []  


debug_on = True


def debug(s):
  if( debug_on ):
    print >>sys.stderr, s



def negsecond( x ):
  return -x[1]



eps = 0.00001 # max error allowed
def lambertw(x): # Lambert W function using Newton's method
  w = x
  while True:
    ew = exp(w)
    wNew = w - (w * ew - x) / (w * ew + ew)
    if abs(w - wNew) <= eps: break
    w = wNew
  return w
