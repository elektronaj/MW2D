'''
Python Multiwinner Package

Solves winner determination for some multiwinner voting rules.
'''


from random import *
from sys import *
from itertools import *
from math import *
from copy import copy
from random import choice
from sets import Set
import sys


debug_on = True


def debug(s):
  if( debug_on ):
    print >>sys.stderr, s


try:
  import numpy as np
  import ilp
except:
  debug( "No numeric libraries! Do not use ILP" )


eps = 0.00001 # max error allowed


def lambertw(x):
  '''
  Computes Lambert's W function using Newton's method.
  '''
  w = x
  while True:
    ew = exp(w)
    wNew = w - (w * ew - x) / (w * ew + ew)
    if abs(w - wNew) <= eps: break
    w = wNew
  return w


def readData( f, k ):
  '''
  Reads data in our format.
  m n  (number of candidates and voters)
  m candidate names
  ...
  pref1  (n preference orders)
  ...

  Returns (m,n,V)
  '''
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


def printWinners( W, C, k ):
  '''
  Prints winners.
  '''
  debug( "printwinners" )
  for i in W:
    print C[i]


def negsecond( x ):
  return -x[1]


'''
Winner computations.
'''


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

##############################################################################


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


def bloc( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, ([1]*k)+([0]*(m-k)) )


def kborda( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, [m-i-1 for i in range(m)] )


def concave_kborda( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, [sqrt(m-i-1) for i in range(m)] )


def convex_kborda( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, [(m-i-1)**2 for i in range(m)] )


def sntv( V, k ):
  m = len( V[0] )
  return weaklySeparable( V, k, [1]+[0]*(m-1) )


#########################################################################

 #####      #     ####    #####  #####      #     ####    #####   ####
 #    #     #    #          #    #    #     #    #    #     #    #
 #    #     #     ####      #    #    #     #    #          #     ####
 #    #     #         #     #    #####      #    #          #         #
 #    #     #    #    #     #    #   #      #    #    #     #    #    #
 #####      #     ####      #    #    #     #     ####      #     ####

#########################################################################


def permute_rand(l):
  for i in range(1, len(l)):
    j = randint(0, i)
    l[i], l[j] = l[j], l[i]


def borda_winner( V ):
  m = len( V[0] )
  n = len( V )
  scores = {}
  for c in V[0]:
    scores[c] = 0
  for v in V:
    for i in range(m):
      scores[v[i]] += m-i-1
  inverse = [(value, key) for key, value in scores.items()]
  
  return max(inverse)[1]


def plurality_winner( V ):
  m = len( V[0] )
  n = len( V )
  scores = {}
  for c in V[0]:
    scores[c] = 0
  for v in V:
    scores[v[0]] += 1
  inverse = [(value, key) for key, value in scores.items()]
  
  return max(inverse)[1]


def districtswinner( V, k, single_winner ):
  '''
  Computes district-based FPTP with random districts of the same size.
  '''
  m = len( V[0] )
  n = len( V )
  permute_rand(V)
  candidates = V[0][:]
  permute_rand(candidates)

  vot_groups = [[] for i in range(k)]
  cand_groups = [[] for i in range(k)]
  for i in range(n):
    vot_groups[i % k].append(V[i])
  for i in range(m):
    cand_groups[i % k].append(candidates[i])
  for i in range(k):
    VG, CG = vot_groups[i], cand_groups[i]
    for j in range(len(VG)):
      VG[j] = [c for c in VG[j] if c in CG]

  return [single_winner(G) for G in vot_groups]


def bordadistricts( V, k ):
  return districtswinner(V, k, borda_winner)


def pluralitydistricts( V, k ):
  return districtswinner(V, k, plurality_winner)


####################################

  ####    #####  #    #
 #          #    #    #
  ####      #    #    #
      #     #    #    #
 #    #     #     #  #
  ####      #      ##

####################################


def pluralityScores( V, m ):
  '''
  Computes plurality scores.
  '''
  score = [[i,0.1*random()] for i in range(m)]
  for v in V:
    if( len(v) > 0 ):
      score[v[0]][1] += 1

  score = sorted(score, key = negsecond )
  return score


def removeCandidateFromVote( v, c ):
  '''
  Removes a given candidate from the whole profile.
  '''
  return [ cand for cand in v if cand != c ]


def removeCandidate( V, c ):
  return [ removeCandidateFromVote( v, c ) for v in V]


def removeVoters( V, c, q, count ):
  '''
  Removes q voters that rank c first, knowing there is count of them (chosen randomly).
  '''
  V     = sorted( V , key = lambda v : 0 if v[0]==c else 1 )
  Vc    = V[:count]
  Vrest = V[count:]
  shuffle(Vc) 
  Vc    = Vc[q:]
  return Vc + Vrest


def noQuota( V, score ):
  '''
  Removes a random candidate due to small quota.
  '''
  S = dict( score )
  v = copy(V[0])
  v = sorted( v, key = lambda x : S[x] )
  low = S[v[0]]
  v = [ cand for cand in v if S[cand] <= low ]
  c = choice( v )
  return removeCandidate( V, c )


def chooseOutOfW( m, W ):
  M = [ i for i in range(m)]
  for c in W:
    M.remove(c)
  return choice(M)
  

def stv( V, k ):
  m = len( V[0] )
  n = len( V )
  quota = int( floor( float(n)/float(k+1) ) + 1 )

#  if( quota * k > n ):
#    quota -= 1

  W = []
  i = 0

  debug( "STV" )
  debug( "quota = " + str(quota) )
  while len(W) < k:
    i+=1
    debug( "W = " + str( W ) + " , i = " + str(i))

    # the case when we removed all candidates
    if( i > m ):
      W += [ chooseOutOfW( m, W ) ]
      continue
   
    # regular STV
    score = pluralityScores( V, m )
    top = score[0]
    # does the highest plurality-score guy meet the quota?
    if( top[1] >= quota ):
      W += [ top[0] ]
      V = removeVoters( V, top[0], quota, int(top[1]) )
      V = removeCandidate( V, top[0] )
    else: 
      V = noQuota( V, score )

#  debug( "W = " + str( W ) )
#  debug( "end STV" )
  return W



########################################################################

  ####   #####   ######  ######  #####    #   #      #####   #####
 #    #  #    #  #       #       #    #    # #      #     # #     #
 #       #    #  #####   #####   #    #     #       #       #
 #  ###  #####   #       #       #    #     #       #       #
 #    #  #   #   #       #       #    #     #       #     # #     #
  ####   #    #  ######  ######  #####      #        #####   #####

########################################################################


def convertVote( v ):
  m = len(v)
  s = [0]*m
  for i in range(m):
    s[v[i]]=i
  return s


def convertProfile( V ):
  return [ convertVote(v) for v in V ]


def ccScoreProfile( P, c, N ):
  S = [ max( m-P[i][c]-1, N[i]) for i in range(len(P)) ]
  return sum( S )


def greedyCC( V, k ):
  debug("CC 3 (rnd)")

  m = len( V[0] )
  n = len( V )
  C = range(m)
  S = convertProfile( V )
  N = [0]*n
  W = []

  # compute each additional member of the committee
  print >>sys.stderr, "CC3 computing"
  for i in range(k):
    print >>sys.stderr, "Greedy CC ", i
    best_score = -1
    best_candidate_set = []
    for i in C:
      s = ccScoreProfile( S, i, N )
#      debug( str(i)+" --> "+str(s) )
      if( s > best_score ):
        best_score = s
        best_candidate_set = [i]
      elif( s == best_score ):
        best_candidate_set += [i]

    best_candidate = choice( best_candidate_set )

    W += [ best_candidate ]  
    C.remove( best_candidate )
    N = [ max( m-S[i][best_candidate]-1, N[i]) for i in range(len(S)) ]
  return W


#######################################################################

######                                                   #####   #####
#     #    ##    #    #  ######  #    #    ##    ###### #     # #     #
#     #   #  #   ##   #      #   #    #   #  #   #      #       #
######   #    #  # #  #     #    ######  #    #  #####  #       #
#     #  ######  #  # #    #     #    #  ######  #      #       #
#     #  #    #  #   ##   #      #    #  #    #  #      #     # #     #
######   #    #  #    #  ######  #    #  #    #  #       #####   #####

#######################################################################


BINOMIALS = {}


def binomial(x, y):
    try:
#        debug( "BBB %d %d" %(x,y) )
        if( x in BINOMIALS ):
          if( y in BINOMIALS[x] ):
            return BINOMIALS[x][y]
        else:
          BINOMIALS[x] = {}
          

        binom = factorial(x) // factorial(y) // factorial(x - y)
        BINOMIALS[x][y] = binom
    except ValueError:
        binom = 1
    return binom


# Banzhaf-CC score for an election where we have
#   m   - number of candidates
#   i   - position of our guy in the vote
#   j   - position of the best committee member fixed in the vote
#   t   - number of committee members already chosen
#   k   - committee size
def BanzhafScorePerVote( m, i, j, t, k ):
  if( j < i ):
    return 0

  score = 0
 
# we want to compute the loop below, but faster 
#
#  for ell in range(i+1,j):
#    if( m-(ell+1)-t-1 >= k-1-t-1 ):
#      score += (ell-i) * binomial(m-(ell+1)-t-1, k-1-t-1)

  B = binomial(m-(j+1)-t-1, k-1-t-1)  
  for ell in range(j-1,i,-1):
    if( m-(ell+1)-t-1 > k-1-t-1 ):
      B = B * (m-(ell+1)-t-1) / (m-(ell+1)-k+1)
      score += (ell-i) * B
    elif( m-(ell+1)-t-1 == k-1-t-1 ):
      score += (ell-i)


  # if there is no other candidate in the committee
  if( j == m ):
    if( m-(i+1) >= k-1 ):
      score += (m-(i+1))*binomial( m-(i+1), k-1 )
  # if there is some better guy
  else:
    if( m-(j+1)-t >= k-1-t ):
      score += (j-i)*binomial( m-(j+1)-t, k-1-t )

  return score


def BanzhafCC( V, k ):
  debug("BanzhafCC")

  m = len( V[0] )
  n = len( V )
  C = range(m)
  S = convertProfile( V )
  N = [0]*n
  B = [m]*n
  W = []

  # compute each additional member of the committee
  print >>sys.stderr, "BanzhafCC computing"
  for i in range(k):
    debug( "Banzhaf CC " + str(i) )
    best_score = -1
    best_candidate_set = []
    for c in C:
      score = 0
      for voter in range(n):
        score += BanzhafScorePerVote( m, S[voter][c], B[voter], i, k )
#      debug( str(c) + " -> " + str(score) )

      if( score > best_score ):
        best_score = score
        best_candidate_set = [c]
      elif( score == best_score ):
        best_candidate_set += [c]

    best_candidate = choice( best_candidate_set )
    print >>sys.stderr, best_candidate_set


    W += [ best_candidate ]  
    C.remove( best_candidate )
    B = [ min( S[i][best_candidate], B[i]) for i in range(n) ]
  return W



#######################################################################

                                                ####### #     #    #
  ####   #####   ######  ######  #####    #   # #     # #  #  #   # #
 #    #  #    #  #       #       #    #    # #  #     # #  #  #  #   #
 #       #    #  #####   #####   #    #     #   #     # #  #  # #     #
 #  ###  #####   #       #       #    #     #   #     # #  #  # #######
 #    #  #   #   #       #       #    #     #   #     # #  #  # #     #
  ####   #    #  ######  ######  #####      #   #######  ## ##  #     #

#######################################################################


# comput the score of committee S under OWA x score
def owaScore( S, v, OWA, score ):
  m = len(v)
  s = 0.0
  t = 0
  for i in range(m):
    if v[i] in S:
      s += score[i]*OWA[t]
      t += 1
  return s


def owaScoreProfile( S, V, OWA, score ):
  return sum( [ owaScore( S, v, OWA, score ) for v in V ] )


def greedyOWA( V, k, OWA, score ):
  debug("greedyOWA")

  m = len( V[0] )
  n = len( V )
  C = range(m)
  W = []

  # compute each additional member of the committee
  print >>sys.stderr, "OWA computing"
  print >>sys.stderr, "OWA =", OWA
  for i in range(k):
    print >>sys.stderr, "Greedy OWA ", i
    best_score = -1
    best_candidate_set = []
    for i in C:
      s = owaScoreProfile( W+[i], V, OWA, score )
#      debug( str(i)+" --> "+str(s) )
      if( s > best_score ):
        best_score = s
        best_candidate_set = [i]
      elif( s == best_score ):
        best_candidate_set += [i]

    best_candidate = choice( best_candidate_set )

    W += [ best_candidate ]  
    C.remove( best_candidate )
  return W


def greedyOWA_borda( V, k, OWA ):
  m = len( V[0] )
  return greedyOWA( V, k, OWA, [m-i-1 for i in range(m)] ) 


def greedyOWA_kborda( V, k ):
  m = len( V[0] )
  return greedyOWA( V, k, [1]*k, [m-i-1 for i in range(m)] ) 


def greedyOWA_cc( V, k ):
  m = len( V[0] )
  return greedyOWA( V, k, [1]+[0]*(k-1), [m-i-1 for i in range(m)] ) 


def greedyOWA_bordaPAV( V, k ):
  m = len( V[0] )
#  return greedyOWA( V, k, [1.0/(i+1) for i in range(k)], [m-i-1 for i in range(m)] ) 
  return greedyOWA( V, k, [1.0/(i+1) for i in range(k)], [m-i-1 for i in range(m)] ) 


def greedyOWA_topkPAV( V, k ):
  m = len( V[0] )
  return greedyOWA( V, k, [1.0/(i+1) for i in range(k)], ([1]*k)+([0]*(m-k)) ) 


##################################################################################

######                                                  ####### #     #    #
#     #    ##    #    #  ######  #    #    ##    ###### #     # #  #  #   # #
#     #   #  #   ##   #      #   #    #   #  #   #      #     # #  #  #  #   #
######   #    #  # #  #     #    ######  #    #  #####  #     # #  #  # #     #
#     #  ######  #  # #    #     #    #  ######  #      #     # #  #  # #######
#     #  #    #  #   ##   #      #    #  #    #  #      #     # #  #  # #     #
######   #    #  #    #  ######  #    #  #    #  #      #######  ## ##  #     #

##################################################################################


# compute the banzhaf value of c (assuming S is already in, k is the committee size)
def BanzhafOWAScore( k, c, S, v, pos, OWA, score ):
  m = len(v)
  f = len(S)  # number of candidates fixed in the committee
#  pos = {}   # position of the candidate
  Worse = []  # set of candidates on positions worse or equal to c

  DELTA = 0 # marginal contribution of c

#  NOW THESE VALUES ARE COMPUTED ONCE FOR THE WHOLE ELECTION
#  # compute positions
#  for i in range(m):
#    pos[v[i]] = i

  # compute candidates ranked below c
  for s in S:
    if pos[s] > pos[c]:
      Worse += [s]
  Worse += [c]

  # consider each candidate ranked below c
  for s in Worse:
    before = sum( [int( pos[x] < pos[s]) for x in S] )  # number of candidates from S ranked before of s (not inluding c)
    after  = sum( [int( pos[x] > pos[s]) for x in S] )  # number of candidates from S ranked after s

#    after  = f - before                                 # number of candidates from S ranked after s (not including c)

    if( s != c ):
      before += 1 # include c among the candidates before c

    pos_before = pos[s] - before        # number of free positions before s
    pos_after  = (m - pos[s]-1) - after # number of free positions after s

    # iterate over the number of committee members ranked before c
    for t in range(before, k):

      t_after = k - (t+1)  # number of guys to be placed after

      # if it is imspossible to have t guys ahaead of s then skip
      if( t-before > pos_before ):
        continue 

      # if there is not enough committee members to be put after, then skip
      if( t_after < after ):
        continue
      # if there is not enough positions after then skip
      if( t_after - after > pos_after ):
        continue

      # compute the number of coalitions that have t guys before, t_after guys after      
      C = binomial( pos_before, t-before) * binomial( pos_after, t_after - after )

      # compute the gain s has
      if( s == c ):
        gain = OWA[t]*score[pos[s]]
      else:
        gain = -OWA[t-1]*score[pos[s]] + OWA[t]*score[pos[s]]

      DELTA += gain*C

  return DELTA


def BanzhafOWAScoreProfile(k, c, S, V, POS, OWA, score ):
  return sum( [ BanzhafOWAScore( k, c, S, V[i], POS[i], OWA, score ) for i in range(len(V)) ] )


def BanzhafOWA( V, k, OWA, score ):
  debug("BanzhafOWA")

  m = len( V[0] )
  n = len( V )
  C = range(m)
  W = []

  # compute profile of positions
  POS = convertProfile( V )

  # compute each additional member of the committee
  debug( "Banzhaf OWA computing" )
  debug( "OWA = " + str(OWA) )
  for i in range(k):
    debug( "Banzhaf OWA %d" % i )
    prev_score = -1
    best_score = -1
    prev = []
    best_candidate_set = []
    for i in C:
      s = BanzhafOWAScoreProfile( k, i, W, V, POS, OWA, score )
#      debug( str(i)+" --> "+str(s) )
      if( s > best_score ):
        prev_score = best_score
        prev       = best_candidate_set
        best_score = s
        best_candidate_set = [i]
      elif( s == best_score ):
        best_candidate_set += [i]

    best_candidate = choice( best_candidate_set )
    print >>sys.stderr, best_candidate_set
    print >>sys.stderr, [best_score, prev_score, prev]

    W += [ best_candidate ]  
    C.remove( best_candidate )
  return W


def BanzhafOWA_borda( V, k, OWA ):
  m = len( V[0] )
  return BanzhafOWA( V, k, OWA, [m-i-1 for i in range(m)] ) 


def BanzhafOWA_kborda( V, k ):
  m = len( V[0] )
  return BanzhafOWA( V, k, [1]*k, [m-i-1 for i in range(m)] ) 


def BanzhafOWA_cc( V, k ):
  m = len( V[0] )
  return BanzhafOWA( V, k, [1]+[0]*(k-1), [m-i-1 for i in range(m)] ) 


def BanzhafOWA_bordaPAV( V, k ):
  m = len( V[0] )
#  return greedyOWA( V, k, [1.0/(i+1) for i in range(k)], [m-i-1 for i in range(m)] ) 
  return BanzhafOWA( V, k, [1.0/(i+1) for i in range(k)], [m-i-1 for i in range(m)] ) 


def BanzhafOWA_topkPAV( V, k ):
  m = len( V[0] )
  return BanzhafOWA( V, k, [1.0/(i+1) for i in range(k)], ([1]*k)+([0]*(m-k)) ) 


##################################################################################

 ######  #       #               #####           #####   #    #  #       ######
 #       #       #               #    #          #    #  #    #  #       #
 #####   #       #               #    #  #####   #    #  #    #  #       #####
 #       #       #               #####           #####   #    #  #       #
 #       #       #               #               #   #   #    #  #       #
 ######  ######  ###### #######  #               #    #   ####   ######  ######

##################################################################################


def ellpScoreProfile( P, N, c, m, scoring_vector, p ):
  n = len( P )
  S = [ (  N[i] + ( scoring_vector[P[i][c]] )**p   )**(1.0/p)  for i in range(n) ]

  return float(sum( S ))


# scoring_vector - m-dimensial scoring protocol to use
# p              - the exponent to use
def greedyEllpRule( V, k, scoring_vector, p ):
  debug("greedyEllpRule")

  m = len( V[0] )
  n = len( V )
  C = range(m)
  S = convertProfile( V )
  W = []
  N = [0]*n   # scores of the committee so far

  debug( "n = %d, m = %d" % (n,m) )

  # compute each additional member of the committee
  for i in range(k):   
    best_score = -1
    best_candidate_set = []
    for i in C:
      s = ellpScoreProfile( S, N, i, m, scoring_vector, p )
      if( s > best_score ):
        best_score = s
        best_candidate_set = [i]
      elif( s == best_score ):
        best_candidate_set += [i]

    debug( "BEST CANDs = %d " % len( best_candidate_set ) )
    best_candidate = choice( best_candidate_set )
    W += [ best_candidate ]  
    C.remove( best_candidate )
    for i in range(n):
      N[i] += ( scoring_vector[S[i][best_candidate]]  )**p
    sssum = sum(N)
    debug( "sum = %d" % sssum )
  return W


def greedyEllpBorda( V, k, p ):
  m = len(V[0])
  return greedyEllpRule( V, k, [m-i-1 for i in range(m)], p )


def greedyEllpBloc( V, k, p ):
  m = len(V[0])
  return greedyEllpRule( V, k, ([1]*k)+([0]*(m-k)), p )


def kborda_score( V, W ):
  m = len( V[0] )
  s = 0
  for v in V:
    for i in range(m):
      if( v[i] in W ):
        s += m-i-1
  return s


def cc_score( V, W ):
  m = len( V[0] )
  s = 0
  for v in V:
    for i in range(m):
      if( v[i] in W ):
        s += m-i-1
        break
  return s


def ellpborda_score( V, W, p ):
  m = len( V[0] )
  s = 0
  for v in V:
    ls = 0.0
    for i in range(m):
      if( v[i] in W ):
        ls += (m-i-1)**p
    s += ls**(1.0/p)
  return s


def metropolis( V, k, score ):
  m = len( V[0] )
  n = len( V )
  C = range(m)  
  best  = 0
  bestW = []

  debug("Metropolis!")

  STEPS  = 1500
  accept = 0.01

  W = sample( C, k )
  notW = {}
  for c in C:
    notW[c] = 1
  for c in W:
    del notW[c] 

  bestW   = W
  best    = score( V, W )
  current = best

  debug( "score = %d" % best )

  for step in range(STEPS):
    i = choice( range(k) )
    newW = W[:i]+ [ choice(notW.keys()) ] +W[i+1:]
    s = score( V, newW )
    
    if( s > best ):
      (best, bestW) = (s, newW)

    if( s > current ):
      W = newW
      current = s
    elif( random() < accept ):
      W = newW
      current = s

    debug( "%d %d (%d)" % (step, best, current) )

  return bestW
  

def ell_p_metropolis( V, k, p ):
  m = len( V[0] )
  n = len( V )
  C = range(m)  

  S = convertProfile( V )
  N = [0]*n   # scores of the committee so far

  best  = 0
  bestW = []

  debug("Metropolis!")

#  STEPS  = 10000
#  accept = 0.001

  STEPS = 2001
  accept = 0.02

  W = sample( C, k )
  notW = {}
  for c in C:
    notW[c] = 1
  for c in W:
    del notW[c] 

  # compute the score and N
  best = 0
  for i in range(n):
    for c in W:
      N[i] += (m-S[i][c]-1)**p
    best += N[i]**(1.0/p)


  bestW   = W
  current = best

  debug( "score = %d" % best )

  for step in range(STEPS):

    accept *= 0.999

    i = choice( range(k) )
    c = W[i]
    new_c = choice(notW.keys())
    newW = W[:i]+ [ new_c ] +W[i+1:]
    
    # specific computation of ell_p-Borda score
    s = 0
    NN = [0]*n
    for i in range(n):
      NN[i] = N[i] - (m-S[i][c]-1)**p  + (m-S[i][new_c]-1)**p
      s += NN[i]**(1.0/p)

    
    if( s > best ):
      (best, bestW) = (s, newW)

    if( s > current ) or ( random() < accept ):
      W = newW
      current = s
      N = NN
      notW[c] = 1
      del notW[new_c]

    if step % 1000 == 0:
      debug( "%d %d (%d)  accept = %f" % (step, best, current, accept) )

  return bestW



# running several tries of the Metropolis algorithm
def ell_p_manyMetropolis( V, k, p ):
  TRIES = 3
  bestW = []
  best  = 0

  for i in range(TRIES):
    W = ell_p_metropolis( V, k, p )
    s = ellpborda_score( V, W, p )
    if( s > best ):
      (best, bestW) = (s,W)

  debug ("#### METROPOLIS BEST = %d" % best )
  return W


def owa_metropolis( V, k, OWA, score ):
  m = len( V[0] )
  n = len( V )
  C = range(m)  


  best  = 0
  bestW = []

  debug("OWA Metropolis!")

#  STEPS  = 10000
#  accept = 0.001

  STEPS = 2001
  accept = 0.02

  W = sample( C, k )
  notW = {}
  for c in C:
    notW[c] = 1
  for c in W:
    del notW[c] 

  # compute the score of W
  best = owaScoreProfile( W, V, OWA, score )
  bestW   = W
  current = best

  debug( "score = %d" % best )

  for step in range(STEPS):

    accept *= 0.999

    i = choice( range(k) )
    c = W[i]
    new_c = choice(notW.keys())
    newW = W[:i]+ [ new_c ] +W[i+1:]
    
    # specific computation of ell_p-Borda score
    s = owaScoreProfile( newW, V, OWA, score )
    
    if( s > best ):
      (best, bestW) = (s, newW)

    if( s > current ) or ( random() < accept ):
      W = newW
      current = s
      notW[c] = 1
      del notW[new_c]

    if step % 1000 == 0:
      debug( "%d %d (%d)  accept = %f" % (step, best, current, accept) )

  return bestW


# running several tries of the Metropolis algorithm
def owa_manyMetropolis( V, k, OWA, score ):
  TRIES = 1
  bestW = []
  best  = 0

  for i in range(TRIES):
    W = owa_metropolis( V, k, OWA, score )
    s = owaScoreProfile( W, V, OWA, score)
    if( s > best ):
      (best, bestW) = (s,W)

  debug ("#### METROPOLIS BEST = %d" % best )
  return W


def metropolis_OWA_borda( V, k, OWA ):
  m = len( V[0] )
  return owa_manyMetropolis( V, k, OWA, [m-i-1 for i in range(m)] ) 


##########################################################

 #    #   ####   #    #  #####    ####   ######
 ##  ##  #    #  ##   #  #    #  #    #  #
 # ## #  #    #  # #  #  #    #  #    #  #####
 #    #  #    #  #  # #  #####   #    #  #
 #    #  #    #  #   ##  #   #   #    #  #
 #    #   ####   #    #  #    #   ####   ######

##########################################################


# # compute monroe score and voters
# # P - profile
# # c - candidate
# # nk- number of voters we seek
# def gmScore( convV, P, c, nk ):
#   n = len( P )
#   S = [ [ i, ccScore(P[i], [c]) ] for i in range( n ) ]   # pairs [i, score-of-c-in-i]
#   S = sorted(S, key=negsecond )
#   S = S[0:nk]
#   score = sum( [x[1] for x in S] )
#   votes = [x[0] for x in S]
#   return (score, votes)

# compute monroe score and voters
# P - profile
# c - candidate
# nk- number of voters we seek
def gmScore( convV, c, nk ):
  n = len( convV )
  S = [ [ i, m-convV[i][c]-1 ] for i in range( n ) if convV[i]!=None]   # pairs [i, score-of-c-in-i]
  S = sorted(S, key=negsecond )
  S = S[0:nk]
  score = sum( [x[1] for x in S] )
  votes = [x[0] for x in S]
  return (score, votes)


def greedyMonroe( P, k ):
  m = len( P[0] )
  C = range(m)
  W = []
  V = list(P)
  shuffle( V )
  convV = convertProfile ( V )
  n = len( V )
  kk= k

  # compute each additional member of the committee
  for i in range(k):
    print >>sys.stderr, "Greedy Monroe ", i
    nk = int(n/kk)
    best_score = -1
    best_candidate = -1
    best_votes = []
    best_cnvs = []     # set of best candidate-voters pairs
    for i in C:
      (s,v) = gmScore( convV, i, nk  )
#      debug( "nk = %d, len(v) = %d" % (nk, len(v)) )
      if( s > best_score ):
        best_score = s
        best_cnvs = [[i,v]]     # store candidate i and votes v
      elif( s == best_score ):
        best_cnvs += [[i,v]]    # store candidate i and votes v

    best_cnv = choice( best_cnvs )
    best_candidate = best_cnv[0]
    best_votes = best_cnv[1]

    W += [ best_candidate ]  
    C.remove( best_candidate )
    for i in range(len(V)):
      if( i in best_votes ):
        V[i] = None
        convV[i] = None
  
    n  -= nk
    kk -= 1
  
  return W


##########################################################

   #                            ######
  # #    #        ####    ####  #     #
 #   #   #       #    #  #    # #     #
#     #  #       #       #    # ######
#######  #       #  ###  #    # #
#     #  #       #    #  #    # #
#     #  ######   ####    ####  #

##########################################################


def algoP_threshold(P, k, threshold):
  m = len( P[0] )
  n = len(P)
  
  VV = []
  for v in P:
    VV += [ v[:threshold] ]
  scores = {}
  for c in range(m):
    scores[c] = 0.1*random()

  for v in VV:
    for c in v:
      scores[c] += 1

  winners = []
  for i in range(k):
#    debug( "scores = " + str(scores) )
    inverse = [(value, key) for key, value in scores.items()]
    best_c = max(inverse)[1]
#    best_s = max(inverse)[0]
#    debug( "algoP: chose %d with score %f" % (best_c, best_s) )
    for i in range(len(VV)):
      if best_c in VV[i]:
        for c in VV[i]:
          scores[c] -= 1
        VV[i] = []

    del scores[best_c]
    winners.append(best_c)
  return winners




def algoP( P, k ):
  m = len( P[0] )
  threshold = int((m * lambertw(k)) / (k))
  debug( "Threshold = %d " % (threshold) )
  return algoP_threshold( P, k, threshold )



def ccScore_simple( p, C ):
  if( p == None ):
    return -1
  m = len(p)
  for i in range(m):
    if p[i] in C:
      return m-i-1   


def ccScoreProfile_simple( P, C ):
  S = [ccScore_simple( p, C ) for p in P ]
  return sum( S )



def algoP_ranging( P, k ):
  m = len( P[0] )
  n = len( P )

  best = -1
  best_t = -1
  winners_set = []

  for threshold in range(1,  int((m * lambertw(k)) / (k))+1 ):
    winners_try = algoP_threshold( P, k, threshold )
    current = ccScoreProfile_simple( P, winners_try )
    if( current > best ):
      winners_set = [ winners_try ]
      best = current
      best_t = [threshold]
    elif( current == best ):
      winners_set += [winners_try]
      best_t += [threshold]

#    debug("algoP ranging: t=%d,  score = %d" % (threshold, current) )

  debug("BEST algoP ranging: t=%s,  score = %d" % (str(best_t), best) )
  winners = choice( winners_set )
  return winners


rangingCC = algoP_ranging


'''
ILP computations.
'''


# compute CC with ILP
def ccILP( V, k ):
  m = len( V[0] )
  n = len( V )

  print >>sys.stderr, "in ccILP for real"
  # call ILP..
  print >>sys.stderr, "CPLEX START"
  (total_satisfaction, winning_committee) = ilp1.run_ilp(np.array(V), k, np.arange(m - 1, -1, -1))
  print >>sys.stderr, "winning_committee"
  print >>sys.stderr, winning_committee
  debug('well')
  debug(list(winning_committee))
  print >>sys.stderr, "CPLEX END"
  return list(winning_committee)


# compute PAV with ILP
def PAV( V, k ):
  m = len( V[0] )
  n = len( V )

  print >>sys.stderr, "in PAV for real"
  # call ILP..
  print >>sys.stderr, "CPLEX START"
  (total_satisfaction, winning_committee) = ilp1.run_ilp_pav(np.array(V), k, np.arange(m - 1, -1, -1))
  print >>sys.stderr, "winning_committee"
  print >>sys.stderr, winning_committee
  debug('well')
  debug(list(winning_committee))
  print >>sys.stderr, "CPLEX END"
  return list(winning_committee)


# compute OWA rule with ILP
def OWA_borda( V, k, OWA ):
  m = len( V[0] )
  n = len( V )

  print >>sys.stderr, "in PAV for real"
  # call ILP..
  print >>sys.stderr, "CPLEX START"
  (total_satisfaction, winning_committee) = ilp1.run_ilp_OWA(np.array(V), k, OWA, np.arange(m - 1, -1, -1))
  print >>sys.stderr, "winning_committee"
  print >>sys.stderr, winning_committee
  debug('well')
  debug(list(winning_committee))
  print >>sys.stderr, "CPLEX END"
  return list(winning_committee)
  

# compute PAV with ILP
def PAVtopk( V, k ):
  m = len( V[0] )
  n = len( V )

  print >>sys.stderr, "in PAV for real"
  # call ILP..
  print >>sys.stderr, "CPLEX START"
  (total_satisfaction, winning_committee) = ilp1.run_ilp_pavtopk(np.array(V), k, np.arange(m - 1, -1, -1))
  print >>sys.stderr, "winning_committee"
  print >>sys.stderr, winning_committee
  debug('well')
  debug(list(winning_committee))
  print >>sys.stderr, "CPLEX END"
  return list(winning_committee)

 
# compute Monroe with ILP
def monroeILP( V, k ):
  m = len( V[0] )
  n = len( V )

  print >>sys.stderr, "CPLEX START: computing Monroe with ILP"
  (total_satisfaction, winning_committee) = ilp1.run_ilp_monroe(np.array(V), k, np.arange(m - 1, -1, -1))
  print >>sys.stderr, "CPLEX END: Monroe ILP winning_committee"
  print >>sys.stderr, winning_committee
  
  return list(winning_committee)


'''
Main.
''' 


# define the greedy ell_p rules 
for i in range(200):
  text = "greedyEll%d_Borda = lambda x,y: greedyEllpBorda( x, y, %d.0 )" % (i,i)
  exec( text )


for i in range(200):
  text = "greedyEll%d_Bloc = lambda x,y: greedyEllpBloc( x, y, %d.0 )" % (i,i)
  exec( text )


# the same for metropolis
for i in range(200):
  text = "metropEll%d_Borda = lambda x,y: ell_p_manyMetropolis( x, y, %d.0 )" % (i,i)
  exec( text )


# define best_t OWA rules
for i in range(1,50):
  text = "Banzhaf_best_%d_borda = lambda x,y: BanzhafOWA_borda( x, y, ([1]*%d)+([0]*(y-%d)) )" % (i,i,i)
  exec( text )


for i in range(1,50):
  text = "greedy_best_%d_borda = lambda x,y: greedyOWA_borda( x, y, ([1]*%d)+([0]*(y-%d)) )" % (i,i,i)
  exec( text )


for i in range(1,50):
  text = "metropolis_best_%d_borda = lambda x,y: metropolis_OWA_borda( x, y, ([1]*%d)+([0]*(y-%d)) )" % (i,i,i)
  exec( text )


for i in range(1,50):
  text = "OWA_best_%d_borda = lambda x,y: OWA_borda( x, y, ([1]*%d)+([0]*(y-%d)) )" % (i,i,i)
  exec( text )


# define OWA-PAV-powers Borda rules
for t in range(1,20):
  text = "PAV_power_%d_borda = lambda x,y: OWA_borda( x, y, [1.0/((i+1)**(%d)) for i in range(k)] )" % (t,t)
  exec( text )
  text = "greedy_PAV_power_%d_borda = lambda x,y: greedyOWA_borda( x, y, [1.0/((i+1)**(%d)) for i in range(k)] )" % (t,t)
  exec( text )
  text = "metropolis_PAV_power_%d_borda = lambda x,y: metropolis_OWA_borda( x, y, [1.0/((i+1)**(%d)) for i in range(k)] )" % (t,t)
  exec( text )

  text = "PAV_rev_power_%d_borda = lambda x,y: OWA_borda( x, y, [1.0/((i+1)**(1.0/%d)) for i in range(k)] )" % (t,t)
  exec( text )
  text = "greedy_PAV_rev_power_%d_borda = lambda x,y: greedyOWA_borda( x, y, [1.0/((i+1)**(1.0/%d)) for i in range(k)] )" % (t,t)
  exec( text )
  text = "metropolis_PAV_rev_power_%d_borda = lambda x,y: metropolis_OWA_borda( x, y, [1.0/((i+1)**(1.0/%d)) for i in range(k)] )" % (t,t)
  exec( text )


if __name__ == "__main__":
  data_in  = stdin
  data_out = stdout

  seed()
 
  R = kborda
  k = 1
  if( len(argv) >= 2 ):
      R = eval(argv[1])

  if( len(argv) >= 3 ):
      k = int(argv[2])

  (m,n,C,V) = readData( data_in, k )

  W = R(V,k)

  printWinners( W, C, k )