from core            import *
from random    	     import *
from sys             import *
from itertools       import *
from math            import *
from copy            import copy
from random          import choice
from sets            import Set

####################################

  ####    #####  #    #
 #          #    #    #
  ####      #    #    #
      #     #    #    #
 #    #     #     #  #
  ####      #      ##


RULES += [("stv","single transferable vote")]


# compute plurality scores
def pluralityScores( V, m ):
  score = [[i,0.1*random()] for i in range(m)]
  for v in V:
    if( len(v) > 0 ):
      score[v[0]][1] += 1

  score = sorted(score, key = negsecond )
  return score



# remove a given candidate from the whole
# profile
def removeCandidateFromVote( v, c ):
  return [ cand for cand in v if cand != c ]

def removeCandidate( V, c ):
  return [ removeCandidateFromVote( v, c ) for v in V]


# remove q voters that rank c first, knowing there is count of them
# (chosen randomly)
def removeVoters( V, c, q, count ):
  V     = sorted( V , key = lambda v : 0 if v[0]==c else 1 )
  Vc    = V[:count]
  Vrest = V[count:]
  shuffle(Vc) 
  Vc    = Vc[q:]
  return Vc + Vrest


# noQuota --- remove a random candidate due to small quota
def noQuota( V, score ):
#  debug( "noQuota" )
  S = dict( score )
  v = copy(V[0])
#  debug( S )
#  debug( v )
  v = sorted( v, key = lambda x : S[x] )
#  debug( v )
  low = S[v[0]]
  v = [ cand for cand in v if S[cand] <= low ]
#  debug( v )
  c = choice( v )
#  debug( c )
  return removeCandidate( V, c )


def chooseOutOfW( m, W ):
  M = [ i for i in range(m)]
  for c in W:
    M.remove(c)
  return choice(M)
  



# compute STV
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



RULES += [("greedyCC","greedy approximation of Chamberlin--Courant")]

def convertVote( v ):
  m = len(v)
  s = [0]*m
  for i in range(m):
    s[v[i]]=i
  return s

def convertProfile( V ):
  return [ convertVote(v) for v in V ]



def ccScoreProfile( P, c, N, m ):
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
      s = ccScoreProfile( S, i, N, m )
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






##########################################################

 #    #   ####   #    #  #####    ####   ######
 ##  ##  #    #  ##   #  #    #  #    #  #
 # ## #  #    #  # #  #  #    #  #    #  #####
 #    #  #    #  #  # #  #####   #    #  #
 #    #  #    #  #   ##  #   #   #    #  #
 #    #   ####   #    #  #    #   ####   ######



RULES += [("greedyMonroe","Greedy Monroe approximation algorithm")]


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
def gmScore( convV, c, nk, m ):
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
      (s,v) = gmScore( convV, i, nk, m  )
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


RULES += [("algoP","Algorithm P (approximation for Chamberlin--Courant)")]




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


RULES += [("rangingCC","Ranging variant of Algorithm P (approximation for Chamberlin--Courant)")]


rangingCC = algoP_ranging




#################################
# ILP START                     #
#################################


RULES += [("ccILP","Chamberlin--Courant (implemented through ILP solving)")]



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
RULES += [("PAV","PAV using Borda scores (implemented through ILP solving)")]


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







RULES += [("PAVtopk","PAV for k-Approval scoring (computed through ILP solving)")]
  

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



RULES += [("monroeILP","Monroe (implemented through ILP solving)")]

 
# compute Monroe with ILP
def monroeILP( V, k ):
  m = len( V[0] )
  n = len( V )

  print >>sys.stderr, "CPLEX START: computing Monroe with ILP"
  (total_satisfaction, winning_committee) = ilp1.run_ilp_monroe(np.array(V), k, np.arange(m - 1, -1, -1))
  print >>sys.stderr, "CPLEX END: Monroe ILP winning_committee"
  print >>sys.stderr, winning_committee
  
  return list(winning_committee)


#################################
# ILP END                       #
#################################  
