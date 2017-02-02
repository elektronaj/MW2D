from sys  import *
from math import *
from PIL  import Image, ImageDraw
from PIL  import ImageColor
from math import pi
from math import atan
from math import ceil
from numpy import *
from winner import *

import networkx as nx

compute_monroe = False



def readData( f ):
  lines = f.readlines()

  (m, n, k) = lines[0].split();
  m = int(m)
  n = int(n)
  k = int(k)

  C = []
  V = []
  V_pos = []
  W = []
  W_pos = []

  for l in lines[1:m+1]:
    s = l.split()[1:]
    s = [float(s[0]), float(s[1])]
    C += [s]

  for l in lines[m+1:m+n+1]:
    s = l.split()
    vote = [int(x) for x in s[:m]]
    V += [vote]
    V_pos += [[float(s[m]),float(s[m+1])]]

  for l in lines[n+m+1:n+m+k+1]:
    W += [int( l.split()[0] )]
    s = l.split()[1:]
    s = [float(s[0]), float(s[1])]
    W_pos += [s]

  return (m,n,k,C,V,V_pos,W, W_pos)


def dist( x, y ):
  return (sum( [ (x[i]-y[i])**2 for i in range(len(x))] ))**(0.5)

def avg(l):
  if len(l) == 0:
    return 0.0
  else:
    return float(sum(l)) / len(l) 

def monroe_dist(V, W):
  G = nx.DiGraph()
  G.add_node('s', demand = -1 * len(V))
  G.add_node('t', demand = len(V))
  for i in range(len(W)):
    w_name = "w_" + str(i) 
    G.add_edge(w_name, 't', weight = 0, capacity = ceil( float(len(V)) / len(W) ) )

  for i in range(len(V)):
    v_name = "v_" + str(i) 
    G.add_edge('s', v_name, weight = 0, capacity = 1 )

  for i in range(len(V)):
    v_name = "v_" + str(i)
    for j in range(len(W)):
      w_name = "w_" + str(j) 
      G.add_edge(v_name, w_name, weight = int(100 * dist(V[i], W[j])), capacity = 1 )
  return nx.min_cost_flow_cost(G) / 100


assert monroe_dist([(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (1, 1)]) == 2
assert monroe_dist([(0, 0), (0, 0), (1, 0), (0, 1)], [(0, 0), (0, 0)]) == 2




################################################################################

  ####    #####    ##     #####     #     ####    #####     #     ####    ####
 #          #     #  #      #       #    #          #       #    #    #  #
  ####      #    #    #     #       #     ####      #       #    #        ####
      #     #    ######     #       #         #     #       #    #            #
 #    #     #    #    #     #       #    #    #     #       #    #    #  #    #
  ####      #    #    #     #       #     ####      #       #     ####    ####



sum_distances_cc_ll = []
winners_per_quarter_ll = []
sum_distances_monroe_ll = []


def statistics( m, n, k, C, V_ord, V, Winner_name, Winner):

  sum_distances_cc = 0
  for v in V:
    closest = float("inf")
    for w in Winner:
      closest = min(closest, dist(w, v))
    sum_distances_cc += closest
  sum_distances_cc_ll.append(sum_distances_cc/n)

  if compute_monroe:
    sum_distances_monroe_ll.append(monroe_dist(V, Winner))
  else:
    sum_distances_monroe_ll.append(0)

  winners_per_quarter = {}
  for (x, y) in Winner:
    key = (x >= 0, y >= 0)
    try:
      winners_per_quarter[key] += 1
    except KeyError:
      winners_per_quarter[key] = 1
  for v in winners_per_quarter.values():
    winners_per_quarter_ll.append(v)


def statistics_final( name, m,n,k ):  
  print name
  print "-"*len(name)
  print "Average distance of a voter to the closest winner:", mean(sum_distances_cc_ll)
  if( compute_monroe ):
    print "Mean distance to closest winner (respecting Monroe criterion", mean(sum_distances_monroe_ll)
  print "Winners per quarter variance   :",   var(winners_per_quarter_ll)
  print "Winners per quarter std. dev.  :",   std(winners_per_quarter_ll)
#  print winners_per_quarter_ll
  print




######################################################

   ##    #####   #####   #####    ####   #    #
  #  #   #    #  #    #  #    #  #    #   #  #
 #    #  #    #  #    #  #    #  #    #    ##
 ######  #####   #####   #####   #    #    ##     ###
 #    #  #       #       #   #   #    #   #  #    ###
 #    #  #       #       #    #   ####   #    #   ###


approx_ranging_cc = []
approx_kborda     = []
approx_bloc       = []


def empirical_approx( m, n, k, C, V, V_pos, Winner, Winner_pos ):
  

  bloc_winner = bloc( V, k )
  kborda_winner = kborda( V, k )  
  rangingCC_winner = rangingCC( V, k )

  bloc_score_W        = owaScoreProfile( Winner     , V, [1]*k, ([1]*k) + ([0]*(m-k)))
  bloc_score_Opt      = owaScoreProfile( bloc_winner, V, [1]*k, ([1]*k) + ([0]*(m-k)))
  kborda_score_W      = owaScoreProfile( Winner       , V, [1]*k, [m-i-1 for i in range(m)]  )
  kborda_score_Opt    = owaScoreProfile( kborda_winner, V, [1]*k, [m-i-1 for i in range(m)] )
  rangingCC_score_W   = owaScoreProfile( Winner          , V, [1]+[0]*(k-1), [m-i-1 for i in range(m)])
  rangingCC_score_Opt = owaScoreProfile( rangingCC_winner, V, [1]+[0]*(k-1), [m-i-1 for i in range(m)])

  approx_bloc.append( float(bloc_score_W)/float(bloc_score_Opt) )
  approx_kborda.append( float(kborda_score_W)/float(kborda_score_Opt) )
  approx_ranging_cc.append( float(rangingCC_score_W)/float(rangingCC_score_Opt) )


def empirical_approx_final( name, m,n,k ):

  print "Empirical approximation ratio for", name
  print "-"*len(name)
  print "Approximatio ration wrt. k-Borda    :", mean(approx_kborda)
  print "Approximatio ration wrt. Bloc       :", mean(approx_bloc)
  print "Approximatio ration wrt. rangingCC  :", mean(approx_ranging_cc)
  print



 #####   ######  #####           #####    ####    ####
 #    #  #       #    #          #    #  #    #  #
 #    #  #####   #    #          #    #  #    #   ####
 #####   #       #####           #####   #    #       #
 #   #   #       #               #       #    #  #    #
 #    #  ######  #      #######  #        ####    ####


rep_position = {}

def rep_pos( m, n, k, C, V, V_pos, Winner, Winner_pos ):

  if len(rep_position) == 0:
    for i in range(k):
      rep_position[i] = []


  for v in V:
    j=0
    for i in range(m):
      if v[i] in Winner:
        rep_position[j].append(i+1)
        j+=1


def rep_pos_final( name, m,n,k ):

  print name
  print "-"*len(name)
  for i in range(k):
    print "Average position of %d-th best committee member :" % (i+1), mean(rep_position[i])
  print



# MAIN

if __name__ == "__main__":


  # introduce youself
  if( len(argv) >= 2 and argv[1].endswith("help")):
    print "This script analzyzes sequences of elections obtained with gendiag"
    print
    print "Invocation:"
    print "  python analyze.py base_name number list-of-functions"
    print
    print "base_name - the base name from which to count (typically rule_committee-size"
    print "number    - number of elections to analyze"
    print "list-of-function - names of functions to run on the elections"
    exit()

  if( len(argv) < 4 ):
    print "try python analyze.py --help"
    exit()
  
  
  # decode the functions to run
  per_election_call = []
  final_call        = []
  for s in argv[3:]:
    exec("per_election_call += [%s]" % s )
    exec("final_call += [%s_final]" % s )
#    per_election_call += [exec(s)]
#    final_call        += [exec(s+"_final")]



  print "LOADING..."


  for i in range(1, int(argv[2])+1):
    print i
    count = 0
    try:
      data_in  = open( argv[1]+("-%d.win" % i), "r")
      (m,n,k,C,V,V_pos,Winner,Winner_pos) = readData( data_in )
      for function in per_election_call:
        function( m,n,k, C, V, V_pos, Winner, Winner_pos )

    except IOError:
      print "No file", argv[1]+"-"+str(i)

  for function in final_call:
        function( argv[1], m,n,k )
