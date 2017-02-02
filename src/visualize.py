from sys  import *
from math import *
from PIL  import Image, ImageDraw
from PIL  import ImageColor

DIMENSION = 2

def perhapsFloat(v):
  try:
    return float(v)
  except:
    return v

def readData( f ):
  lines = f.readlines()

  (m, n, k) = lines[0].split();
  m = int(m)
  n = int(n)
  k = int(k)

  C = []
  V = []
  W = []

  for l in lines[1:m+1]:
    s = l.split()[1:]
    s = [perhapsFloat(x) for x in s]
    C += [s]

  for l in lines[m+1:m+n+1]:
    s = l.split()[m:]
    s = [float(x) for x in s]
    V += [s]

  for l in lines[n+m+1:n+m+k+1]:
    s = l.split()[1:]
    s = [perhapsFloat(x) for x in s]
    W += [s]

#  print len(C)
#  print C
#  print "---"
#  print V
#  print "---"
#  print W

  return (m,n,k,C,V,W)

def dist( x, y ):
  return (sum( [ (x[i]-y[i])**2 for i in range(DIMENSION)] ))**(0.5)

# Computes distances of the voters to the closest members of the committee
def compute_dist(V, Winners):
  d = 0.0
  max_dist = 0.0
  n = 0
  for v in V:
    dmin = float("inf")
    for w in Winners:
      dmin = min(dmin, dist(v, w))
    d += dmin
    max_dist = max(max_dist, dmin)
    n += 1
  return d / n, max_dist

# Computes distance of each committee member to the closest n/k voters 
def compute_dist_of_representatives_to_virt_districts(V, Winners):
  n = len(V)
  k = len(Winners)
  d = 0.0
  max_dist = 0.0
  for w in Winners:
    distances = []
    dmin = 0.0
    for v in V:
      distances.append(dist(v, w))
    for val in sorted(distances)[: int(n/k) ]:
      dmin += val
    d += dmin
    max_dist = max(max_dist, dmin)
  return d / k, max_dist

def compute_winners_per_party(C, W):
  result = {}
  for c in C:
    party = c[-1]
    if party not in result.keys():
      result[party] = 0
  for w in W:
    party = w[-1]
    result[party] += 1
  return result

data_in  = open( argv[1]+".win", "r")

(m,n,k,C,V,Winner) = readData( data_in )

avg_d, max_d = compute_dist(V, Winner)
rep_avg_d, rep_max_d = compute_dist_of_representatives_to_virt_districts(V, Winner)
perParty = compute_winners_per_party(C, Winner)

stats_out  = open("stats.out", "a")
stats_out.write(argv[1] + ": \n")
stats_out.write("  avg_d = " + str(avg_d) + "\n")
stats_out.write("  max_d = " + str(max_d) + "\n")
stats_out.write("  rep_avg_d = " + str(rep_avg_d) + "\n")
stats_out.write("  rep_max_d = " + str(rep_max_d) + "\n")
for (p, v) in perParty.iteritems():
  stats_out.write("  party-" + str(p) + " = "+ str(v) + "\n")
stats_out.close()


W = 600
H = 600

im = Image.new("RGB", (W,H), "white")
dr = ImageDraw.Draw( im )


hx = 2
hy = 2


dr.line( (0, H/2, W, H/2), fill=128)
dr.line( (W/2, 0, W/2, H), fill=128)
for z in C:
  dr.ellipse( (W/2+z[0]*100-hx, H/2-z[1]*100-hy, W/2+z[0]*100+hx, H/2-z[1]*100+hy), fill= "rgb(220,220,220)" )
#  print z


hx = 2
hy = 2


dr.line( (0, H/2, W, H/2), fill=128)
dr.line( (W/2, 0, W/2, H), fill=128)
for z in V:
  dr.ellipse( (W/2+z[0]*100-hx, H/2-z[1]*100-hy, W/2+z[0]*100+hx, H/2-z[1]*100+hy), fill= "rgb(150,150,150)" )
#  print z


wx = 5
wy = 5
for z in Winner:
  dr.ellipse( (W/2+z[0]*100-wx, H/2-z[1]*100-wy, W/2+z[0]*100+wx, H/2-z[1]*100+wy), fill="red" )
#  print z


dr.text((0,0), argv[1]+" (%d out of %d)"%(len(Winner),len(C)), fill="blue")

im.save( argv[1]+".png")
