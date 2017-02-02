from sys  import *
from math import *
from PIL  import Image, ImageDraw
from PIL  import ImageColor
from math import pi
from math import atan


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
    s = [float(s[0]), float(s[1])]
    C += [s]

  for l in lines[m+1:m+n+1]:
    s = l.split()[m:]
    s = [float(x) for x in s]
    V += [s]

  for l in lines[n+m+1:n+m+k+1]:
    s = l.split()[1:]
    s = [float(s[0]), float(s[1])]
#    s = [float(x) for x in s]
    W += [s]

  return (m,n,k,C,V,W)



X1 = -3
X2 =  3
Y1 = -3
Y2 =  3
PPU = 20  # points per unit

W = (X2-X1)*PPU
H = (Y2-Y1)*PPU


HISTOGRAM = [ [0]*W for i in range(H)]

MAX = 0

print "LOADING..."


for i in range(1, int(argv[2])+1):
  print i
  count = 0
  try:
    data_in  = open( argv[1]+("-%d.win" % i), "r")
    (m,n,k,C,V,Winner) = readData( data_in )

    for (x,y) in Winner:
      if( x < X1 or x > X2 or y < Y1 or y > Y2): 
        continue
      x -= X1
      y -= Y1
      x*= PPU
      y*= PPU
      HISTOGRAM[int(y)][int(x)] += 1

#      for dy in [-1,0,1]:
#        for dx in [-1,0,1]:
#          HISTOGRAM[int(y)+dy][int(x)+dx] += 0.5


      count += 1
      if( HISTOGRAM[int(y)][int(x)] > MAX ):
        MAX = HISTOGRAM[int(y)][int(x)]
  except:
    print "No file", argv[1]+"-"+str(i)

#  print count  

print "MAX = ", MAX

im = Image.new("RGB", (W,H), "white")
dr = ImageDraw.Draw( im )

hx = 1
hy = 1


#for y in range(H):
#  for x in range(W):
##    HISTOGRAM[y][x] =  (HISTOGRAM[y][x])**(2)
#    if( HISTOGRAM[y][x] >= 5 ):
#       HISTOGRAM[y][x] = 5
#    if( HISTOGRAM[y][x] <= 3 ):
#       HISTOGRAM[y][x] = 0

MAX=0
TOTAL = 0
for y in range(H):
  for x in range(W):
    TOTAL += HISTOGRAM[y][x]
    if( HISTOGRAM[y][x] > MAX ):
      MAX = HISTOGRAM[y][x]


print "WRITING"
print "MAX = %d   TOTAL = %d" % (MAX, TOTAL)

output = file( argv[1]+".hist", "w" )

output.write("%d %d\n" % (W,H))

for y in range(H):
  for x in range(W):
    output.write("%d " % HISTOGRAM[y][x])
  output.write("\n")
