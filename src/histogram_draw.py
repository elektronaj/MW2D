from sys  import *
from math import *
from PIL  import Image, ImageDraw
from PIL  import ImageColor


def readData( f ):
  lines = f.readlines()

  (W, H) = lines[0].split()
  W = int(W)
  H = int(W)

  HISTOGRAM = []

  for l in lines[1:H+1]:
    s = l.split()
    s = [int(v) for v in s]
    HISTOGRAM += [s]

  return (W, H, HISTOGRAM)





if( len( argv ) <= 1 ):
  print "Invocation: "
  print "  python histogram_draw.py file.hist [epsilon [color]]"
  print ""
  print "epsilon -- small scaling value (eps = 0.0004 is typical)"
  print "color   -- (r,g,b) value of the most intense color (r,g,b values should be between 0 and 1)"
  print 
  exit()


print "LOADING...", argv[1]
f = open( argv[1], "r" )
(W, H, HISTOGRAM) = readData(f)

TRADITIONAL = False
try:
  threshold = float( argv[2] )
except:
  TRADITIONAL = True

try:
  color = eval( argv[3] )
  r = float(color[0])
except:
  color = (0,0,1)

(col_r, col_g, col_b) = color
(col_r, col_g, col_b) = (1-col_r, 1-col_g, 1-col_b)



im = Image.new("RGB", (W,H), "white")
dr = ImageDraw.Draw( im )

hx = 1
hy = 1


TOTAL = 0
MAX   = 0
for y in range(H):
  for x in range(W):
    TOTAL += HISTOGRAM[y][x]
    if( HISTOGRAM[y][x] > MAX ):
      MAX = HISTOGRAM[y][x]


print "DRAWING, TOTAL = %d" % TOTAL

dr.line( (0, H/2, W, H/2), fill=128)
dr.line( (W/2, 0, W/2, H), fill=128)

if( TRADITIONAL ):
  print "LOCAL NORMALIZATION"
  for y in range(H):
    for x in range(W):
      if( HISTOGRAM[y][x] > 0 ):
        inte = 255-int(255*(    float(HISTOGRAM[y][x]) / MAX) )
  #      dr.ellipse( (x-hx, y-hy, x+hx, y+hy), fill= "rgb(255,%d,%d)" % (inte,inte)  )
        dr.point( (x,(H-1)-y), fill= "rgb(255,%d,%d)" % (inte,inte)  )
else:
  MAX_VAL = 0.0
#  threshold = 0.0005
  print "GLOBAL NORMALIZATION"
  test = 0
  for y in range(H):
    for x in range(W):
      if( HISTOGRAM[y][x] > 0 ):
        inte = float(HISTOGRAM[y][x])/ TOTAL
#        if( inte > threshold ):   
#          dr.point( (x,(H-1)-y), fill= "rgb(128,255,128)"  )
#          test += 1
#        else:
        val = float(inte)/threshold
        MAX_VAL = max( val, MAX_VAL )
        val = (atan(val))/(pi/2)
#        val = log(1+val)
#        val = min(1.0, val)
####        val = 255-int(val*255)
#        dr.point( (x,(H-1)-y), fill= "rgb(%d,%d,255)" % (val,val)  )
        dr.point( (x,(H-1)-y), fill= "rgb(%d,%d,%d)" % (255-255*(col_r*val),255-255*(col_g*val),255-255*(col_b*val))  )


#        inte = 255-int(255*(    float(HISTOGRAM[y][x]) / MAX) )
#        dr.ellipse( (x-hx, y-hy, x+hx, y+hy), fill= "rgb(255,%d,%d)" % (inte,inte)  )

  print "MAX_VAL = ", MAX_VAL



im.save( argv[1].replace(".","_")+".png")
