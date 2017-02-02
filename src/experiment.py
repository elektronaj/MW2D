from os import system
from sys import *
from random import *
from subprocess import call
from PIL import Image

C = []
V = []

DATA = "C"
NAME = "data"


# GENERATE POINTS

def generateFromImage( filename, x1, y1, x2, y2, N, Party ):
  img = Image.open(filename)
  rgb_im = img.convert('RGB')
  
  x, y = rgb_im.size
  density_map = []
  total_color_intensity = 0
  for i in range(x):
    for j in range(y):
      pixel = rgb_im.getpixel((i,j))
      color_intensity = (255 - pixel[0]) + (255 - pixel[1]) + (255 - pixel[2])
      coor1 = x1 + (float(i * (x2 - x1)) / x)
      coor2 = y2 - (float(j * (y2 - y1)) / y)
      density_map.append((coor1, coor2, color_intensity))
      total_color_intensity += color_intensity
  random_list = [random()*total_color_intensity for i in range(N)]
  result = []
  i = 0
  passed_intensity = 0  
  for v in sorted(random_list):
    while passed_intensity < v:
      passed_intensity += density_map[i][2]
      i += 1
    result.append((density_map[i][0], density_map[i][1], Party))
  return result

def generateUniform( x1, y1, x2, y2, N, Party ):
  (x1,x2) = (min(x1,x2),max(x1,x2))
  (y1,y2) = (min(y1,y2),max(y1,y2))
  return [ (random()*(x2-x1)+x1, random()*(y2-y1)+y1, Party) for i in range(N)]


def generateGauss( x,y, sigma , N, Party ):
  return [ (gauss( x, sigma ), gauss( y, sigma ), Party) for i in range(N)]


def generateCircle( x, y, r, N, Party ):
  count = 0
  L = []
  while( count < N ):
    (px,py) = (random()*(2*r)-r, random()*(2*r)-r)
    if( px**2 + py**2 <= r**2 ):
      L += [(px+x,py+y, Party)]
      count += 1
  return L



# save data

def saveData( name ):
  f = open( name+".in", "w" )
  m = len( C )
  n = len( V )
  print >>f, m, n
  for p in C:
    print >>f, p[0], p[1], p[2]
  for p in V:
    print >>f, p[0], p[1], p[2]
  f.close()

  system( "python 2d2pref.py <%s.in >%s.out" % (name, name) )



# compute winners

def computeWinners( rule, k, output ):
  global NAME
  system( "python winner.py <%s.out >%s.win %s %d" % (NAME, output, rule, k) )
  system( "python visualize.py %s" % (output) )

def getOrNone(l, n):
  try:
    return l[n]
  except:
    return "NONE"

# COMMAND EXECUTION

def execute( command ):
  global DATA
  global NAME
  print command
  if( command[0] == "candidates" ):
    DATA = "C"
  elif( command[0] == "voters" ):
    DATA = "V"
  elif( command[0] == "circle" ):
    P = generateCircle( float(command[1]), float(command[2]), float(command[3]), int(command[4]), getOrNone(command, 5))
    X = eval(DATA)
    X += P
  elif( command[0] == "gauss" ):
    P = generateGauss( float(command[1]), float(command[2]), float(command[3]), int(command[4]), getOrNone(command, 5))
    X = eval(DATA)
    X += P
  elif( command[0] == "uniform" ):
    P = generateUniform( float(command[1]), float(command[2]), float(command[3]), float(command[4]), int(command[5]), getOrNone(command, 6))
    X = eval(DATA)
    X += P
  elif( command[0] == "image" ):
    P = generateFromImage( command[1], float(command[2]), float(command[3]), float(command[4]), float(command[5]), int(command[6]), getOrNone(command, 7))
    X = eval(DATA)
    X += P
  elif( command[0] == "generate" ):
    NAME = command[1]
    saveData( NAME )
  elif( command[0] == "#" ):
    None
  else:
    computeWinners( command[0], int(command[1]), command[2] )



# READ DATA IN
def readData( f ):
  cmd = []
  lines = f.readlines()

  for l in lines:
    s = l.split()
    if( len(s) > 0 ):
      cmd += [s]

  return cmd




# MAIN

if __name__ == "__main__":


  if( len(argv) > 1 ):
    print "This scripts runs a single experiment (generates an elections, \ncomputes the results accoring to specified rules, and prepares visualizations)"
    print
    print "Invocation:"
    print "  python experiment.py  <description.input"
    exit()


  seed()


  data_in  = stdin
  data_out = stdout

  cmd = readData( data_in )

  for command in cmd:
    if not command[0].lstrip()[0] == '#':
      execute( command )


