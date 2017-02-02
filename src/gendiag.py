from os  import system
from sys import *

str = """candidates
uniform -4 -4 4 4 400
voters
uniform -4 -4 4 4 400
generate square
sntv 50 ccg-5-%d"""



print argv

if( len(argv) < 5 ):
  print "Invocation:"
  print "  python gendiag.py  input_template rule #from #to #committee_size"
  exit()

# get arguments
DIR   = "data_"+argv[1]
INPUT = argv[1]
RULE  = argv[2]
FROM  = int(argv[3])
TO    = int(argv[4])
K     = int(argv[5])

# prepare directory for the experiment

system("mkdir %s" % DIR )
system("cp *.py %s" % DIR )

# read the input file
in_file     = open( INPUT, "r" )
in_template = in_file.read()

print in_template



for i in range(FROM,TO+1):
  dataname = "data_%s_%d_%d" % (RULE, K, i)
  inputname = "input_%s_%d" % (RULE, i)

  data_out = open( '%s/%s' % (DIR, inputname), 'w')
  data_out.write( in_template )
  data_out.write( "generate %s\n" % dataname )
  data_out.write( "%s %d %s_%d-%d\n" % (RULE, K, RULE, K, i) )
  data_out.close()  
  cmd = "cd %s; python experiment.py <%s; rm %s; rm %s.out; rm %s.in" % (DIR, inputname, inputname, dataname, dataname)
  print cmd
  system( cmd )
#  system("python experiment.py <input_diag")

