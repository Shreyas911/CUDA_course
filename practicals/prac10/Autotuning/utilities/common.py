"""
Autotuning System

output_common.py

A collection of functions which are common to several of the different output 
methods. This script doesn't directly produce output.

So far, this consists of the functions required to read in the CSV file.
"""


# File opening in Python 2.5
from __future__ import with_statement
# CSV file reading
import csv
# Maths
import math

# Read VarTree
# To get to ../tuner/vartree.py, we need to add it to sys.path, as this script 
# will be run directly, so relative imports won't work.
import sys
import os

# Trying to be safe against any odd cases...
tuner_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tuner'))
if tuner_path not in sys.path:
    sys.path.insert(0, tuner_path)

from vartree import vt_parse, treeprint

# Read the conf file
from ConfigParser import RawConfigParser






# Given the name of a CSV file, read it in and convert the data into the list 
# of tests used by these utilities. Also compute some other info, such as the 
# list of variables and their possible values.
# Return the tuple (tests, vars, possValues, repeat).
def readCSV(csv_file):
    
    try:
        # Read in the CSV file
        with open(csv_file, 'rb') as f: # This construct requires python 2.5
            
            rows = csv.reader(f, skipinitialspace=True)
            
            # The field names are given in the first line
            header = rows.next()
            
            vars, repeat = processHeader(header)
            
            # The row format is as follows:
            # TestNo, <-- params -->, <-- Score_1 to Score_n -->, Score_Overall
            
            # We will convert to the following:
            # list of tuples: (test_no, valuation, score_list, score_overall)
            # As we do this, build up the possValues mapping.
            
            
            # Initialise possValues to none for each varaible
            possValues = {}
            for v in vars:
                possValues[v] = []
            # Process each row, building up the list of tests and possValues
            tests = []
            for r in rows:
                scores = r[-2-repeat+1:-1]
                vals = r[1:-1-repeat]
                valuation = {}
                for var, val in zip(vars, vals):
                    valuation[var] = val
                    if val not in possValues[var]:
                        possValues[var].append(val)
                
                tests.append((int(r[0]), valuation, scores, r[-1])) #N.B. leaving scores as strings, as some may be empty/missing
            
            
            
            #print "Vars: " + str(vars)
            #print "Repetitions: " + str(repeat)
            #print "Possible Values: " + str(possValues)
            #print "No. of Tests: " + str(len(tests))
            #print
            #for t in tests:
            #    print t
            
            
            
            return (tests, vars, possValues, repeat)
        
    except IOError:
        print "Could not open file '" + csv_file + "'"
        exit()
    









# Given a header line from the CSV file, determne the list of variable names
# (in order) and the number of repetitions for each test.
# Return the pair (vars, repeat).
def processHeader(header):
    # First, quite a lot of validation of the headers,
    # which must exactly match the pattern:
    # TestNo, <-- params -->, <-- Score_1 to Score_n -->, Score_Overall
    
    if len(header) < 4:
        print "The CSV file doesn't seem to be in the required format."
        print "(Not enough fields)"
        exit()
    
    if header[0] != "TestNo":
        print "The CSV file doesn't seem to be in the required format."
        print "(First field is not 'TestNo')"
        exit()
    
    if header[-1] != "Score_Overall":
        print "The CSV file doesn't seem to be in the required format."
        print "(Final field is not 'Score_Overall')"
        exit()
    
    # Get number of repetitions from header[-2] (of the form Score_n)
    if header[-2][:6] != "Score_":
        print "The CSV file doesn't seem to be in the required format."
        print "(Not enough 'Score_*' fields)"
        exit()
    
    repeat = int(header[-2][6:])
    
    # And some more validation...
    if repeat < 1:
        print "The CSV file doesn't seem to be in the required format."
        print "(Incorrect 'Score_*' fields)"
        exit()
    
    for i in range(1, repeat+1):
        if header[-2 - (repeat - i)] != ("Score_" + str(i)):
            print "The CSV file doesn't seem to be in the required format."
            print "(Incorrect 'Score_*' fields, 'Score_" + str(i) + "' not where expected)"
            exit()
    
    # Then fields header[1] to header[-1 -repeat] are the variable names
    vars = header[1:-1-repeat]
    
    
    return (vars, repeat)
    





# These may assume that at least one entry in xs is valid.
def score_std_dev(xs):
    return math.sqrt( (sum([float(x)**2 for x in xs if x != '']) / len(xs)) - (score_mean(xs)**2) )

def score_mean(xs):
    return sum([float(x) for x in xs if x != '']) / len(xs)




# A canonical representation of valuations
def dict2key(d):
    return tuple(sorted(d.items()))



# Ranges of scores

def score_range(l):
    return max(l) - min(l)

def avg_range(l):
    if len(l) > 0:
        return sum(l) / len(l)
    else:
        return 0






# Read a VarTree from a conf file
def readVarTree(conf_file):
    config = RawConfigParser()
    success = config.read(conf_file)
    
    if success == []:
        print "Could not open file '" + conf_file + "'"
        exit()
    
    if not(config.has_option("variables", "variables")):
        print "Config file does not contain the option 'variables' in section [variables]."
        exit()
    
    vartree_str = config.get("variables", "variables")
    
    return vt_parse(vartree_str)
 




# Return the cross product of a list of lists
def crossproduct(xss):
    cp = [[]]
    for xs in xss:
        cp = [xs2 + [x] for x in xs for xs2 in cp]
    return cp








if __name__ == "__main__":
    print __doc__

