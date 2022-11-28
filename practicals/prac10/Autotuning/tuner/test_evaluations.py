"""
Autotuning System

test_evaluations.py

Defines a function generateEvalFunc() which creates and returns an evaluation 
function. It takes a VarTree (as a string) as input and the evaluation function
returned exhibits exactly the pattern of variable independence described by the 
VarTree.

Also defines FuncEvaluator, a class implementing the same interface as 
Evaluator, but which uses an evaluation function to score tests, rather than 
executing a command.
"""

# Used for hashing in the simulated evaluation functions.
import hashlib

from vartree import VarTree, vt_parse

from evaluator import SingleTest



# evaluation functions are of the form:
# evaluate(valuation) = FOM
# valuation is a dictionary mapping variable names to values.
# eventually, the values will be strings, but for now we use integers.
# the function returns the FOM (figure of merit, usually time taken) of runnning the test with these values.




# Some sample evaluation functions #############################################


# This one ignores independence and gives a sum
# so the least possible value for each will be optimum
def evaluate_by_sum(valuation):
    
    # First, check we have all the variables we will want
    requireVars(valuation, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
    
    return sum([valuation[k] for k in needKeys], 0)
    


# This one has all variables depend on each other
def evaluate_all_dep(valuation):
    
    # First, check we have all the variables we will want
    __requireVars(valuation, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
    
    return HASH([valuation['A'], valuation['B'], valuation['C'], 
                valuation['D'], valuation['E'], valuation['F'], 
                valuation['G'], valuation['H'], valuation['I']])




################################################################################





# Generate general evaluation functions ########################################

# Takes a VarTree (as a string) as input.
# Returns an evaluation function with variable independence matching the given VarTree.
def generateEvalFunc(s):
    
    vt = vt_parse(s)
    
    # generate list of downward paths in vt.
    paths = genPaths(vt)
    
    def evaluate(valuation):
        # check we have the required variables
        requireVars(valuation, vt.flatten())
        
        # for each downward path, hash the variables together, as these are dependent on each other.
        dependencies = [[valuation[x] for x in xs] for xs in paths]
        hashes = [HASH(xs) for xs in dependencies]
        
        # add them all together, as the different valuations are independent.
        tot = sum(hashes)
        
        return tot
    
    return evaluate


################################################################################





# Some helpers #################################################################


# Check that the correct variables have been supplied.
def requireVars(valuation, vs):
    if any(not (v in valuation.keys()) for v in vs):
        print "evaluate() called with some missing variable(s)"
        exit()


# A scrambling function which some of these use
def HASH(xs):
    return int(int(hashlib.md5(''.join(map(str,xs))).hexdigest(),16) % 5000)


# Returns a list of lists of variables on each downward path in vt.
def genPaths(vt):
    
    if vt.subtrees == []:
        return [vt.vars]
    else:
        return [vt.vars + sp for sp in sum(map(genPaths, vt.subtrees), [])]


################################################################################



# A class implementing the same interface as Evaluator, but which uses an 
# evaluation function to score each test, instead of running commands.
class FuncEvaluator:
    
    def __init__(self, evaluationFunc):
        self.log = {}
        self.testNum = 0
        self.evaluationFunc = evaluationFunc
    
    
    def score(self, v):
        if self.__getTest(v) is None:
            self.testNum += 1
            t = SingleTest(self.testNum, v)
            t.overall = self.evaluationFunc(v)
            t.results = [t.overall]
            self.log[self.testNum] = t
            
            return t.overall
        else:
            return self.__getTest(v).overall
    
    
    def evaluate(self, valuations_list):
        for v in valuations_list:
            self.score(v)
    
    
    # Given a valuation, returns the matching test from the log.
    def __getTest(self, valuation):
        for t in self.log.values():
            if t.valuation == valuation:
                return t
        
        return None
    
    
    def clearData(self):
        self.log = {}
        self.testNum = 0







################################################################################


if __name__ == "__main__":
    print __doc__
