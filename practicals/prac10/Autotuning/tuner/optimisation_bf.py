"""
Autotuning System

optimisation_bf.py

Implements the OptimisationBF class, a brute force optimiser implementing the 
same methods as the Optimisation class. Used for comparison to that class.

Much of this is simply copied over from optimisation.py, only the optimisation 
algorithm itself has changed.
"""


# defines the VarTree class and a parser converting strings to VarTrees.
from vartree import VarTree, vt_parse
# Helpers
from helpers import crossproduct



# The OptimisationBF class #####################################################


class OptimisationBF:
    
    
    def __init__(self, vartree, possValues, evaluator):
        
        vt = vt_parse(vartree)
        
        self.__vartree = vt
        self.__possValues = possValues
        self.__evaluator = evaluator
        
        self.minimiseScore() # by default we take smaller scores as better.
        
        self.__resetStoredVals()
    
    
    # Resets memoized info and saved results.
    # Can be called if anything is changed which would invalidate this.
    def __resetStoredVals(self):
        self.__optValuation = None
        self.__optScore = None
        self.__numTests = None
        self.__successful = None
        self.__evaluator.clearData()
    
    
    # updates possValues
    def setPossValues(self, possValues):
        self.__possValues = possValues
        self.__resetStoredValues()
    
    
    # updates Evaluator
    def setEvaluator(self, evaluator):
        self.__evaluator = evaluator
        self.__resetStoredValues()
    
    
    # use 'min' to calculate optimum valuation.
    def minimiseScore(self):
        self.__best = min
        self.__resetStoredVals()
    
    # use 'max' to calculate optimum valuation.
    def maximiseScore(self):
        self.__best = max
        self.__resetStoredVals()
    
    
    # Returns an optimal valuation
    def optimalValuation(self):
        if self.__optValuation is None:
            self.calculateOptimum()
        
        return self.__optValuation
    
    # Returns the score of an optimal valuation
    def optimalScore(self):
        if self.__optScore is None:
            self.calculateOptimum()
        
        return self.__optScore
    
    # Returns the number of tests performed during optimisation.
    # Strictly, this could instead be calculated from the structure of the VarTree and possValues.
    def numTests(self):
        if self.__numTests is None:
            self.calculateOptimum()
        
        return self.__numTests
    
    
    # Performs optimisation routine
    # then sets __optValuation, __optScore and __numTests
    # (__numTests is set implicitly by __evaluate)
    def calculateOptimum(self):
        
        opt = self.__optimise()
        
        if opt is None:
            self.__successful = False
            self.__resetStoredVals()
        else:
            self.__optValuation, self.__optScore = opt
            self.__numTests = len(self.__evaluator.log)
            self.__successful = True
        
        
        return None
    
    
    # Checks if the optimisation was a success.
    # This will only be false if no evaluations could be successfully performed.
    def successful(self):
        return self.__successful
    
    
    
    # Works out beforehand how many TESTS (not command executions) will be required
    def testsRequired(self):
        # We use brute force, so just the product of the numbers of possible values.
        vars = self.__vartree.flatten()
        t = 1
        for v in vars:
            t *= len(self.__possValues[v])
        
        return t
    
    
    # A brute-force optimisation function.
    # Used when the recursive optimisation bottoms out.
    # This is split into a seperate function so that different test strategies can be implemented more easily.
    def __optimise(self):
        
        # vt is the subtree to be optimised by brute-force (typically a leaf node)
        # presets gives values for all other variables not in vt.
        
        # list of lists of (variable name, possible value) pairs (each sublist deals with a single variable)
        varVals = [[(var, val) for val in self.__possValues[var]] for var in self.__vartree.flatten()]
        
        # tests is a list of dictionaries mapping variables to single values.
        # each is a variable valuation which is to be tested.
        tests = map(dict, crossproduct(varVals))
        
        # Run the testing for this batch of tests.
        self.__evaluator.evaluate(tests)
        
        # Filter any tests which failed
        tests = filter(lambda v: self.__evaluator.score(v) is not None, tests)
        if tests == []:
            return None # There are no tests which evaluated correctly.
        
        # Now choose the best to be returned.
        optValuation = self.__best(tests, key=self.__evaluator.score)
        optScore = self.__evaluator.score(optValuation)
        
        return (optValuation, optScore)
        
    
    
    
    







if __name__ == "__main__":
    print __doc__
    
