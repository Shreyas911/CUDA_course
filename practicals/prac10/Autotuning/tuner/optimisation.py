"""
Autotuning System

optimisation.py

Defines the Optimisation class.
This represents the optimisation algorithm.
"""


# defines the VarTree class and a parser converting strings to VarTrees.
from vartree import VarTree, vt_parse
# cross product function
from helpers import crossproduct



# The optimisation class #######################################################

class Optimisation:
    
    # vartree    - an instance of VarTree giving the variable tree to work on.
    # possValues - a dictionary mapping variable names to lists of possible 
    #              values. (Must include mappings for all vars in vartree.)
    # evaluator  - an Evaluator, used to evaluate tests (in batches).
    
    # Initialisation
    def __init__(self, vartree, possValues, evaluator):
        
        vt = vt_parse(vartree)
        
        self.__vartree = vt
        self.__possValues = possValues
        self.__evaluator = evaluator
        
        self.minimiseScore() # by default we take smaller scores as better.
        
        self.__resetStoredVals()
    
    
    # Resets memoized info and saved results.
    # Can be called if anything is changed which would invalidate these.
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
    def numTests(self):
        if self.__numTests is None:
            self.calculateOptimum()
        
        return self.__numTests
    
    
    # Performs optimisation routine
    # then sets __optValuation and __optScore
    # (__numTests is set implicitly by __evaluate)
    def calculateOptimum(self):
        
        opt = self.__optimise(self.__vartree, {})
        
        if opt is None:
            self.__successful = False
            #self.__resetStoredVals() # I don't think this is needed, and furthermore, it calls evaluator.clearData(), clearing the failures log!
        else:
            self.__optValuation, self.__optScore = opt
            self.__numTests = len(self.__evaluator.log)
            self.__successful = True
        
        return None
    
    
    # Checks if the optimisation was a success.
    # This will only be false if no evaluations were successful.
    def successful(self):
        return self.__successful
    
    
    
    # Works out beforehand how many TESTS (not command executions) will be required
    def testsRequired(self):
        return self.__testsReq(self.__vartree)
    
    # Recursively works out the number of tests required
    def __testsReq(self, vt):
        topLevel = 1
        for var in vt.vars:
            topLevel *= len(self.__possValues[var])
        
        if vt.subtrees == []:
            # If a leaf, then brute force:
            # So just the product of the number of possible values
            return topLevel
        
        else:
            # If not a leaf, for each topLevel valuation, optimise children.
            # One test from each child overlaps.
            childTests = [self.__testsReq(st) for st in vt.subtrees]
            
            return topLevel * (sum(childTests) - (len(vt.subtrees) - 1))
    
    
    
    
    # Optimise a variable tree recursively, exploiting variable independence.
    def __optimise(self, vt, presets):
        # vt is the subtree to be optimised.
        # presets is a mapping which assigns values to any variables not in vt.
        
        
        
        # First, calculate all the possible valuations at this node.
        # The valuations at this level are the cross product of the possible values of the variables at this level.
       
        # List of lists of (variable name,possible value) pairs (each sublist deals with a single variable)
        topLevelVarVals = [[(var, val) for val in self.__possValues[var]] for var in vt.vars]
        
        # List of dictionaries of possible tests (each dict contains a single value for each var at this level) 
        topLevelTests = map(dict, crossproduct(topLevelVarVals))
        
        # These dictionaries only contain mappings for variables at this level.
        # So we merge the existing presets into topLevelTests
        [t.update(presets) for t in topLevelTests] # (update topLevelTests in place)
        
        
        # Split the branch node and leaf node cases
        if vt.subtrees != []: # Then vt.subtrees is nonempty and so vt is a branch node.
            
            # For each valuation, we must optimise the subtrees,
            # then we can find the optimal valuation.
            
            # possValuations will store valuations for ALL variables, which have optimised subtrees.
            possValuations = []
            
            for valuation in topLevelTests:
                
                # To optimise the subtrees, we must choose arbitrary values
                # for the variables in the other subtrees.
                # These are arbitrary because different subtrees are independent.
                
                valuation.update(self.__restrictArb(vt.flattenchildren(), self.__possValues))
                
                # Now valuation contains mappings for ALL variables.
                # before testing each subtree, the variables in that subtree should be removed from the valuation.
                
                for st in vt.subtrees:
                    
                    localValuation = valuation.copy()
                    
                    # Remove the variables in this subtree from the valuation
                    for v in st.flatten():
                        del localValuation[v]
                    
                    
                    # Recursively optimise the subtree
                    # the local optValuation returned here will be the same as valuation
                    # for all variables opther than those within st
                    # so we can overwrite it here.
                    # This also means we 'accumulate' an optimum valuation overall.
                    recurse = self.__optimise(st, localValuation)
                    
                    if recurse is None:
                        return None # There are no valuations of the subtrees which can be successfuly evaluated.
                    
                    valuation, localOptScore = recurse
                
                # Because we overwrote valuation, it is now set so that 
                # all subtree varaibles are set to their optimums
                # (for this valuation of vt)
                
                # So valuation now holds optimal settings for this choice of variables st this level.
                # And localOptScore is set to the score of the optimum for this choice at this level.
                
                possValuations.append(valuation)
                
            
            # Once the loop is complete, possValuations contains one entry
            # for each possible valuation at this level, but with the subtree variables
            # set to their optimums for that particular valuation at this level.
            
            # First filter out any tests which failed (evaluate returns None).
            possValuations = filter((lambda v: (v is not None) and (self.__evaluator.score(v) is not None)), possValuations)
            if possValuations == []:
                return None # There are no tests which evaluated correctly.
            
            # Now choose the best to be returned.
            optValuation = self.__best(possValuations, key=self.__evaluator.score)
            optScore = self.__evaluator.score(optValuation)
            
            return (optValuation, optScore)
            
            
        else: # Then vt.subtrees is empty and so vt is a leaf node.
            
            # Run the testing for this batch of tests.
            self.__evaluator.evaluate(topLevelTests)
            
            # Filter any tests which failed
            topLevelTests = filter(lambda v: self.__evaluator.score(v) is not None, topLevelTests)
            if topLevelTests == []:
                return None # There are no tests which evaluated correctly.
            
            # Now choose the best to be returned.
            optValuation = self.__best(topLevelTests, key=self.__evaluator.score)
            optScore = self.__evaluator.score(optValuation)
            
            return (optValuation, optScore)
            
            
        
    
    
    # Return a dictionary mapping each variable to one of its possible values
    # in this case we choose the first one which was listed
    def __restrictArb(self, vs, vals):
        return dict([(var,vals[var][0]) for var in vs])

    


################################################################################



if __name__ == "__main__":
    print __doc__
    

