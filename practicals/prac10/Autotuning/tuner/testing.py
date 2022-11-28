"""
Autotuning System

testing.py

Checks the results given by Optimisation against those of OptimisationBF.
"""

from optimisation import Optimisation
from optimisation_bf import OptimisationBF
from evaluator import Evaluator
from test_evaluations import generateEvalFunc, FuncEvaluator
from vartree import treeprint_str
from helpers import strVarVals
import random


# Perform some sample tests.
def run_testing():
    
    # The Possible Values
    #possValues = {'A' : [3,2,1],
    #              'B' : [3,5,7],
    #              'C' : [2,3],
    #              'D' : [10,5],
    #              'E' : [4,2],
    #              'F' : [1,2],
    #              'G' : [9,4],
    #              'H' : [11,2],
    #              'I' : [6,4,2]}
    
    # To make this a little different every time, lets randomise possValues.
    def randValList():
        return [random.randint(1,10) for i in range(0,random.choice([2,2,2,3]))]
    
    # The Possible Values
    possValues = {'A' : randValList(),
                  'B' : randValList(),
                  'C' : randValList(),
                  'D' : randValList(),
                  'E' : randValList(),
                  'F' : randValList(),
                  'G' : randValList(),
                  'H' : randValList(),
                  'I' : randValList()}
    
    
    print "Possible Values of Variables:\n(we use the same ones for all examples)"
    print strVarVals(possValues)
    print
    
    
    # First Example ------------------------------------------------------------ 
    
    
    print
    print "Example 1".center(80)
    print
    
    sampleTree1 = "{A, B, {C, D}, {E, F}}"
    
    # Generate a suitable evaluation function for testing
    evaluate_ex1 = generateEvalFunc(sampleTree1)
    
    
    # Run the testing
    print compare_optimisation(sampleTree1, possValues, evaluate_ex1, True)
    
    
    
    
    # Second Example -----------------------------------------------------------
    
    print
    print "Example 2".center(80)
    print
    
    sampleTree2 = "{A, B, {I, {C, D}, {E, F}}, {G, H}}"

    # Generate a suitable evaluation function for testing
    evaluate_ex2 = generateEvalFunc(sampleTree2)
    
    
    # Run the testing
    print compare_optimisation(sampleTree2, possValues, evaluate_ex2, True)
    
    
    







# Returns a string detailing the comparison of Optimisation and OptimisationBF
# vt_str, possValues, evaluationFunc are the data to be passed to the optimisation object.
# showPossValues - whether the possible values used for the testing should be printed.
# showTree - whether to include a tree view of the vartree.
# testMin, testMax - whether to test either the minimisation, maximisation, or both.
def compare_optimisation(vt_str, possValues, evaluationFunc, showTree = False, showPossValues = False, testMin = True, testMax = True):
    
    if not (testMin or testMax):
        return "No tests to perform\n"
    
    
    
    result = "Syntax:\n" + vt_str + "\n\n"
    
    if(showTree):
        result += treeprint_str(vt_str) + "\n\n"
    
    if(showPossValues):
        result += "Possible Values of variables:\n"
        result += strVarVals(possValues) + "\n\n"
    
    
    # Run the brute force test
    
    evaluator = FuncEvaluator(evaluationFunc)
    
    test_bf = OptimisationBF(vt_str, possValues, evaluator)
    
    
    if testMin:
        test_bf.calculateOptimum()
        
        result += "Minimal Valuation (by brute force):\n"
        result += strVarVals(test_bf.optimalValuation(), ", ")
        result += "\nThe score is %(s)d, found in %(n)d evaluations.\n\n" % {'n' : test_bf.numTests(), 's' : test_bf.optimalScore()}
        
        min_bf = test_bf.optimalScore()
        min_bf_v = test_bf.optimalValuation()
    
    
    if testMax:
        test_bf.maximiseScore()
        test_bf.calculateOptimum()
        
        result += "Maximal Valuation (by brute force):\n"
        result += strVarVals(test_bf.optimalValuation(), ", ")
        result += "\nThe score is %(s)d, found in %(n)d evaluations.\n\n" % {'n' : test_bf.numTests(), 's' : test_bf.optimalScore()}
        
        max_bf = test_bf.optimalScore()
        max_bf_v = test_bf.optimalValuation()
    
    
    # Run the main test
    
    evaluator.clearData()
    
    test = Optimisation(vt_str, possValues, evaluator)
    
    if testMin:
        test.calculateOptimum()
        
        result += "Minimal Valuation (by observing independence):\n"
        result += strVarVals(test.optimalValuation(), ", ")
        result += "\nThe score is %(s)d, found in %(n)d evaluations.\n\n" % {'n' : test.numTests(), 's' : test.optimalScore()}
        
        min = test.optimalScore()
        min_v = test.optimalValuation()
    
    
    if testMax:
        test.maximiseScore()
        test.calculateOptimum()
        
        result += "Maximal Valuation (by observing independence):\n"
        result += strVarVals(test.optimalValuation(), ", ")
        result += "\nThe score is %(s)d, found in %(n)d evaluations.\n\n" % {'n' : test.numTests(), 's' : test.optimalScore()}
        
        max = test.optimalScore()
        max_v = test.optimalValuation()
    

    
    
    # Compare the tests
    
    
    if ((not testMin) or (min_bf == min)) and ((not testMax) or (max_bf == max)):
        result += "These results are equal.\n"
        if (testMin and (min_bf_v != min_v)) or (testMax and (max_bf_v != max_v)):
            result += "(Althouh they found different valuations.)\n"
        result += "The new algorithm performed %(pct)d%% of the number of tests required by brute-force.\n" % {'pct' : round(float(test.numTests()) / float(test_bf.numTests()) * 100.0)}
    else:
        result += "These results are DIFFERENT." # Hopefully not reached.
    print
    
    
    
    
    
    return result






def strVarVals(d, sep="\n"):
    return sep.join([str(var) + " = " + str(val) for var,val in sorted(d.items())])




if __name__ == "__main__":
    print __doc__
   
   
   
    
