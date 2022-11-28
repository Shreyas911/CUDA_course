"""
Autotuning System

evaluator.py

Defines the Evaluator class.
This provides a method to actually execute tests which are required by the 
optimisation algorithm. This class handles compilation, execution and cleaning, 
and keeps a log of all tests performed.
"""

# Running shell commands
from subprocess import Popen, PIPE, STDOUT
# Timing commands
import time
# Maths
import math
# Helpers
from helpers import avg, med, strVarVals, ordinal
# Control output
import output




# A class to encapsulate the inforamtion about a single test. Used in the log.
class SingleTest:
    
    def __init__(self, testId, valuation):
        self.testId = testId
        self.valuation = valuation
        self.results = []
        self.overall = None


# The main evaluator class deals with compiling and running tests.
class Evaluator:
    
    def __init__(self, compile_mkStr, test_mkStr, custom_fom, clean_mkStr, 
                 repeat, aggregator, past_evaluator=None):
        # The arguments are three "template" functions, which convert a 
        # valuation into a command string to compile/etc that particular test.
        # Any of them may be 'None', meaning that step is not performed.
        # repeat is the number of times to repeat each test.
        # aggregator is a function from a list of scores to the overall score.
        # (e.g. min, max, avg, med, ...)
        # past_evaluator is (optionally) another Evaluator which we can use the 
        # results from if it has already run a test we want.
        
        self.compile_mkStr = compile_mkStr
        self.test_mkStr    = test_mkStr
        self.clean_mkStr   = clean_mkStr
        
        self.custom_fom = custom_fom
        
        self.repeat = repeat
        self.aggregator = aggregator
        
        self.past_evaluator = past_evaluator
        self.testsRun = 0
        
        self.log = {}
        self.failures = []
        self.testNum = 0
        
        self.output = {'progress': True, 'testing': True}
    
    
    # Creates a new test in the log.
    def _createTest(self, testId, valuation):
        self.log[testId] = SingleTest(testId, valuation)
        return self.log[testId]
    
    # Adds a new test score to the log.
    def _logTest(self, testId, score):
        if testId in self.log:
            self.log[testId].results.append(score)
    
    # Adds an overall score to a test in the log.
    def _logOverall(self, testId, score):
        if testId in self.log:
            self.log[testId].overall = score
    
    
    
    # Given a valuation, returns the matching test from the log.
    def _getTest(self, valuation):
        # Check if it has been run by this evaluator.
        for t in self.log.values():
            if t.valuation == valuation:
                return t
        
        # Check if it has been run in the past.
        if self.past_evaluator is not None:
            t = self.past_evaluator._getTest(valuation)
            if t is not None:
                # Create and populate a local test
                self.testNum += 1
                t2 = self._createTest(self.testNum, valuation)
                for r in t.results:
                    self._logTest(self.testNum, r)
                self._logOverall(self.testNum, t.overall)
                
                return t2
        
        # It has not been run.
        return None
    
    
    # Resets all stored data, etc.
    # This is used when something changes in the optimiser
    # requiring it to flush all state since its creation.
    def clearData(self):
        self.log = {}
        self.failures = []
        self.testNum = 0
    
    
    # Returns the score of a valuation.
    # This will already have been computed, so is returned from the log.
    def score(self, valuation):
        t = self._getTest(valuation)
        if t is None:
            return None
        else:
            return t.overall
        
    
    
    # Evaluates a list of valuations.
    # The scores are saved to the test log.
    def evaluate(self, valuations_list):
        
        # This simple implementation runs each test in sequence.
        # There is no interleaving or variation of test repetitions.
        
        for valuation in valuations_list:
            # Process this individual test.
            
            # First check if this test has already been performed.
            # If so, we need not evaluate it again.
            if self._getTest(valuation) is None:
                
                self.testNum += 1
                self.testsRun += 1
                
                self._createTest(self.testNum, valuation)
                
                
                if self.output['progress']:
                    print >>output.full, "Test " + str(self.testNum) + ":"
                    print >>output.full, strVarVals(valuation, ", ")
                    
                    print >>output.short, "Test " + str(self.testNum) + ": ",
                    output.short.flush() 
            
                
                # First, compile the test, if needed.
                if self.compile_mkStr is not None:
                    if self.output['progress']:
                        print >>output.full, "Compiling test " +  str(self.testNum)
                        
                        print >>output.short, "Compiling, ",
                        output.short.flush()
                    
                    cmdStr = self.compile_mkStr(self.testNum, valuation)
                    
                    # Start the compilation
                    # Collect the output, without printing.
                    p = Popen(cmdStr, shell=True, stdout=PIPE, stderr=STDOUT)
                    
                    # Wait for the compilation to finish, this sets the return code.
                    p.wait()
                    
                    # Print the output
                    if self.output['testing']:
                        out = p.stdout.readlines()
                        print >>output.full, ''.join(out)
                    
                    # Check the retun code.
                    if(p.returncode != 0):
                        self.failures.append(("COMPILATION OF TEST " + str(self.testNum) + " FAILED.", valuation))
                        
                        if self.output['progress']:
                            print >>output.short, "(FAILED)"
                            output.short.flush()
                        
                        continue # This test cannot be compiled, skip ahead to the next one.
                
                
                # Repeat the tests the number of times specified
                for i in xrange(1, self.repeat +1):
                    
                    # Run the test
                    if self.custom_fom:
                        
                        if self.output['progress']: 
                            nthRun = ""
                            if self.repeat > 1:
                                nthRun = " ("+ordinal(i)+" run)"
                            print >>output.full, "Running test " +  str(self.testNum) + nthRun
                            if self.repeat > 1:
                                print >>output.short, ordinal(i) + " Run, ",
                                output.short.flush()
                            else:
                                print >>output.short, "Running, ",
                                output.short.flush()
                        
                        # Execute the evaluation, the result will be output on the last line.
                        cmdStr = self.test_mkStr(self.testNum, valuation)
                        
                        # Start the evaluation, capture output
                        p = Popen(cmdStr, shell=True, stdout=PIPE, stderr=STDOUT)
                        
                        # Wait for the evaluation to finish, this sets the return code.
                        p.wait()
                        
                        
                        # Get the program output.
                        out = p.stdout.readlines()
                        
                        if self.output['testing']: 
                            print >>output.full, ''.join(out)
                        
                        
                        # Check the retun code.
                        if(p.returncode != 0):
                            self.failures.append(("EVALUATION OF TEST " + str(self.testNum) + " FAILED.", valuation))
                            
                            if self.output['progress']:
                                print >>output.short, "(FAILED)"
                                output.short.flush()
                            
                            continue # This test cannot be run, skip ahead to the next one (poss just the next repetition).
                        
                        
                        if len(out) == 0:
                            print >>output.short, ""
                            print "The test did not produce any output."
                            print "When using a custom figure-of-merit, the 'test' command must output the score as the final line of output."
                            exit() # Should probably throw some exception to be caught by the main program.
                        
                        # Take the last line of output to be the FOM.
                        # Add this score to the log.
                        try:
                            self._logTest(self.testNum, float(out[-1]))
                        except ValueError:
                            # The final line could not be interpretd as a float.
                            print >>output.short, ""
                            print "The final line of output could not be interpreted as a score."
                            print "When using a custom figure-of-merit, the 'test' command must output the score as the final line of output."
                            print "This should be an integer or float, with no other text on the line."
                            print "Score could not be read from the following line: "
                            print out[-1]
                            exit() # Should probably throw some exception to be caught by the main program.
                        
                        if self.output['progress'] and self.repeat > 1:
                            print >>output.full, "Result of test " + str(self.testNum) + ", " + ordinal(i) + " run: " + str(float(out[-1]))
                        
                        
                        
                        
                        
                    else: # Not using a custom FOM, so we'll do the timing
                        
                        if self.output['progress']: 
                            nthRun = ""
                            if self.repeat > 1:
                                nthRun = " ("+ordinal(i)+" run)"
                            print >>output.full, "Running test " +  str(self.testNum) + nthRun
                            if self.repeat > 1:
                                print >>output.short, ordinal(i) + " Run, ",
                                output.short.flush()
                            else:
                                print >>output.short, "Running, ",
                                output.short.flush()
                        
                        # Execute test, the result will be the time taken.
                        cmdStr = self.test_mkStr(self.testNum, valuation)
                        
                        start = time.time()
                        
                        # Start the test
                        # Collect the output, without printing.
                        p = Popen(cmdStr, shell=True, stdout=PIPE, stderr=STDOUT)
                        
                        # Wait for the test to finish, this sets the return code.
                        p.wait()
                        
                        stop = time.time()
                        
                        # Print the output
                        if self.output['testing']:
                            out = p.stdout.readlines()
                            print >>output.full, ''.join(out)
                        
                        # Check the retun code.
                        if(p.returncode != 0):
                            self.failures.append(("RUNNING OF TEST " + str(self.testNum) + " FAILED.", valuation))
                            
                            if self.output['progress']:
                                print >>output.short, "(FAILED)",
                                output.short.flush()

                            
                            continue # This test cannot be run, skip ahead to the next one (poss just the next repetition).
                        
                        
                        # Take the difference between the start and stop times as the FOM.
                        # Add this score to the log.
                        self._logTest(self.testNum, stop - start)
                        
                        if self.output['progress'] and self.repeat > 1:
                            print >>output.full, "Result of test " + str(self.testNum) + ", " + ordinal(i) + " run: " + str(stop - start)
                        
                        
                        
                    
                    
                # End of for loop running the test multiple times
                
                # Add the overall score to the test log.
                scores = self._getTest(valuation).results
                
                if len(scores) > 0: # Then some tests ran successfully
                    
                    if self.repeat > 1:
                        overall = self.aggregator(scores)
                    else: # self.repeat == 1 and len(scores) == 1
                        overall = scores[0]
                    
                    self._logOverall(self.testNum, overall)
                    
                    if self.output['progress']: 
                        if self.repeat > 1:
                            stats = self._test_stats(scores)
                            
                            print >>output.full, "Results of test " + str(self.testNum) + ":"
                            print >>output.full, "Average Result: " + str(stats['avg'])
                            print >>output.full, "Minimum Result: " + str(stats['min'])
                            print >>output.full, "Maximum Result: " + str(stats['max'])
                            print >>output.full, "Median Result:  " + str(stats['med'])
                            print >>output.full, "Variance:       " + str(stats['variance'])
                            print >>output.full, "Std. Deviation: " + str(stats['std_dev'])
                            print >>output.full, "Coeff. of Var.: " + str(stats['cv'])
                        else:
                            print >>output.full, "Result of test " + str(self.testNum) + ": " + str(overall)
                    
                    
                
                
                # Run the cleanup, if needed
                if self.clean_mkStr is not None:
                    if self.output['progress']:
                        print >>output.full, "Cleaning test " +  str(self.testNum)
                        print >>output.short, "Cleaning, ",
                        output.short.flush()

                    
                    cmdStr = self.clean_mkStr(self.testNum, valuation)
                    
                    # Start the cleanup
                    # Collect the output, without printing.
                    p = Popen(cmdStr, shell=True, stdout=PIPE, stderr=STDOUT)
                    
                    # Wait for the cleanup to finish, this sets the return code.
                    p.wait()
                    
                    # Print the output
                    if self.output['testing']:
                        out = p.stdout.readlines()
                        print >>output.full, ''.join(out)
                    
                    # Check the retun code.
                    if(p.returncode != 0):
                        self.failures.append(("CLEANUP OF TEST " + str(self.testNum) + " FAILED.\n(test was still used)", valuation))
                        
                        if self.output['progress']:
                            print >>output.short, "(FAILED) ",
                            output.short.flush()
                        
                        # Need not 'continue', as we still got a result.
                    
                
                
                if self.output['progress']: 
                    print >>output.full, ""
                    print >>output.short, "Done. "
                    output.short.flush()
                
                
            # End of if stsement checking if test is fresh
            
        # End of for loop running multiple tests.
        
    # End of evaluate()
    
    
    
    
    
    # Calculates various statistics about the test results.
    def _test_stats(self, scores):
        # May assume that scores is not empty and contains floats.
        
        scores.sort()
        
        stats = {}
        
        stats['avg'] = avg(scores)
        
        stats['min'] = scores[0]
        
        stats['max'] = scores[-1]
        
        stats['med'] = med(scores)
        
        stats['variance'] = sum([s**2 for s in scores]) / len(scores) - stats['avg']**2
        
        stats['std_dev'] = math.sqrt(stats['variance'])
        
        stats['cv'] = stats['std_dev'] / abs(stats['avg']) if stats['avg'] != 0 else "Undefined (avg is 0)"
        
        return stats
        











if __name__ == "__main__":
    print __doc__


