"""
Autotuning System

evaluator_batch.py

Defines the BatchEvaluator class.
This provides a method to actually execute tests which are required by the 
optimisation algorithm. This class handles compilation, execution and cleaning, 
and keeps a log of all tests performed.

It is different from Evaluator because it first compiles all valuations, 
creating a 'pool' of tests, which are run. Evaluator compiles and runs tests 
one at a time. BatchEvaluator is the sequential version of what 
ParallelEvaluator will be.
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
# The main Evaluator Definition
from evaluator import SingleTest, Evaluator




# The main evaluator class deals with compiling and running tests.
# For BatchEvaluator, we only redefine evaluate(), the rest is inherited.
class BatchEvaluator(Evaluator):
    
    # Evaluates a list of valuations.
    # The scores are saved to the test log.
    def evaluate(self, valuations_list):
        
        # This implementation first compiles all tests in the batch, then runs 
        # them (possibly multiple times), then cleans them.
        
        
        # First, ignore any tests which have already been performed.
        valuations_to_test = [v for v in valuations_list if self._getTest(v) is None]
        
        
        # Set up all the tests.
        for idx, valuation in enumerate(valuations_to_test):
            test_num = self.testNum + idx + 1
            self._createTest(test_num, valuation)
        
        if self.output['progress']:
            for idx, valuation in enumerate(valuations_to_test):
                test_num = self.testNum + idx + 1
                print >>output.full, "Test " + str(test_num) + ":"
                print >>output.full, strVarVals(valuation, ", ")
        
        
        
        # Next, compile the tests, if needed.
        if self.compile_mkStr is not None:
            for idx, valuation in enumerate(valuations_to_test):
                test_num = self.testNum + idx + 1
                
                if self.output['progress']:
                    print >>output.full, "Compiling test " +  str(test_num)
                    
                    print >>output.short, "Compiling test " +  str(test_num),
                    output.short.flush()
                
                cmdStr = self.compile_mkStr(test_num, valuation)
                
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
                    self.failures.append(("COMPILATION OF TEST " + str(test_num) + " FAILED.", valuation))
                    
                    if self.output['progress']:
                        print >>output.short, "(FAILED)"
                        output.short.flush()
                        
                else:
                    if self.output['progress']:
                        print >>output.short
                        output.short.flush()
        
        # Finished compilation
        
        
        
        # Create a pool of tests to be run.
        # This is a list of (test_num, valuation, run_num) pairs
        test_pool = []
        for idx, valuation in enumerate(valuations_to_test):
            test_num = self.testNum + idx + 1
            test_pool += [(test_num, valuation, i) for i in range(1,self.repeat+1)]

        
        
        
        # Run all the tests.
        for (test_num, valuation, run_num) in test_pool:
            
            if self.output['progress']: 
                nthRun = ""
                if self.repeat > 1:
                    nthRun = " ("+ordinal(run_num)+" run)"
                print >>output.full, "Running test " +  str(test_num) + nthRun
                print >>output.short, "Running test " +  str(test_num) + nthRun,
                
            
            # The command required to execute the test
            cmdStr = self.test_mkStr(test_num, valuation)
            
            
            
            # Run the test
            if self.custom_fom:
                
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
                    self.failures.append(("EVALUATION OF TEST " + str(test_num) + " FAILED.", valuation))
                    
                    if self.output['progress']:
                        print >>output.short, "(FAILED)"
                        output.short.flush()
                    
                    continue # This test cannot be run, skip ahead to the next one (poss just the next repetition).
                else:
                    if self.output['progress']:
                        print >>output.short 
                        output.short.flush()
                
                
                
                if len(out) == 0:
                    print "The test did not produce any output."
                    print "When using a custom figure-of-merit, the 'test' command must output the score as the final line of output."
                    exit() # Should probably throw some exception to be caught by the main program.
                
                # Take the last line of output to be the FOM.
                # Add this score to the log.
                try:
                    self._logTest(test_num, float(out[-1]))
                except ValueError:
                    # The final line could not be interpretd as a float.
                    print "The final line of output could not be interpreted as a score."
                    print "When using a custom figure-of-merit, the 'test' command must output the score as the final line of output."
                    print "This should be an integer or float, with no other text on the line."
                    print "Score could not be read from the following line: "
                    print out[-1]
                    exit() # Should probably throw some exception to be caught by the main program.
                
                if self.output['progress'] and self.repeat > 1:
                    print >>output.full, "Result of test " + str(test_num) + ", " + ordinal(i) + " run: " + str(float(out[-1]))
                
                
                
                
                
            else: # Not using a custom FOM, so we'll do the timing
                
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
                    self.failures.append(("RUNNING OF TEST " + str(test_num) + " FAILED.", valuation))
                    
                    if self.output['progress']:
                        print >>output.short, "(FAILED)",
                        output.short.flush()
                    
                    continue # This test cannot be run, skip ahead to the next one (poss just the next repetition).
                else:
                    if self.output['progress']:
                        print >>output.short
                        output.short.flush()
                
                
                # Take the difference between the start and stop times as the FOM.
                # Add this score to the log.
                self._logTest(test_num, stop - start)
                
                if self.output['progress'] and self.repeat > 1:
                    print >>output.full, "Result of test " + str(test_num) + ", " + ordinal(i) + " run: " + str(stop - start)
                
                
            
        # Finished the loop running all tests in pool.
        
        
        # Process the test results for each test.
        for idx, valuation in enumerate(valuations_to_test):
            test_num = self.testNum + idx + 1
            
            # Add the overall score to the test log.
            scores = self._getTest(valuation).results
            
            if len(scores) > 0: # Then some tests ran successfully
                
                if self.repeat > 1:
                    overall = self.aggregator(scores)
                else: # self.repeat == 1 and len(scores) == 1
                    overall = scores[0]
                
                self._logOverall(test_num, overall)
                
                if self.output['progress']: 
                    if self.repeat > 1:
                        stats = self._test_stats(scores)
                        
                        print >>output.full, "Results of test " + str(test_num) + ":"
                        print >>output.full, "Average Result: " + str(stats['avg'])
                        print >>output.full, "Minimum Result: " + str(stats['min'])
                        print >>output.full, "Maximum Result: " + str(stats['max'])
                        print >>output.full, "Median Result:  " + str(stats['med'])
                        print >>output.full, "Variance:       " + str(stats['variance'])
                        print >>output.full, "Std. Deviation: " + str(stats['std_dev'])
                        print >>output.full, "Coeff. of Var.: " + str(stats['cv'])
                    else:
                        print >>output.full, "Result of test " + str(test_num) + ": " + str(overall)
                
                
        # Finished processing the scores
        
        
        # Clean up all the tests
        if self.clean_mkStr is not None:
            for idx, valuation in enumerate(valuations_to_test):
                test_num = self.testNum + idx + 1
                
                if self.output['progress']:
                    print >>output.full, "Cleaning test " +  str(test_num)
                    print >>output.short, "Cleaning test " +  str(test_num),
                    output.short.flush()

                
                cmdStr = self.clean_mkStr(test_num, valuation)
                
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
                    self.failures.append(("CLEANUP OF TEST " + str(test_num) + " FAILED.\n(test was still used)", valuation))
                    
                    if self.output['progress']:
                        print >>output.short, "(FAILED)"
                        output.short.flush()
                    
                else:
                    if self.output['progress']:
                        print >>output.short
                        output.short.flush()
        
        
        
        # Finished cleaning the tests
        
        
        # Errata
        
        self.testNum += len(valuations_to_test)
        self.testsRun += len(valuations_to_test)
        
        # Done
    
    
    
    
    








if __name__ == "__main__":
    print __doc__


