#! /usr/bin/env python

# Autotuning System

# v0.16
version = "v0.16"
# This is made global simply for convenience of updating it.


# Command line arguments, version check
import sys

# Check for python version before we do/import anything else.
# We require version 2.5 at the moment...
if sys.version_info[0] != 2 or sys.version_info[1] < 5:
    print("This script requires Python version 2.5")
    sys.exit(1)



# File path handling
import os
# Time entire execution
import time
# Configuration file handling
from tune_conf import get_settings
# The Evaluator
from evaluator import Evaluator
from evaluator_batch import BatchEvaluator
# The optimiser
from optimisation import Optimisation
# Printing VarTrees
from vartree import treeprint_str, get_variables
# Writing out of log files
from logging import writeCSV
# Helpers
from helpers import strVarVals, ordinal
# Control where output goes
import output


# The main script ##############################################################

def main():
    global version
    # The script has been executed directly
    
    
    # Command Line Arguments
    # Read name of config file from command line

    if len(sys.argv) == 2:
        configFile = sys.argv[1]
    else:
        print
        print "Autotuning System".center(80)
        print version.center(80)
        print
        print "Usage: \nPlease provide the path to a configuration file as an argument."
        print "When no arguments are provided, some sample tests are run."
        print "Press Enter to run the tests."
        
        try:
            raw_input()
        except KeyboardInterrupt: # Ctrl-C
            print
            exit()
        
        from testing import run_testing
        
        run_testing()
        
        exit()
    
    
    
    
    # Get the settings
    settings = get_settings(configFile)
    
    
    
    # Before anything, set up the type of output we use.
    # If they specify a script file, use production mode,
    # Otherwise use screen mode.
    if settings['script'] is not None:
        b = output.output_production(settings['script'])
        if not b:
            output.output_screen() # Revert to safe default
            print "Could not open script file '" + settings['script'] + "'"
            print "No script will be saved."
            
            input = None
            yes = ["y", "yes", "ye", ""]
            no = ["n", "no"]
            while (input is None) or (input.lower() not in (yes+no)):
                print "Do you want to continue anyway? [Y/n] ",
                try:
                    input = raw_input()
                except KeyboardInterrupt: # Ctrl-C
                    print "Quitting"
                    exit()
            
            if input.lower() in yes:
                # Proceed as normal, but make a note of this failure.
                settings['script'] = False
            else:
                print "Quitting"
                exit()
            
    else:
        output.output_screen()
    
    
    
    # Now redefine sys.stdout to be output.all, so that by default all output 
    # is both printed and logged. Parts of the system which need to print to 
    # output.short or output.full can do so manually.
    sys.stdout = output.all
    
    
    print 
    print "Autotuning System".center(80)
    print version.center(80)
    print 
    
    
    
    
    # Set the working directory to that of the config file
    # This means that commands in the config file can be written
    # RELATIVE TO THE CONFIG FILE
    # and this program will respect this. The config writer need not assume 
    # where the program will be run from, only where the config is stored.
    path = os.path.dirname(os.path.realpath(configFile))
    os.chdir(path)
    
    
    if(True):
        print "Retrieved settings from config file:"
        print
        print "Variables:\n" + settings['vartree']
        print
        
        print "Displayed as a tree:\n"
        print treeprint_str(settings['vartree'])
        
        print "Possible values:\n" + strVarVals(settings['possValues'])
        print
        
        for opt in ['compile', 'test', 'clean']:
            if opt in settings and settings[opt] is not None:
                print opt + ": \n" + str(settings[opt]) + "\n"
        
        #print 
    
    
    
    
    # Set up the Evaluator
    # This performs the actual testing, by running and timing tests.
    evaluator = Evaluator(settings['compile_mkStr'], 
                          settings['test_mkStr'], settings['custom_fom'],
                          settings['clean_mkStr'], 
                          settings['repeat'], 
                          settings['aggregator'])
    
    
    
    # Set up the optimiser
    # This runs the recursive optimisation algorithm over the variable tree.
    test = Optimisation(settings['vartree'], settings['possValues'], evaluator)
    
    
    if(settings['optimal'] == 'min'):
        test.minimiseScore()
    if(settings['optimal'] == 'max'):
        test.maximiseScore() 
    
    
    # Let them know how many test will be needed.
    print "Number of tests to be run: " + str(test.testsRequired())
    if(settings['repeat'] > 1):
        print "(with " + str(settings['repeat']) + " repetitions each)"
    print 
    print
    
    
    # Start timing the testing
    execution_start = time.time()
    
    # Running the tests will take a long time, if they quit part way through,
    # then catch this, and write what we know to the log.
    try:
        
        test.calculateOptimum()
        
    except KeyboardInterrupt:
        # Interrupted by Ctrl+C
        print "\n\nQuitting Tuner"
        
        # Write the CSV log file, if needed
        if len(evaluator.log) > 0:
            if settings['log'] is not None:
                b = writeCSV(evaluator.log, get_variables(settings['vartree']), settings['possValues'], settings['log'])
                if b:
                    print "A partial testing log was saved to '" + settings['log'] + "'"
                else:
                    print "Failed to write CSV log file."
            
        
        exit()
    
    
    # Finish timing execution
    execution_stop = time.time()
    
    
    if not test.successful():
        
        print
        print "Not enough evaluations could be performed."
        print "There were too many failures."
        
    else:
        
        # If requested, run some additonal tests to get the parameter importance data.
        if settings['importance'] is not None:
            importance_evaluator = Evaluator(settings['compile_mkStr'], 
                settings['test_mkStr'], settings['custom_fom'],
                settings['clean_mkStr'], settings['repeat'], 
                settings['aggregator'], 
                evaluator) # Pass evaluator so no tests are repeated.
            
            
            additional_start = time.time()
            
            parameter_importance(importance_evaluator, settings, test.optimalValuation())
            
            additional_stop = time.time()
            
        
        
        print
        print settings['optimal'].capitalize() + "imal valuation:" # Minimal or Maximal
        print strVarVals(test.optimalValuation(), ", ")
        print settings['optimal'].capitalize() + "imal Score:" # Minimal or Maximal
        print test.optimalScore()
        print "The system ran %d tests, taking %s." % (test.numTests(), "%dm%.2fs" % divmod(execution_stop-execution_start, 60))
        if settings['importance'] is not None:
            print "(and %d additional tests%s)" % (importance_evaluator.testsRun, (", taking %dm%.2fs" % divmod(additional_stop-additional_start, 60) if importance_evaluator.testsRun > 0 else ""))
        
        
    
    
    # Check for any failures during the evaluations
    if(len(evaluator.failures) > 0):
        print
        print "FAILURES:"
        for f in evaluator.failures:
            print "    " + f[0]
            print "    " + strVarVals(f[1], ", ")
            print
    
    
    
    # Assuming there were some tests performed, write the log file
    if len(evaluator.log) > 0:
        if settings['log'] is not None:
            b = writeCSV(evaluator.log, get_variables(settings['vartree']), settings['possValues'], settings['log'])
            if b:
                print "A testing log was saved to '" + settings['log'] + "'"
            else:
                print "Failed to write CSV log file."
        
    
    if (settings['importance'] is not None) and (len(importance_evaluator.log) > 0):
        if True:
            b = writeCSV(importance_evaluator.log, get_variables(settings['vartree']), settings['possValues'], settings['importance'])
            if b:
                print "Additional data was saved to '" + settings['importance'] + "'"
            else:
                print "Failed to write parameter importance data."
    
    
    
    
    # If requested, the script file will already be written by output.py
    # So report this here.
    if settings['script'] is not None:
        # Remember, we checked that it opened OK at the beginning.
        if settings['script'] is not False:
            print "A testing transcript was written to '" + settings['script'] + "'"
        else:
            print "Failed to write a script file."
        




################################################################################





def parameter_importance(importance_evaluator, settings, optimal):

    print >>output.full, "\n\n"
    print "Additional tests to check parameter importance:"
    print >>output.full
    
    vars = get_variables(settings['vartree'])
    possValues = settings['possValues']
    
    
    tests = []
    for var in vars:
        for val in possValues[var]:
            t = dict(optimal) # copy
            t[var] = val
            #print t
            if t not in tests:
                tests.append(t)
    
    
    importance_evaluator.evaluate(tests)
    
    if importance_evaluator.testsRun == 0:
        print "(None required)"
    





# Actually run the script ######################################################

if __name__ == '__main__':
    main()
    
    


