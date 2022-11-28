"""
Autotuning System

tune_conf.py

Sets up the configuration for the optimisation. The settings (variable names, 
testing methods, etc.) are read from a configuration file provided.
"""


from ConfigParser import RawConfigParser
from vartree import get_variables
from helpers import avg, med


# The settings function takes the name of a configuration file, and returns the 
# settings to be used for the optimisation.
def get_settings(configFile):
    
    
    # The settings dictionary will contain the configuration data.
    settings = {}
    
    
    
    # Read Configuration File ##################################################

    config = RawConfigParser()

    config.read(configFile)


    # Must have the following sections:
    # [variables], [values], [testing], [scoring], [output] 
    if not(config.has_section('variables') 
            and config.has_section('values') 
            and config.has_section('testing')
            and config.has_section('scoring')
            and config.has_section('output')):
        print "Config file does not contain all the sections:"
        print "[variables], [values], [testing], [scoring], [output]"
        exit()
    
    
    # Must have option 'variables' in [variables]
    if not(config.has_option("variables", "variables")):
        print "Config file does not contain the option 'variables' in section [variables]."
        exit()
    
    
    # Get variable tree
    varTree = config.get("variables", "variables")
    
    settings['vartree'] = varTree
    
    variables = get_variables(varTree)
    
    
    # Must have the correct variables defined in [values]
    if not(all([config.has_option("values", v) for v in variables])):
        print "Config file does not contain possible values (in [values]) for all the variables defined in [variables]."
        exit()
    
    
    # possValues is a dictionary keyed by variable names [strings] with values which are lists of possible values [also string]
    possValues = {}
    
    for thisVar in variables:
        possValues[thisVar] = [x.strip() for x in config.get("values", thisVar).split(",")]

    settings['possValues'] = possValues
    
    
    
    
    
    # Set whichever of 'compile', 'test', 'clean' are present.
    # Also build functions to create the actual compile (etc.) commands to run.
    # Given a test ID n and a mapping of variables to values, returns an 
    # executable string (i.e. a shell command).
    settings['compile']   = settings['compile_mkStr']   = None
    settings['test']       = settings['test_mkStr']       = None
    settings['clean']    = settings['clean_mkStr']    = None
    
    if(config.has_option("testing", "compile")):
        settings['compile'] = config.get('testing', 'compile')
        
        def compiler_mkStr(n, varDict):
            s = settings['compile'].replace("%%ID%%", str(n))
            for varName, varVal in varDict.iteritems():
                s = s.replace("%" + varName + "%", str(varVal))
            return s
        
        settings['compile_mkStr'] = compiler_mkStr
    
    
    if(config.has_option('testing', 'test')):
        settings['test'] = config.get('testing', 'test')
        
        def test_mkStr(n, varDict):
            s = settings['test'].replace("%%ID%%", str(n))
            for varName, varVal in varDict.iteritems():
                s = s.replace("%" + varName + "%", str(varVal))
            return s
        
        settings['test_mkStr'] = test_mkStr
    else:
        print "Config file does not contain option 'test' in section [testing]."
        exit()
    
    
    if(config.has_option('testing', 'clean')):
        settings['clean'] = config.get('testing', 'clean')
        
        def cleanup_mkStr(n, varDict):
            s = settings['clean'].replace("%%ID%%", str(n))
            for varName, varVal in varDict.iteritems():
                s = s.replace("%" + varName + "%", str(varVal))
            return s
        
        settings['clean_mkStr'] = cleanup_mkStr
    
    
    
    
    # Check if they have chosen to maximise or minimise the FOM.
    if(config.has_option('scoring', 'optimal')):
        if(config.get('scoring','optimal').lower() in ['max_time', 'min_time', 'max', 'min']):
            settings['optimal'] = config.get('scoring','optimal')[0:3].lower()
            settings['custom_fom'] = len(config.get('scoring','optimal')) == 3
        else:
            print "Config file contains an invalid setting for 'optimal' in section [scoring]."
            exit()
    else:
        # Default to min_time
        settings['optimal'] = 'min'
        settings['custom_fom'] = False
    
    
    
    # Check if they have set a number of tests to be run.
    if(config.has_option('scoring', 'repeat')):
        tmp = config.get('scoring','repeat').partition(',')
        
        try:
            settings['repeat'] = int(tmp[0])
            if settings['repeat'] < 1:
                raise ValueError("Option 'repeat' must be at least 1.")
        except ValueError:
            print "Config file contains an invalid setting for 'repeat' in section [scoring]."
            exit()
        
        if tmp[1] == '':
            # Then they chose to only specify the number of repetitions
            # So default to 'min'
            settings['overall'] = 'min'
            settings['aggregator'] = min
        else:
            # They specified some aggregate
            if tmp[2].lower().strip() in ['max', 'min', 'med', 'avg']:
                settings['overall'] = tmp[2].lower().strip()
                settings['aggregator'] = {'max': max, 'min': min, 'med': med, 'avg': avg}[settings['overall']]
            else:
                print "Config file contains an invalid setting for 'repeat' in section [scoring]."
                exit()
            
    else:
        # Default to 1
        settings['repeat'] = 1
        # Default to min
        settings['overall'] = 'min'
        settings['aggregator'] = min
    
    
    
    
    
    # Check if they want to log tests.
    if(config.has_option('output', 'log')):
        settings['log'] = config.get('output', 'log')
        
    else:
        # Default to None
        settings['log'] = None
    
    
    # Check if they want to keep a testing script.
    if(config.has_option('output', 'script')):
        settings['script'] = config.get('output', 'script')
        
    else:
        # Default to None
        settings['script'] = None
    
    
    
    # Check if they want to perform extra tests for "parameter importance".
    if(config.has_option('output', 'importance')):
        settings['importance'] = config.get('output', 'importance')
        
    else:
        # Default to None
        settings['importance'] = None
    
    
    
    
    
    return settings














if __name__ == "__main__":
    print __doc__
    
    
