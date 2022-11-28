"""
Autotuning System

output.py

Controls what kind of output is produced by the system and where it is sent.
(i.e. to a log file, to the screen or ignored)
"""

# Access stdout
import sys



# This module defines three file-like (i.e. with a 'write' method) objects:
# all, short, full
# These are used to write output of different sorts.
# writing to 'all' puts text into both the 'short' and 'full' outputs.
# The 'short' and 'full' outputs are used to give different levels of detail.

# The INFORMATION in the 'full' output should be a superset of 'short'.
# They are different in case you want different formatting, etc.

# The objects are only defined once you call one of the output_* functions.






################################################################################


def output_screen():
    # Output everything to the screen, this is the default.
    global short, full, all
    
    short = WriteNull()
    full = sys.stdout
    all = full
    

def output_production(log_file):
    # "Production mode", write the short output to the screen and the full 
    # output to a log file.
    global short, full, all
    
    try:
        file = open(log_file, 'w')
    except IOError:
        return False
    
    short = sys.stdout
    full = file
    all = WriteMult(short, full)
    
    return True


def output_verbose(log_file):
    # Print the full output to the screen and log it to a file.
    # Ignore the shot output.
    global short, full, all
    
    try:
        file = open(log_file, 'w')
    except IOError:
        return False
    
    short = WriteNull()
    full = WriteMult(sys.stdout, file)
    all = full
    
    return True






################################################################################





class WriteMult:
    """
    WriteMult objects are writeable objects which simply pass the write() 
    call on to all of its arguments.
    """
    
    def __init__(self, *args):
        self.writers = args
        
        if not all([callable(getattr(x, 'write', None)) for x in self.writers]):
            # Then one of the objects passed doesn't privide a write() method.
            raise TypeError("An object passed to WriteMult is not writeable.")
    
    def write(self, string):
        for x in self.writers:
            x.write(string)
    
    def flush(self):
        for x in self.writers:
            x.flush()



class WriteNull:
    """
    WriteNull objects are writeable, but simply ignore any data written.
    """
    
    def __init__(self):
        pass
    
    def write(self, string):
        pass
    
    def flush(self):
        pass








################################################################################



if __name__ == "__main__":
    print __doc__

