"""
Autotuning System

vartree.py

Defines the VarTree class.
Provides a parser, vt_parse, for converting strings to instances of VarTree.
"""


# VarTree parser generated with wisent
from vartree_parser import Parser
# Built in regex based lexer
from re import Scanner




# The VarTree data type ########################################################

class VarTree:
    # vars is a list of strings of variable names at the current node
    # subtrees is a list of VarTree which are the children of the node
    
    def __init__(self, vars, subtrees):
        self.vars = vars
        self.subtrees = subtrees
    
    def __str__(self): # Convert the VarTree to a string representastion
        allStrs = self.vars + [st.__str__() for st in self.subtrees]
        return "{" + ", ".join(allStrs) + "}"
    
    def flatten(self): # Return a list of all the variables in the tree
        return self.vars + self.flattenchildren()
    
    def flattenchildren(self): # Return a list of all variables in child subtrees
        return sum([st.flatten() for st in self.subtrees], [])


################################################################################




# Some example VarTrees, used for testing ######################################

# {A, B, C, D}
sample0 = VarTree(["A", "B", "C", "D"], [])

# {A, B, {C, D}, {E, F}}
sample1 = VarTree(['A','B'], [VarTree(['C','D'],[]), VarTree(['E','F'],[])])

# {A, B, {I, {C, D}, {E, F}}, {G, H}}
sample2 = VarTree(["A","B"], [VarTree(["I"],[VarTree(["C","D"],[]), VarTree(["E","F"],[])]), VarTree(["G","H"],[])])


################################################################################




# A helper function used to return a list of the variable names used in a (string) VarTree
def get_variables(s):
    return vt_parse(s).flatten()




# The Lexer/Parser #############################################################


def vt_parse(str):
    
    # We'll memoise this function so several calls on the same input don't 
    # require re-parsing.
    
    if(str in vt_parse.memory):
        return vt_parse.memory[str]
    
    
    # Use the built in re.Scanner to tokenise the input string.
    
    def s_lbrace(scanner, token):  return ("LBRACE", token)
    def s_rbrace(scanner, token):  return ("RBRACE", token)
    def s_comma(scanner, token):   return ("COMMA",  token)
    def s_varname(scanner, token): return ("VAR",    token)
    
    scanner = Scanner([
        (r'{', s_lbrace),
        (r'}', s_rbrace),
        (r',', s_comma),
        (r'[a-zA-Z_]\w*', s_varname),
        (r'\s+', None)
    ])
    
    tokens = scanner.scan(str)
    
    # tokens is a pair of the tokenised string and any "uneaten" part.
    # check the entire string was eaten.
    
    if(tokens[1] != ''):
        print "Could not read the variable tree given:"
        print str
        #print "could not lex: " + tokens[1].__str__()
        exit()
    
    
    tokens = tokens[0] # Just the list of tokens.
    
    p = Parser()
    try:
        tree = p.parse(tokens)
    except p.ParseErrors, e:
        print "Could not read the variable tree given:"
        print str
        exit()
    
    
    
    # A function converting the parse tree to a VarTree.
    def pt_to_vt(tree):
        
        # If the current node is a VARTREE,
        # then create a new VarTree object and fill it with the children of this node.
        #
        # If the current node is not a VARTREE, then something has gone wrong.
        
        def is_var(t):
            return t[0] == "VAR"
        
        def is_vt(t):
            return t[0] == "VARTREE" or t[0] == "VARTREE_BR"
        
        if is_vt(tree):
            
            vars = filter(is_var, tree[1:])
            vars = map(lambda t: t[1], vars)
            
            children = filter(is_vt, tree[1:])
            children = map(pt_to_vt, children)
            
            return VarTree(vars,children)
            
        else:
            return None # Should not be reached


    
    
    # Put the result in the memoisation table.
    vt_parse.memory[str] = pt_to_vt(tree)
    
    
    # Check nothing went wrong
    if vt_parse.memory[str] is None:
        print "Could not read the variable tree given:"
        print str
        #print "error in conversion from parse tree to vartree"
        exit()
    
    
    # Finally, check there is no repettition of variables in the VarTree.
    # Each variable can appear at most once.
    def hasDups(xs):
        return len(set(xs)) != len(xs)
    
    if hasDups(vt_parse.memory[str].flatten()):
        print "A variable was repeated in the variable tree."
        print "Variables can only appear once."
        exit()
    
    
    return vt_parse.memory[str]


# Define the memoisation table, which will be satic inside vt_parse
vt_parse.memory = {}






################################################################################






# A tree printer for VarTree ###################################################



# {A, B, {I, {C, D}, {E, F}}, {G, H}}
#
#          {A, B}
#            |
#        +---+--------+
#        |            |
#       {I}         {G, H}
#        |
#   +----+---+
#   |        |
# {C, D}   {E, F}


# The following tree is a good test, which demonstrates many (all?) cases:
# {A, B, {C, D, {CC}, {DD}, {EE}}, {E, F}, {G, H, {I, J}}}



# N.B. this could easily be wider than the terminal.
def treeprint(vt):
    
    return "\n".join(print_vt(vt)) + "\n"
    

def treeprint_str(s):
    return treeprint(vt_parse(s))



# Trees are represented as a list of lines
def print_vt(vt):
    
    char = chars_to_use()
    
    if vt.subtrees: # Recursive case
        
        # Recursively get subtrees
        subtrees = map(print_vt, vt.subtrees)
        
        # find the max height of a subtree
        subtreeheight = len(max(subtrees, key=len))
        
        # pads a subtree to be subtreeheight layers tall and then to be square
        # (no ragged edge)
        def padout(st):
            
            # Add empty lines to the subtree until it is subtreeheight layers tall
            stheight = len(st)
            
            newlines = [""] * (subtreeheight - stheight)
            
            st2 = st + newlines
            
            # Pad each line to be the same width
            linewidth = len(max(st2, key=len))
            
            st3 = [line + (" " * (linewidth - len(line))) for line in st2]
            
            return st3
        
        subtrees = map(padout, subtrees)
        
        
        # Add connecting bars to the top of each subtree
        subtrees = [[char['vert'].center(len(st[0]))] + st for st in subtrees]
        
        
        # Stick the subtrees together
        tree = subtrees[0]
        
        for st in subtrees[1:]:
            # Add st to tree, with a bit of padding.
            
            tree = [tline + " " + app for (tline, app) in zip(tree, st)]
        
        
        
        # Add the connecting branches above the subtrees
        
        # The width of the whole tree
        fullwidth = len(tree[0])
        
        
        # Generate the wide connecting branch
        # By looking at the top line of tree (which is now the upwards pointing 
        # connecting branches) and basically copying it.
        connectingbranch = ""
        numencountered = 0
        
        for c in tree[0]:
            if c == char['vert']:
                if numencountered == 0:
                    connectingbranch += char['left']
                elif numencountered == len(subtrees)-1:
                    connectingbranch += char['right']
                else:
                    connectingbranch += char['down']
                numencountered += 1
            else:
                if numencountered <= 0 or numencountered >= len(subtrees):
                    connectingbranch += " "
                else:
                    connectingbranch += char['horiz']
        
        
        
        
        # Add this node on the very top
        
        topline = VarTree(vt.vars, []).__str__().center(fullwidth)
        topconnector = char['vert'].center(fullwidth)
        
        
        
        # Add one final "+" at the point in connectingbranch which will connect upwards.
        # Again, we will cheat a little by copying the position of the | in topconnector.
        midpoint = topconnector.find(char['vert'])
        
        if len(subtrees) == 1:
            ch = char['vert']
        elif connectingbranch[midpoint] == char['horiz']:
            ch = char['up']
        else:
            ch = char['cross']
        
        connectingbranch = connectingbranch[:midpoint] + ch + connectingbranch[midpoint+1:]
        
        
        
        
        
        # Put it all together to finish
        
        tree = [topline, topconnector, connectingbranch] + tree
        
        
        return tree
        
        
    else: # Base case
        
        return [" " + vt.__str__() + " "]
        




# Selects the box-drawing characters to be used for drawing trees.
def chars_to_use():
    char_ascii = {
        'horiz': '-',  # A horizontal bar
        'vert':  '|',  # A vertiacl bar
        'cross': '+',  # A four-way connector
        'left':  '+',  # A south/east connector (left end of bar)
        'right': '+',  # A south/west connector (right end of bar)
        'down':  '+',  # A downwards, T shaped connector: east/south/west
        'up':    '+'   # An upwards connector, east/north/west
        }
    
    char_test_1 = {
        'horiz': '=',  # A horizontal bar
        'vert':  '!',  # A vertiacl bar
        'cross': '*',  # A four-way connector
        'left':  'r',  # A south/east connector (left end of bar)
        'right': '`',  # A south/west connector (right end of bar)
        'down':  'T',  # A downwards, T shaped connector: east/south/west
        'up':    '^'   # An upwards connector, east/north/west
        }
    
    char_unicode = {
        'horiz': u'\u2500',  # A horizontal bar
        'vert':  u'\u2502',  # A vertiacl bar
        'cross': u'\u253C',  # A four-way connector
        'left':  u'\u250C',  # A south/east connector (left end of bar)
        'right': u'\u2510',  # A south/west connector (right end of bar)
        'down':  u'\u252C',  # A downwards, T shaped connector: east/south/west
        'up':    u'\u2534'   # An upwards connector, east/north/west
        }
    
    char_unicode_thick = {
        'horiz': u'\u2501',  # A horizontal bar
        'vert':  u'\u2503',  # A vertiacl bar
        'cross': u'\u254B',  # A four-way connector
        'left':  u'\u250F',  # A south/east connector (left end of bar)
        'right': u'\u2513',  # A south/west connector (right end of bar)
        'down':  u'\u2533',  # A downwards, T shaped connector: east/south/west
        'up':    u'\u253B'   # An upwards connector, east/north/west
        }
    
    char_unicode_double = {
        'horiz': u'\u2550',  # A horizontal bar
        'vert':  u'\u2551',  # A vertiacl bar
        'cross': u'\u256C',  # A four-way connector
        'left':  u'\u2554',  # A south/east connector (left end of bar)
        'right': u'\u2557',  # A south/west connector (right end of bar)
        'down':  u'\u2566',  # A downwards, T shaped connector: east/south/west
        'up':    u'\u2569'   # An upwards connector, east/north/west
        }
    
    char_unicode_curve = {
        'horiz': u'\u2500',  # A horizontal bar
        'vert':  u'\u2502',  # A vertiacl bar
        'cross': u'\u253C',  # A four-way connector
        'left':  u'\u256D',  # A south/east connector (left end of bar)
        'right': u'\u256E',  # A south/west connector (right end of bar)
        'down':  u'\u252C',  # A downwards, T shaped connector: east/south/west
        'up':    u'\u2534'   # An upwards connector, east/north/west
        }
    
    return char_ascii # Don't know how to intelligently choose between these..?


################################################################################





if __name__ == "__main__":
    print __doc__
