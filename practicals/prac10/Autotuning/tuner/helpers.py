"""
Autotuning System

helpers.py

A few helper functions used in the system, colected here for convenience.
"""



# Creates a string giving variable/value pairs.
def strVarVals(d, sep="\n"):
    return sep.join([str(var) + " = " + str(val) for var,val in sorted(d.items())])




# Returns a string ordinal of the argument.
def ordinal(n):
    if 10 <= n % 100 < 20:
        return str(n) + 'th'
    else:
        return  str(n) + {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(n % 10, 'th')




# Return the cross product of a list of lists
def crossproduct(xss):
    cp = [[]]
    for xs in xss:
        cp = [xs2 + [x] for x in xs for xs2 in cp]
    return cp




# Return the median of a list
def med(xs):
    ys = sorted(xs)
    return (ys[len(ys)/2]) if bool(len(ys)%2) else (ys[len(ys)/2] + ys[len(ys)/2 -1])/2.0



# Return the average of a list
def avg(xs):
    return sum(xs) / len(xs)








if __name__ == "__main__":
    print __doc__


