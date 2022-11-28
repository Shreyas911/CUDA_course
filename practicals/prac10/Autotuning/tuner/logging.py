"""
Autotuning System

logging.py

Used to output a CSV log of the testing, which is logged by the Evaluator.
"""

# Write output file
import csv


# Writes a .csv file of the testing process.
# Returns whether this was successful.
def writeCSV(log, vars, possValues, filename):
    
    try:
        with open(filename, 'wb') as f:
            
            writer = csv.writer(f)
            
            # Number of results per test
            nResults = 0
            for t in log.values():
                nResults = max(nResults, len(t.results))
            
            # Create title line
            writer.writerow(["TestNo"] + vars + ["Score_"+str(n) for n in range(1,nResults+1)] + ["Score_Overall"])
            
            # Create each row
            for k, t in sorted(log.iteritems()):
                
                # Add test no.
                l = [str(t.testId)]
                
                # Add variable values
                for v in vars:
                    if v in t.valuation:
                        l.append(str(t.valuation[v]))
                    else:
                        l.append('')
                
                # Add scores
                l += [str(x) for x in t.results]
                l += [""] * (nResults - len(t.results))
                
                # Add overall score
                if t.overall is None:
                    l.append('')
                else:
                    l.append(str(t.overall))
                
                # Finished line
                writer.writerow(l)
            
            # Done
            
    except IOError:
        return False
    
    return True




if __name__ == "__main__":
    print __doc__
    

