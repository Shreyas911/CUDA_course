UTILITIES

These utilities provide different methods to visualise and analyse the 
results of the tuning process.


output_gnuplot.py
        This script converts a CSV log file into a gnuplot PLT file. This PLT 
        file can be used with the gnuplot plotting program to produce a 
        detailed graph of the testing process. If required, the PLT file can 
        be modified by hand. 
        
        The -h or --help option prints some usage information.
        The '-r' or '--reference' option allows a reference score to be plotted 
        for comparison with the tuner's results.
        
        Usage: ./output_gnuplot.py [-h] [-r SCORE] mylog.csv myplot.plt


output_screen.py
        This script reads a CSV log file and produces a graph displayed on the 
        screen. This can then be saved if needed. The 'matplotlib' python 
        library is required, which may not be installed by default. 
        
        The -h or --help option prints some usage information.
        The '-r' or '--reference' option allows a reference score to be plotted 
        for comparison with the tuner's results. 
        The -s or --stddev option adds the standard deviation of multiple test 
        repetitions from the mean to the plot.
        
        Usage: ./output_screen [-h] [-r SCORE] [-s] mylog.csv


csv_plot.m
        This is a MATLAB program which can be used to display a graph of the 
        testing process.
        
        Usage: Modify the file as needed.





FILES TO IGNORE

common.py - Defines some common finctions of the Python utilities 
            (in particular, the functions to read in the CSV files).


