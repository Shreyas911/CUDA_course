# sample.plt
# Automatically generated gnuplot file to display the results of testing.
# 
# For PNG output, use something like the following:
# $ gnuplot
# gnuplot> set terminal png large size 800, 1200
# gnuplot> set output 'sample.png'
# gnuplot> load 'sample.plt'
# gnuplot> exit
# 


reset

set datafile separator ","
set datafile missing ""

unset xlabel
set ylabel "Score"
unset key
set title "Tuning Results"



set xrange [0:16]
set yrange [9.52107539174:13.5721534253]

set format x ""

set xtics 1, 1, 15




set xtics nomirror
set ytics nomirror
set ytics out
set xtics in

set border 3


set lmargin 12





# MULTIPLOT
# The main graph above, smaller graphs below showing how the variables change.


set size 1,1
set origin 0,0
set multiplot





# Main graph gets 40%, labels, tics etc at bottom get 10%
# Remaining 50% is divided evenly between the variables.


set size 1,0.4
set origin 0,0.6


set bmargin 0

set boxwidth 1
set style fill solid 0.2
set grid front
unset grid


plot "-" using 1:2 with boxes lc rgb "black", \
     "-" using 1:2 with points pt 2 lc rgb "black", \
     "-" using 1:2 with points pt 2 lc rgb "black", \
     "-" using 1:2 with points pt 2 lc rgb "black"
1, 12.8513569832
2, 11.3451521397
3, 11.4516351223
4, 11.0573651791
5, 10.8725659847
6, 10.9468522072
7, 11.0326418877
8, 10.8091490269
9, 10.4505469799
10, 
11, 11.2035610676
12, 10.7626430988
13, 10.2889459133
14, 10.2381851673
15, 10.0998008251
e
1, 12.8513569832
2, 11.3451521397
3, 11.4516351223
4, 11.0573651791
5, 10.8725659847
6, 10.9468522072
7, 11.0326418877
8, 10.8091490269
9, 10.4505469799
10, 
11, 11.2035610676
12, 10.7626430988
13, 10.2889459133
14, 10.2381851673
15, 10.0998008251
e
1, 12.8602671623
2, 11.7126541138
3, 11.6016530991
4, 11.3189618587
5, 10.8975639343
6, 11.8204932213
7, 11.4210488796
8, 10.8452339172
9, 10.5981900692
10, 
11, 11.2549870014
12, 10.8307909966
13, 10.4059588909
14, 10.2516989708
15, 10.1896440983
e
1, 12.9934279919
2, 11.8558809757
3, 11.700740099
4, 11.3285858631
5, 10.9090161324
6, 11.8779690266
7, 11.4792678356
8, 10.8535571098
9, 10.5993440151
10, 
11, 11.8886649609
12, 11.0776929855
13, 10.4554610252
14, 10.3096880913
15, 10.2373991013
e





set border 27
set tmargin 0
unset xtics
unset title
set grid ytics lt 1 lc rgb "#cccccc"
set grid layerdefault
set grid noxtics



# Plot the graph for variable BLOCK_I
set size 1, 0.252
set origin 0,0.348

set ylabel "BLOCK_I"
set yrange [0:6]
set ytics ("4" 1, "8" 2, "16" 3, "32" 4, "64" 5)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 2
3, 3
4, 4
5, 5
6, 1
7, 2
8, 3
9, 4
10, 5
11, 1
12, 2
13, 3
14, 4
15, 5
e




# Plot the graph for variable BLOCK_K
set size 1, 0.084
set origin 0,0.264

set ylabel "BLOCK_K"
set yrange [0:2]
set ytics ("4" 1)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 1
5, 1
6, 1
7, 1
8, 1
9, 1
10, 1
11, 1
12, 1
13, 1
14, 1
15, 1
e




# Plot the graph for variable BLOCK_J
set size 1, 0.168
set origin 0,0.096

set ylabel "BLOCK_J"
set yrange [0:4]
set ytics ("4" 1, "8" 2, "16" 3)



set xlabel "Test No."
set xtics out
set xtics nomirror
set format x


set xtics 1, 1, 15


#set bmargin



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 1
5, 1
6, 2
7, 2
8, 2
9, 2
10, 2
11, 3
12, 3
13, 3
14, 3
15, 3
e






unset multiplot
reset

