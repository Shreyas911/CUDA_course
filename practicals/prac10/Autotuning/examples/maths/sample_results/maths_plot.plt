# maths_plot.plt
# Automatically generated gnuplot file to display the results of testing.
# 
# For PNG output, use something like the following:
# $ gnuplot
# gnuplot> set terminal png large size 800, 1200
# gnuplot> set output 'maths_plot.png'
# gnuplot> load 'maths_plot.plt'
# gnuplot> exit
# 


reset

set datafile separator ","
set datafile missing ""

unset xlabel
set ylabel "Score"
unset key
set title "Tuning Results"



set xrange [0:28]
set yrange [3.4:28.6]

set format x ""

set xtics 5, 5, 25
set xtics add ("" 1 1, "" 2 1, "" 3 1, "" 4 1)
set xtics add ("" 26 1, "" 27 1)
set mxtics 5




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
     "-" using 1:2 with points pt 2 lc rgb "black"
1, 21.0
2, 22.0
3, 23.0
4, 22.0
5, 23.0
6, 24.0
7, 23.0
8, 24.0
9, 25.0
10, 13.0
11, 14.0
12, 15.0
13, 14.0
14, 15.0
15, 16.0
16, 15.0
17, 16.0
18, 17.0
19, 7.0
20, 8.0
21, 9.0
22, 8.0
23, 9.0
24, 10.0
25, 9.0
26, 10.0
27, 11.0
e
1, 21.0
2, 22.0
3, 23.0
4, 22.0
5, 23.0
6, 24.0
7, 23.0
8, 24.0
9, 25.0
10, 13.0
11, 14.0
12, 15.0
13, 14.0
14, 15.0
15, 16.0
16, 15.0
17, 16.0
18, 17.0
19, 7.0
20, 8.0
21, 9.0
22, 8.0
23, 9.0
24, 10.0
25, 9.0
26, 10.0
27, 11.0
e





set border 27
set tmargin 0
unset xtics
unset title
set grid ytics lt 1 lc rgb "#cccccc"
set grid layerdefault
set grid noxtics



# Plot the graph for variable X
set size 1, 0.168
set origin 0,0.432

set ylabel "X"
set yrange [0:4]
set ytics ("1" 1, "2" 2, "3" 3)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 2
3, 3
4, 1
5, 2
6, 3
7, 1
8, 2
9, 3
10, 1
11, 2
12, 3
13, 1
14, 2
15, 3
16, 1
17, 2
18, 3
19, 1
20, 2
21, 3
22, 1
23, 2
24, 3
25, 1
26, 2
27, 3
e




# Plot the graph for variable Y
set size 1, 0.168
set origin 0,0.264

set ylabel "Y"
set yrange [0:4]
set ytics ("4" 1, "5" 2, "6" 3)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 2
5, 2
6, 2
7, 3
8, 3
9, 3
10, 1
11, 1
12, 1
13, 2
14, 2
15, 2
16, 3
17, 3
18, 3
19, 1
20, 1
21, 1
22, 2
23, 2
24, 2
25, 3
26, 3
27, 3
e




# Plot the graph for variable Z
set size 1, 0.168
set origin 0,0.096

set ylabel "Z"
set yrange [0:4]
set ytics ("16" 1, "8" 2, "2" 3)



set xlabel "Test No."
set xtics out
set xtics nomirror
set format x


set xtics 5, 5, 25
set xtics add ("" 1 1, "" 2 1, "" 3 1, "" 4 1)
set xtics add ("" 26 1, "" 27 1)
set mxtics 5


#set bmargin



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
10, 2
11, 2
12, 2
13, 2
14, 2
15, 2
16, 2
17, 2
18, 2
19, 3
20, 3
21, 3
22, 3
23, 3
24, 3
25, 3
26, 3
27, 3
e






unset multiplot
reset

