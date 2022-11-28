# hello_size_plot.plt
# Automatically generated gnuplot file to display the results of testing.
# 
# For PNG output, use something like the following:
# $ gnuplot
# gnuplot> set terminal png large size 800, 1200
# gnuplot> set output 'hello_size_plot.png'
# gnuplot> load 'hello_size_plot.plt'
# gnuplot> exit
# 


reset

set datafile separator ","
set datafile missing ""

unset xlabel
set ylabel "Score"
unset key
set title "Tuning Results"



set xrange [0:31]
set yrange [8463.4:8474.6]

set format x ""

set xtics 5, 5, 30
set xtics add ("" 1 1, "" 2 1, "" 3 1, "" 4 1)
set xtics add ()
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
     "-" using 1:2 with points pt 2 lc rgb "black", \
     "-" using 1:2 with points pt 2 lc rgb "black", \
     "-" using 1:2 with points pt 2 lc rgb "black"
1, 8465.0
2, 8465.0
3, 8465.0
4, 8465.0
5, 8465.0
6, 8465.0
7, 8473.0
8, 8473.0
9, 8473.0
10, 8473.0
11, 8473.0
12, 8473.0
13, 8473.0
14, 8473.0
15, 8473.0
16, 8473.0
17, 8473.0
18, 8473.0
19, 8473.0
20, 8473.0
21, 8473.0
22, 8473.0
23, 8473.0
24, 8473.0
25, 8473.0
26, 8473.0
27, 8473.0
28, 8473.0
29, 8473.0
30, 8473.0
e
1, 8465.0
2, 8465.0
3, 8465.0
4, 8465.0
5, 8465.0
6, 8465.0
7, 8473.0
8, 8473.0
9, 8473.0
10, 8473.0
11, 8473.0
12, 8473.0
13, 8473.0
14, 8473.0
15, 8473.0
16, 8473.0
17, 8473.0
18, 8473.0
19, 8473.0
20, 8473.0
21, 8473.0
22, 8473.0
23, 8473.0
24, 8473.0
25, 8473.0
26, 8473.0
27, 8473.0
28, 8473.0
29, 8473.0
30, 8473.0
e
1, 8465.0
2, 8465.0
3, 8465.0
4, 8465.0
5, 8465.0
6, 8465.0
7, 8473.0
8, 8473.0
9, 8473.0
10, 8473.0
11, 8473.0
12, 8473.0
13, 8473.0
14, 8473.0
15, 8473.0
16, 8473.0
17, 8473.0
18, 8473.0
19, 8473.0
20, 8473.0
21, 8473.0
22, 8473.0
23, 8473.0
24, 8473.0
25, 8473.0
26, 8473.0
27, 8473.0
28, 8473.0
29, 8473.0
30, 8473.0
e
1, 8465.0
2, 8465.0
3, 8465.0
4, 8465.0
5, 8465.0
6, 8465.0
7, 8473.0
8, 8473.0
9, 8473.0
10, 8473.0
11, 8473.0
12, 8473.0
13, 8473.0
14, 8473.0
15, 8473.0
16, 8473.0
17, 8473.0
18, 8473.0
19, 8473.0
20, 8473.0
21, 8473.0
22, 8473.0
23, 8473.0
24, 8473.0
25, 8473.0
26, 8473.0
27, 8473.0
28, 8473.0
29, 8473.0
30, 8473.0
e





set border 27
set tmargin 0
unset xtics
unset title
set grid ytics lt 1 lc rgb "#cccccc"
set grid layerdefault
set grid noxtics



# Plot the graph for variable FOO
set size 1, 0.152
set origin 0,0.448

set ylabel "FOO"
set yrange [0:4]
set ytics ("1" 1, "2" 2, "34" 3)



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
28, 1
29, 2
30, 3
e




# Plot the graph for variable BAR
set size 1, 0.114
set origin 0,0.334

set ylabel "BAR"
set yrange [0:3]
set ytics ("1" 1, "12" 2)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 2
5, 2
6, 2
7, 1
8, 1
9, 1
10, 2
11, 2
12, 2
13, 1
14, 1
15, 1
16, 2
17, 2
18, 2
19, 1
20, 1
21, 1
22, 2
23, 2
24, 2
25, 1
26, 1
27, 1
28, 2
29, 2
30, 2
e




# Plot the graph for variable OPTLEVEL
set size 1, 0.228
set origin 0,0.106

set ylabel "OPTLEVEL"
set yrange [0:6]
set ytics ("-O0" 1, "-O1" 2, "-O2" 3, "-O3" 4, "-Os" 5)



set xlabel "Test No."
set xtics out
set xtics nomirror
set format x


set xtics 5, 5, 30
set xtics add ("" 1 1, "" 2 1, "" 3 1, "" 4 1)
set xtics add ()
set mxtics 5


#set bmargin



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 1
5, 1
6, 1
7, 2
8, 2
9, 2
10, 2
11, 2
12, 2
13, 3
14, 3
15, 3
16, 3
17, 3
18, 3
19, 4
20, 4
21, 4
22, 4
23, 4
24, 4
25, 5
26, 5
27, 5
28, 5
29, 5
30, 5
e






unset multiplot
reset

