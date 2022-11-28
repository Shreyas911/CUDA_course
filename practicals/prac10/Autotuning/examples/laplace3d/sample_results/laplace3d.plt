# laplace3d.plt
# Automatically generated gnuplot file to display the results of testing.
# 
# For PNG output, use something like the following:
# $ gnuplot
# gnuplot> set terminal png large size 800, 1200
# gnuplot> set output 'laplace3d.png'
# gnuplot> load 'laplace3d.plt'
# gnuplot> exit
# 


reset

set datafile separator ","
set datafile missing ""

unset xlabel
set ylabel "Score"
unset key
set title "Tuning Results"



set xrange [0:10]
set yrange [5.90668873786:11.4419052125]

set format x ""

set xtics 1, 1, 9




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
1, 8.98541998863
2, 9.81590294838
3, 8.09569907188
4, 6.69743394852
5, 7.28998184204
6, 10.033272028
7, 7.47861003876
8, 7.64705705643
9, 6.96617293358
e
1, 8.98541998863
2, 9.81590294838
3, 8.09569907188
4, 6.69743394852
5, 7.28998184204
6, 10.033272028
7, 7.47861003876
8, 7.64705705643
9, 6.96617293358
e
1, 8.99301505089
2, 9.84374117851
3, 9.49392795563
4, 9.56936693192
5, 8.45071101189
6, 10.3622369766
7, 8.76150798798
8, 7.680300951
9, 8.99450707436
e
1, 9.78470993042
2, 9.85856199265
3, 9.75308895111
4, 9.63162899017
5, 9.6081161499
6, 10.6511600018
7, 9.70002102852
8, 9.61640405655
9, 10.127120018
e





set border 27
set tmargin 0
unset xtics
unset title
set grid ytics lt 1 lc rgb "#cccccc"
set grid layerdefault
set grid noxtics



# Plot the graph for variable BLOCK_X
set size 1, 0.252
set origin 0,0.348

set ylabel "BLOCK_X"
set yrange [0:4]
set ytics ("32" 1, "64" 2, "128" 3)



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
e




# Plot the graph for variable BLOCK_Y
set size 1, 0.252
set origin 0,0.096

set ylabel "BLOCK_Y"
set yrange [0:4]
set ytics ("2" 1, "4" 2, "6" 3)



set xlabel "Test No."
set xtics out
set xtics nomirror
set format x


set xtics 1, 1, 9


#set bmargin



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
e






unset multiplot
reset

