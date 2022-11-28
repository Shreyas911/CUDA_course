# looping_plot.plt
# Automatically generated gnuplot file to display the results of testing.
# 
# For PNG output, use something like the following:
# $ gnuplot
# gnuplot> set terminal png large size 800, 1200
# gnuplot> set output 'looping_plot.png'
# gnuplot> load 'looping_plot.plt'
# gnuplot> exit
# 


reset

set datafile separator ","
set datafile missing ""

unset xlabel
set ylabel "Score"
unset key
set title "Tuning Results"



set xrange [0:9]
set yrange [0.002157020568:1.85518307686]

set format x ""

set xtics 1, 1, 8




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
1, 0.267271995544
2, 0.532776117325
3, 1.06207895279
4, 1.58980298042
5, 0.267539024353
6, 0.531336069107
7, 1.05993795395
8, 1.59015583992
e
1, 0.266932010651
2, 0.532402038574
3, 1.06172394753
4, 1.58957695961
5, 0.26687502861
6, 0.53103685379
7, 1.05975294113
8, 1.58744287491
e
1, 0.267271995544
2, 0.532776117325
3, 1.06207895279
4, 1.58980298042
5, 0.267539024353
6, 0.531336069107
7, 1.05993795395
8, 1.59015583992
e
1, 0.271310806274
2, 0.533079147339
3, 1.07352185249
4, 1.5902569294
5, 0.268051862717
6, 0.534675121307
7, 1.06027603149
8, 1.59046506882
e





set border 27
set tmargin 0
unset xtics
unset title
set grid ytics lt 1 lc rgb "#cccccc"
set grid layerdefault
set grid noxtics



# Plot the graph for variable OPTLEVEL
set size 1, 0.15
set origin 0,0.45

set ylabel "OPTLEVEL"
set yrange [0:3]
set ytics ("-O0" 1, "-funroll-loops" 2)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 1
5, 2
6, 2
7, 2
8, 2
e




# Plot the graph for variable XLOOP
set size 1, 0.2
set origin 0,0.25

set ylabel "XLOOP"
set yrange [0:4]
set ytics ("5000" 1, "10000" 2, "20000" 3)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 2
3, 3
4, 3
5, 1
6, 2
7, 3
8, 3
e




# Plot the graph for variable YLOOP
set size 1, 0.15
set origin 0,0.1

set ylabel "YLOOP"
set yrange [0:3]
set ytics ("12000" 1, "18000" 2)



set xlabel "Test No."
set xtics out
set xtics nomirror
set format x


set xtics 1, 1, 8


#set bmargin



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 2
5, 1
6, 1
7, 1
8, 2
e






unset multiplot
reset

