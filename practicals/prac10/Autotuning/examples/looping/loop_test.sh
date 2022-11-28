#!/bin/bash

# First argument is the executable to time
# Silently runs the program and gives the sum of user and system times
# as reported by the time utility

#time $1




# Send output and errors (incl that of time) to stdout, so we catch it all
OUTPUT=$((time $1) 2>&1)

#echo $OUTPUT


# Get the timing info from the end of $OUTPUT

AWK_CODE='{user=$(NF-2); sys=$NF; user=split(user, u, "m"); sys=split(sys, s, "m"); tot=u[1]*60; tot+=s[1]*60; tot+=u[2]+s[2]; printf "%5s", tot }'

TIME=$( echo $OUTPUT | awk "$AWK_CODE" )

echo $TIME

