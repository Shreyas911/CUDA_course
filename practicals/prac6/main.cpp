//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


//
// declare external routine
//

extern
int prac6(int nblocks, int nthreads);

//
// main code
//

int main(int argc, char **argv)
{
  // set number of blocks, and threads per block

  int nblocks  = 2;
  int nthreads = 8;

  // call CUDA routine

  prac6(nblocks,nthreads);

  return 0;
}

 
