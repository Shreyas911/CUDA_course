//
// Pattern-matching program
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

// #include <match_kernel.cu>

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void gold_match(unsigned int *, unsigned int *, int *, int, int);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  char *ctext, *cwords[] = {"cuti", "gold", "text", "word"};
  unsigned int  *text,  *words;

  int   length, len, nwords=4, matches[4]={0, 0, 0, 0};

  // read in text for processing

  FILE *fp;
  fp = fopen("match.cu","r");

  length = 0;
  while (getc(fp) != EOF) length++;

  ctext = (char *) malloc(length+4);

  rewind(fp);

  for (int l=0; l<length; l++) ctext[l] = getc(fp);
  for (int l=length; l<length+4; l++) ctext[l] = ' ';

  fclose(fp);

  // define number of words of text, and set pointers

  len  = length/4;
  text = (unsigned int *) ctext;

  // define words for matching

  words = (unsigned int *) malloc(nwords*sizeof(unsigned int));

  for (int w=0; w<nwords; w++) {
    words[w] = ((unsigned int) cwords[w][0])
             + ((unsigned int) cwords[w][1])*256
             + ((unsigned int) cwords[w][2])*256*256
             + ((unsigned int) cwords[w][3])*256*256*256;
  }

  // CPU execution

  gold_match(text, words, matches, nwords, len);

  printf(" CPU matches = %d %d %d %d \n",
         matches[0],matches[1],matches[2],matches[3]);

  // GPU execution




  // Release GPU and CPU memory


  free(ctext);
}
