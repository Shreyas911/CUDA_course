#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void gold_match(unsigned int *text, unsigned int *words,
                int *matches, int nwords, int length)
{
  unsigned int word;

  for (int l=0; l<length; l++) {
    for (int offset=0; offset<4; offset++) {
      if (offset==0)
        word = text[l];
      else
        word = (text[l]>>(8*offset)) + (text[l+1]<<(32-8*offset)); 

      /*
      putchar(word&255);
      putchar((word>>8)&255);
      putchar((word>>16)&255);
      putchar((word>>24)&255);
      printf("\n");
      */

      for (int w=0; w<nwords; w++) {
        matches[w] += (word==words[w]);
	/*
        if (word==words[w]) {
          printf(" match: l=%d, w=%d \n",l,w);
	  matches[w]++;
        }
	*/
      }
    }
  }
}
