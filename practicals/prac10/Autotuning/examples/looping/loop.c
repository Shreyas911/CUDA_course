#include <stdio.h>

#ifndef XLOOP
	#define XLOOP 0
#endif
#ifndef YLOOP
	#define YLOOP 0
#endif

int main(){
	
	/* Loop through x and y */
	/* the bounds, XLOOP and YLOOP will be provided by the compiler */
	int x=0,y=0,i=0;
	
	for(x=0; x<XLOOP; x++)
		for(y=0; y<YLOOP; y++)
			i++;
	
	printf("%d x %d = %d\n", x, y, i);
	
	return 0;
}
