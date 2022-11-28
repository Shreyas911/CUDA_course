/* 
 * Autotuning System
 * 
 * matrix.c
 * AUTO-TUNING VERSION
 * 
 * A simple blocked matrix-matrix multiply algorithm,
 * partitioned by the block size. This is used as an example and 
 * demonstration of the auto-tuner.
 */

#include <stdio.h>
#include <math.h>

/* Define the size of the matrices to work on. */
#define A_COLS 512
#define A_ROWS 512
#define B_COLS 512
#define B_ROWS A_COLS
#define C_COLS B_COLS
#define C_ROWS A_ROWS

/* The block size for the multiplication */
#ifndef BLOCK_I
    #define BLOCK_I 1
#endif
#ifndef BLOCK_J
    #define BLOCK_J 1
#endif
#ifndef BLOCK_K
    #define BLOCK_K 1
#endif

/* For timing the test, repeat the multiplication a number of times */
#define TEST_REP 5


int main(void)
{
    double A[A_ROWS][A_COLS], B[B_ROWS][B_COLS], C[C_ROWS][C_COLS];
    int i, j, k, i_bl, j_bl, k_bl, rep;
    
    printf("Blocked Matrix-Matrix Multiplication\n");
    
    /* Generate some arbitrary sample data. */
    
    for(i=0; i<A_ROWS; i++)
        for(j=0; j<A_COLS; j++)
            A[i][j] = exp(-fabs(i-j));
    
    for(i=0; i<B_ROWS; i++)
        for(j=0; j<B_COLS; j++)
            B[i][j] = exp(-fabs(i-j));
    
    
    /* Blocked Multiplication: C = AB */
    /* Instead of processing an entire row of C at a time, 
     * process in small blocks of dimensions BLOCK_I * BLOCK_J. Elements 
     * required from A and B are also processed in blocks.
     * This should improve local memory reuse. */
    printf("(BLOCK_I = %d, BLOCK_J = %d, BLOCK_K = %d)\n", BLOCK_I, BLOCK_J, 
            BLOCK_K);
    for(rep=0; rep<TEST_REP; rep++){
        
        /* Set C[][] = 0 first */
        for(i=0; i<C_ROWS; i++)
            for(j=0; j<C_COLS; j++)
                C[i][j] = 0;
        
        
        /* Perform C = C + A*B */
        for(i=0; i<C_ROWS; i+= BLOCK_I)
            for(j=0; j<C_COLS; j+= BLOCK_J)
                for(k=0; k<A_COLS; k+= BLOCK_K)
                    for(i_bl=i; i_bl<(i+BLOCK_I) && i_bl<C_ROWS; i_bl++)
                        for(j_bl=j; j_bl<(j+BLOCK_J) && j_bl<C_COLS; j_bl++)
                            for(k_bl=k; k_bl<(k+BLOCK_K) && k_bl<A_COLS; k_bl++)
                                C[i_bl][j_bl] += A[i_bl][k_bl] * B[k_bl][j_bl];
        
    }
    
    /* Use C ... */
    
    return 0;
}
