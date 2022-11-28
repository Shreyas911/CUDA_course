/* 
 * Autotuning System
 * 
 * matrix.c
 * ORIGINAL VERSION
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
#define BLOCK_I 16
#define BLOCK_J 16
#define BLOCK_K 16


int main(void)
{
    double A[A_ROWS][A_COLS], B[B_ROWS][B_COLS], C[C_ROWS][C_COLS];
    int i, j, k, i_bl, j_bl, k_bl;
    
    printf("Blocked Matrix-Matrix Multiplication\n");
    
    /* Generate some arbitrary sample data. */
    
    for(i=0; i<A_ROWS; i++)
        for(j=0; j<A_COLS; j++)
            A[i][j] = exp(-fabs(i-j));
    
    for(i=0; i<B_ROWS; i++)
        for(j=0; j<B_COLS; j++)
            B[i][j] = exp(-fabs(i-j));
    
    
    /* Set C[][] = 0 first */
    for(i=0; i<C_ROWS; i++)
        for(j=0; j<C_COLS; j++)
            C[i][j] = 0;
    
    
    /* Blocked Multiplication: C = AB */
    /* Instead of processing an entire row of C at a time, 
     * process in small blocks of dimensions BLOCK_I * BLOCK_J. Elements 
     * required from A and B are also processed in blocks.
     * This should improve local memory reuse. */
    printf("(BLOCK_I = %d, BLOCK_J = %d, BLOCK_K = %d)\n", BLOCK_I, BLOCK_J, 
            BLOCK_K);
    
    
    /* Perform C = C + A*B */
    for(i=0; i<C_ROWS; i+= BLOCK_I)
        for(j=0; j<C_COLS; j+= BLOCK_J)
            for(k=0; k<A_COLS; k+= BLOCK_K)
                for(i_bl=i; i_bl<(i+BLOCK_I) && i_bl<C_ROWS; i_bl++)
                    for(j_bl=j; j_bl<(j+BLOCK_J) && j_bl<C_COLS; j_bl++)
                        for(k_bl=k; k_bl<(k+BLOCK_K) && k_bl<A_COLS; k_bl++)
                            C[i_bl][j_bl] += A[i_bl][k_bl] * B[k_bl][j_bl];
    
    /* Use C ... */
    
    return 0;
}
