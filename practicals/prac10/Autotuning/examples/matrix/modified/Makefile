
CC = gcc
CFLAGS = -lm

matrix: matrix.c
	$(CC) $(CFLAGS) -o matrix matrix.c \
					-D BLOCK_I=$(BLOCK_I) \
					-D BLOCK_J=$(BLOCK_J) \
					-D BLOCK_K=$(BLOCK_K)
