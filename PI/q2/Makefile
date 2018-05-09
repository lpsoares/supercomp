CFLAGS=--std=c99 -pedantic -Wall -fopenmp

all: main.o lin_regression.o
	gcc ${CFLAGS} lin_regression.o main.o -o q2

main.o: main.c
	gcc ${CFLAGS} -c main.c

lin_regression.o: lin_regression.c
	gcc ${CFLAGS} -c lin_regression.c