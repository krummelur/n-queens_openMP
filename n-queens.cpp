/*
	Generates solutions for the n-queens problem up to n=11.
	open-mp is used for parallelization.

	The program stores the board as a 128-bit int, so 128-bit integers must be
	supported to run it.

	usage: g++ n-queens.cpp -fopenmp -o 8q.bin
		./nq.bin <boardSize> <numThreads>
*/

#include <iostream>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <stdio.h>

#define DEBUG

typedef unsigned __int128 uint128;

//This variable is shared and therefore any modification is a critical section.
//OpenMP must know about this.
int total_boards = 0;
int MAX_N = 11;
int n = 8;
int numThreads = 8;

//Function declarations
double read_timer();
void find_valid_boards();
void tryPositions(int, int, uint128, uint128, int);
void markUnavailablePositions(uint128, uint128, int, int, uint128*, uint128*);
void printBoard(uint128);


int main(int argc, char** args)
{
	n = (argc > 1)? atoi(args[1]) : MAX_N;
	int numThreads = n;
	numThreads = (argc > 2)? atoi(args[2]) : numThreads;
	n = n > MAX_N ? MAX_N : n;

	omp_set_num_threads(numThreads);
	find_valid_boards();
	printf("Search done on %d x %d board, with %d threads.\n", n, n, numThreads);
	printf("boards found: %d\n", total_boards);
}

/* the search is implemented as a depth first search. */
void find_valid_boards() {
	double start, total = 0;

	start = omp_get_wtime();
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
			tryPositions(i,0, 0, 0, 0);
	}
	total = omp_get_wtime() - start;
	printf("Total time: %g sec\n", total);
}

/* tries possible queen placements, based on an initial board state */
void tryPositions(int positionX, int positionY, uint128 availablePositions, uint128 queenPlacements, int queensPlaced) {
	if (queensPlaced == n) {
		#pragma omp critical(total_boards)
		{
		total_boards++;
		#ifdef DEBUG
			printBoard(queenPlacements);
		#endif
		}
		return;
	}

	for (int i = positionX; i < n; i++) {
		for (int j = i == positionX ? 0 : positionY; j < n; j++) {
			if (((availablePositions >> ((n*n-1) - (i*n+j))) & 1) == 0) {
				//We must create copies for the recursive call
				uint128 positionsresult = 0;
				uint128 queensResult = 0;
				#pragma omp task shared(total_boards)
				{
					markUnavailablePositions(availablePositions, queenPlacements, i, j, &positionsresult, &queensResult);
					tryPositions(i+1, j+1, positionsresult, queensResult, queensPlaced + 1);
				}
			}
		}
	}
}

/* marks positions on a board as unavailable, given a new queen position */
void markUnavailablePositions(uint128 current, uint128  currentQueens, int rowNum, int colNum, uint128 * pResult, uint128 * qResult) {
	//Making masks for the row and column makes for more readable code
	uint128 one = 0x1;
	uint128 rowMask = (one << (n*n-1)) >> rowNum;
	uint128 colMask = (one << (n*n-1)) >> colNum;
	uint128 position = colMask >> (rowNum *n);

	//This must be in a loop to allow for different board sizes
	uint128 blockedRowMask = 0;
	for (int i = 0; i < n; i++)
		blockedRowMask = (blockedRowMask | (one << ((n*n-1)-i)));

	//First mark the row as unavailable, it is very easy with the long int representation of the board
	current = current | blockedRowMask >> (rowNum * n);

	//then mark the column as unavailable, is also a similar operation
	for(int i = 0; i < n; i++)
	current = current |  (colMask >> (n*i));

	//Then the diagonals
	for (int i = rowNum, j = 0; i < n; i++, j++) {
		current = current | (position >> j * (n+1)) & (blockedRowMask >> (i * n)) |
			((position >> j * (n-1)) & (blockedRowMask >> (i * n)));
	}
	for (int i = 0, j = 0; i <= rowNum; i++, j++) {
		current = current | (position << j * (n+1)) & (blockedRowMask >> ((rowNum - i) * n)) |
			(position << j * (n-1) & (blockedRowMask >> ((rowNum - i) * n)));
	}

	*pResult = current;
	*qResult = currentQueens | position;
}

void printBoard(uint128 board) {
	uint128 zeroth = 0x1;
	zeroth = zeroth << n*n-1;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d ", ((board << (((n-1)-i) * n + j)) & zeroth) && true);
		}
		printf("\n");
	}
	printf("\n");
}
