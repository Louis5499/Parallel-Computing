#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int rank, size;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // the total number of process
  MPI_Comm_size(MPI_COMM_WORLD, &size); // the rank (id) of the calling process

	unsigned long long global_sum;

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	#pragma omp parallel for schedule(guided, 30) reduction(+:pixels)
	for (unsigned long long x = rank; x < r; x+=size) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		if (pixels % 100000 == 0) pixels %= k;
	}

	MPI_Reduce(&pixels, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("%llu\n", (4 * global_sum) % k);
	}
	// printf("%llu\n", (4 * pixels) % k);

	MPI_Finalize();
}
