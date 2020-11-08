#include <cstdio>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define HIGH 1
#define LOW 0

// TODO: Check 1000 is enough or not?

int comp(const void *a, const void *b) 
{
	// Note that if we simply return the substraction of these two value, the return value may overflow!
	// So the better way is to compare it first and return the value
	// If *(float *)a > *(float *)b, then it returns 1, else return -1.
	// 
	// return(*(float *)a-*(float *)b); 

	// float fa = *(float*) a;
  // float fb = *(float*) b;
  // return (fa > fb) - (fa < fb);
	return (*(float *)a > *(float *)b) - (*(float *)a < *(float *)b);
}


void Merge(int size_a, int size_b, float *data_a, float *data_b, int high_low){
	float *tmp = (float *) malloc(sizeof(float)*size_a);

	// Merge the list 
	if(high_low == LOW) {
		// keep low
		int idx_a = 0, idx_b = 0, idx_out = 0;
		while(idx_a < size_a && idx_b < size_b && idx_out < size_a){
			if(data_a[idx_a] < data_b[idx_b])
				tmp[idx_out++] = data_a[idx_a++];
      else
	    	tmp[idx_out++] = data_b[idx_b++];
    }
		while(idx_out < size_a) tmp[idx_out++] = (idx_a < size_a) ? data_a[idx_a++]:data_b[idx_b++];
	} else {
		int idx_a = size_a-1, idx_b = size_b-1, idx_out = size_a-1;
		while(idx_a >= 0 && idx_b >= 0 && idx_out >= 0){
			if(data_a[idx_a] < data_b[idx_b])
				tmp[idx_out--] = data_b[idx_b--];
			else
				tmp[idx_out--] = data_a[idx_a--];
		}
		while(idx_out >= 0) tmp[idx_out--] = (idx_a >= 0) ? data_a[idx_a--]:data_b[idx_b--];
	}

	// split the data
	for(int i = 0; i < size_a; i++) {
			data_a[i] = tmp[i];
	}
	free(tmp);
}

float data[44739160];
float tempArr[44739160];

int main(int argc, char** argv) {
	MPI_Init(&argc,&argv);
	int rank, size;
	int isFirstTime = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_File f;
	int totalNum = atoll(argv[1]);
	int rc = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	if (rc) {
		printf( "Unable to open file \"temp\"\n" );
		fflush(stdout);
	}

	// Divide data
	int avgNumPerNode = ((int)(totalNum/size));
	// printf("avg: %d ", avgNumPerNode);
	int modNumber = totalNum%size;
	if (totalNum < size) size = totalNum;
	int containNum = (rank < modNumber) ? avgNumPerNode + 1 : avgNumPerNode;
	int offset = 0;
	for (int i=0;i<rank;i++) {
		offset += (i < modNumber) ? (avgNumPerNode + 1) : avgNumPerNode;
	}

	if (rank < size) {
		MPI_File_read_at(f, sizeof(float) * offset, &data, containNum, MPI_FLOAT, MPI_STATUS_IGNORE);
		// printf("rank %d got float: %f\n", rank, data[0]);

		// Self sorting
		// printf("Rank: %d sorting.\n", rank);
		qsort(data, containNum, sizeof(float), comp);
		// printf("Rank: %d Finished sorting.\n", rank);
	}

	// Even-Odd sorting
	int sorted = 0;
	int partner = 0;
	int oddFormer = (rank%2) == 0;
	int root = 0;

	for (int phase=0; phase<=size; phase++) {
		int shouldContinue = 0;
		if (rank < size) {
			if (phase%2 == 0) {
				// Odd phase
				if (oddFormer && rank + 1 < size) partner = rank + 1;
				else if (!oddFormer && rank - 1 >= 0) partner = rank - 1;
				else partner = MPI_PROC_NULL;
			} else {
				// even phase
				if (!oddFormer && rank + 1 < size) partner = rank + 1;
				else if (oddFormer && rank - 1 >= 0) partner = rank - 1;
				else partner = MPI_PROC_NULL;
			}
		} else partner = MPI_PROC_NULL;
		// printf("Phase: %d  Rank: %d Finished selected. partner: %d  \n", phase, rank, partner);

		int partnerNum = (partner < modNumber) ? avgNumPerNode + 1 : avgNumPerNode;

		MPI_Request req1;
		
		if (partner != MPI_PROC_NULL){
			// printf("Rank: %d sending to partner: %d \n", rank, partner);
			MPI_Isend(&data, containNum, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &req1);
			MPI_Irecv(&tempArr, partnerNum, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &req1);
			// printf("Rank: %d receive\n", rank);
			MPI_Wait(&req1, MPI_STATUS_IGNORE);
			if (rank < partner) {
				Merge(containNum, partnerNum, data, tempArr, LOW);
			} else {
				Merge(containNum, partnerNum, data, tempArr, HIGH);
			}
			// printf("Rank: %d finish sorting\n", rank);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		// printf("Rank: %d out barrier\n", rank);

		int isSorted = 1;
		int sumUp = 0;
		MPI_Allreduce(&shouldContinue, &sumUp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (sumUp > 0 || isFirstTime) isSorted = 0;
		isFirstTime = 0;

		// printf("Rank: %d isSorted: %d\n", rank, isSorted);
		// if (isSorted) break; // TODO: Early stop
	}

	// for (int i=0;i<containNum;i++) {
	// 	printf("rank: %d element[%d]: %f\n", rank, i, data[i]);
	// }

	MPI_File wf;
	int wc = MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &wf);
	MPI_File_write_at(wf, sizeof(float) * offset, &data, containNum, MPI_FLOAT, MPI_STATUS_IGNORE);

	MPI_Finalize();
}
