// When array is very large, the only way to create array is malloc!
// comparator can't use default method
// lesser MPI is perfect
// Problem with Non-blocking
// Early start can't speedup
// Early declare function
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>

#define HIGH 1
#define LOW 0  
#define MERGE_TMP_ARRAY_SIZE 536869890

// function initialization
int comparator(const void *a, const void *b);
void merge(int size_a, int size_b, float *data_a, float *data_b, int high_low);

float *L = (float *)malloc(MERGE_TMP_ARRAY_SIZE * sizeof(float));
float *R = (float *)malloc(MERGE_TMP_ARRAY_SIZE * sizeof(float));
void mergeAndCombine(float arr[], int l, int m, int r)  
{  
    int i, j, k;  
    int n1 = m - l + 1;  
    int n2 = r - m;
  
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)  
        L[i] = arr[l + i];  
    for (j = 0; j < n2; j++)  
        R[j] = arr[m + 1 + j];  
  
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray  
    j = 0; // Initial index of second subarray  
    k = l; // Initial index of merged subarray  
    while (i < n1 && j < n2) {  
        if (L[i] <= R[j]) {  
            arr[k] = L[i];  
            i++;  
        }  
        else {  
            arr[k] = R[j];  
            j++;  
        }  
        k++;  
    }  
  
    /* Copy the remaining elements of L[], if there  
    are any */
    while (i < n1) {  
        arr[k] = L[i];  
        i++;  
        k++;  
    }  
  
    /* Copy the remaining elements of R[], if there  
    are any */
    while (j < n2) {  
        arr[k] = R[j];  
        j++;  
        k++;  
    }
}

/* l is for left index and r is right index of the  
sub-array of arr to be sorted */
void mergeSort(float arr[], int l, int r)  
{  
    if (l < r) {  
        // Same as (l+r)/2, but avoids overflow for  
        // large l and h  
        int m = l + (r - l) / 2;  
  
        // Sort first and second halves  
        mergeSort(arr, l, m);  
        mergeSort(arr, m + 1, r);  
  
        mergeAndCombine(arr, l, m, r);  
    }  
}

double timeDiff(struct timespec start, struct timespec end){
    // function used to measure time in nano resolution
    float output;
    float nano = 1000000000.0;
    if(end.tv_nsec < start.tv_nsec) output = ((end.tv_sec - start.tv_sec -1)+(nano+end.tv_nsec-start.tv_nsec)/nano);
    else output = ((end.tv_sec - start.tv_sec)+(end.tv_nsec-start.tv_nsec)/nano);
    return output;
}


int main(int argc, char *argv[]){
    // initialization for parameters
    const int N = atoi(argv[1]);
    
    // initialize for MPI parameters
    int rank, size, rc;
    MPI_File file_in, file_out;
    MPI_Status status;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Group old_group;

    // initialize time measurement
    struct timespec start, end;
    double io_time, compute_time, comm_time;

    // Initial MPI environment
    rc = MPI_Init(&argc, &argv);
    if(rc != MPI_SUCCESS){
        printf("Error starting MPI program. Terminating\n");
        MPI_Abort(mpi_comm, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // if the number of process is larger than N...
    if(N < size){
    	// obtain the group of processes in the world comm.
        MPI_Comm_group(mpi_comm, &old_group);

        // Remove unnecessary processes
        MPI_Group new_group;
        int ranges[][3] = {{N, size-1, 1}};
        MPI_Group_range_excl(old_group, 1, ranges, &new_group);

        // Create new comm.
        MPI_Comm_create(mpi_comm, new_group, &mpi_comm);

        if(mpi_comm == MPI_COMM_NULL){
            MPI_Finalize();
            exit(0);
        }
        size = N;
    }

    clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
    int avgNumPerNode = ((int)(N/size));
	int modNumber = N%size;

	int local_size = (rank < modNumber) ? avgNumPerNode + 1 : avgNumPerNode;
	int offset = 0;
	for (int i=0;i<rank;i++) { // TODO: Prevent for-loop
		offset += (i < modNumber) ? (avgNumPerNode + 1) : avgNumPerNode;
	}
    offset = offset*sizeof(float); // TODO: One line

    float *local_data = (float *)malloc(local_size * sizeof(float));

    clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
    // compute_time += timeDiff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------

    int irc = MPI_File_open(mpi_comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &file_in);
    if(irc != MPI_SUCCESS){
        printf("[Error] input read in FAILED");
        MPI_Abort(mpi_comm, irc);
    }
    // Read at certain offset
    MPI_File_read_at(file_in, offset, local_data, local_size, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&file_in);

    clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
    io_time += timeDiff(start, end);

    // Start processor level odd-even sort
    // perform local sort first
    clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
    mergeSort(local_data, 0, local_size - 1);

    int right_size = ((rank+1) < modNumber) ? avgNumPerNode + 1 : avgNumPerNode;
    int left_size = ((rank-1) < modNumber) ? avgNumPerNode + 1 : avgNumPerNode;

    clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
    compute_time += timeDiff(start, end);

    // start merge sort
    MPI_Request req1;
    int is_first_time = 1;
    for(int phase=0; phase<=size; phase++){
        int is_swap = 0;
        int combine = (rank+phase);
        if(combine%2 == 0 && rank != size-1) {
            float *my_temp = (float *)malloc(right_size * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
            MPI_Sendrecv(local_data, local_size, MPI_FLOAT, rank+1, 1, my_temp, right_size, MPI_FLOAT, rank+1, 1, mpi_comm, MPI_STATUS_IGNORE);
            clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
            comm_time += timeDiff(start, end);
            clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
            if (local_data[local_size - 1] > my_temp[0]) {
                merge(local_size, right_size, local_data, my_temp, LOW);
                is_swap = 1;
            }
            free(my_temp);
            clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
            compute_time += timeDiff(start, end);
        } else if(combine%2 == 1 && rank != 0) {
            float *my_temp = (float *)malloc(left_size * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
            MPI_Sendrecv(local_data, local_size, MPI_FLOAT, rank-1, 1, my_temp, left_size, MPI_FLOAT, rank-1, 1, mpi_comm, MPI_STATUS_IGNORE);
            clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
            comm_time += timeDiff(start, end);
            clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
            if (my_temp[left_size - 1] > local_data[0]) {
                merge(local_size, left_size, local_data, my_temp, HIGH);
                is_swap = 1;
            }
            free(my_temp);
            clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
            compute_time += timeDiff(start, end);
        }

        int sum_up = 0;
        clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
        MPI_Allreduce(&is_swap, &sum_up, 1, MPI_INT, MPI_SUM, mpi_comm);
        clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
        comm_time += timeDiff(start, end);
        if (sum_up == 0 && !is_first_time) break;
        is_first_time = 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
	int wc = MPI_File_open(mpi_comm, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file_out);
	MPI_File_write_at(file_out, offset, local_data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    clock_gettime(CLOCK_MONOTONIC, &end); // E---------------------------------------------------------------------------------
    io_time += timeDiff(start, end);

    // double total_comm_time = 0.0, total_io_time = 0.0, total_compute_time = 0.0;
    // MPI_Reduce(&comm_time, &total_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
    // MPI_Reduce(&io_time, &total_io_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
    // MPI_Reduce(&compute_time, &total_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
    // if (rank == 0) printf("comm_time: %f\nio_time: %f\ncompute_time: %f\n", total_comm_time, total_io_time, total_compute_time);
    if (rank == 0) printf("comm_time: %f\nio_time: %f\ncompute_time: %f\n", comm_time, io_time, compute_time);
    MPI_Finalize();

    return 0;
}

// Note that if we simply return the substraction of these two value, the return value may overflow
// So the better way is to compare it first and return the value
// If *(float *)a > *(float *)b, then it returns 1, else return -1.
int comparator(const void *a, const void *b){
    return (*(float *)a > *(float *)b) - (*(float *)a < *(float *)b);
}

float *tmp = (float *)malloc(sizeof(float)*MERGE_TMP_ARRAY_SIZE);
void merge(int my_size, int other_size, float *my_data, float *other_data, int high_low){
    // merge the list 
	if(high_low == LOW){
		// keep low
    	int my_idx = 0, other_idx = 0, out_idx = 0;
        while(out_idx < my_size) {
            if (my_idx < my_size && other_idx < other_size) {
                if(my_data[my_idx] < other_data[other_idx]) tmp[out_idx++] = my_data[my_idx++];
                else tmp[out_idx++] = other_data[other_idx++];
            }
            else if (my_idx < my_size) tmp[out_idx++] = my_data[my_idx++];
            else tmp[out_idx++] = other_data[other_idx++];
        }
	} else{
		int my_idx = my_size-1, other_idx = other_size-1, out_idx = my_size-1;
        while (out_idx >= 0) {
            if (my_idx >= 0 && other_idx >= 0) {
                if(my_data[my_idx] < other_data[other_idx]) tmp[out_idx--] = other_data[other_idx--];
        	    else tmp[out_idx--] = my_data[my_idx--];
            }
            else if (my_idx >= 0) tmp[out_idx--] = my_data[my_idx--];
            else tmp[out_idx--] = other_data[other_idx--];
        }
	}
    for(int i = 0; i < my_size; i++){
        my_data[i] = tmp[i];
    }
}