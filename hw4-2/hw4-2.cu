#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#define BLOCK_SIZE 64
#define HALF_BLOCK_SIZE 32
#define inf 1e9

__global__ void Phase_1(int *dist, int Round, int n) {
    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int offset = BLOCK_SIZE * Round;

    __shared__ int C[BLOCK_SIZE][BLOCK_SIZE]; // 2d

    // Every thread read its own value
    // how index: blockIndex (to next diagonal block) + innerBlockIndex (every thread has its own index)
    C[innerI][innerJ] = dist[offset*(n+1) + innerI*n + innerJ];
    C[innerI+HALF_BLOCK_SIZE][innerJ] = dist[offset*(n+1) + (innerI+HALF_BLOCK_SIZE)*n + innerJ];
    C[innerI][innerJ+HALF_BLOCK_SIZE] = dist[offset*(n+1) + innerI*n + innerJ + HALF_BLOCK_SIZE];
    C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = dist[offset*(n+1) + (innerI+HALF_BLOCK_SIZE)*n + innerJ + HALF_BLOCK_SIZE];
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        C[innerI][innerJ] = (C[innerI][k] + C[k][innerJ]) < C[innerI][innerJ] ? (C[innerI][k] + C[k][innerJ]) : C[innerI][innerJ];

        C[innerI+HALF_BLOCK_SIZE][innerJ] = (C[innerI+HALF_BLOCK_SIZE][k] + C[k][innerJ]) < C[innerI+HALF_BLOCK_SIZE][innerJ] ? (C[innerI+HALF_BLOCK_SIZE][k] + C[k][innerJ]) : C[innerI+HALF_BLOCK_SIZE][innerJ];

        C[innerI][innerJ+HALF_BLOCK_SIZE] = (C[innerI][k] + C[k][innerJ+HALF_BLOCK_SIZE]) < C[innerI][innerJ+HALF_BLOCK_SIZE] ? (C[innerI][k] + C[k][innerJ+HALF_BLOCK_SIZE]) : C[innerI][innerJ+HALF_BLOCK_SIZE];

        C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = (C[innerI+HALF_BLOCK_SIZE][k] + C[k][innerJ+HALF_BLOCK_SIZE]) < C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] ? (C[innerI+HALF_BLOCK_SIZE][k] + C[k][innerJ+HALF_BLOCK_SIZE]) : C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
        __syncthreads(); // TODO: only phase one
    }

    dist[offset*(n+1) + innerI*n + innerJ] = C[innerI][innerJ];
    dist[offset*(n+1) + (innerI+HALF_BLOCK_SIZE)*n + innerJ] = C[innerI+HALF_BLOCK_SIZE][innerJ];
    dist[offset*(n+1) + innerI*n + innerJ + HALF_BLOCK_SIZE] = C[innerI][innerJ+HALF_BLOCK_SIZE];
    dist[offset*(n+1) + (innerI+HALF_BLOCK_SIZE)*n + innerJ + HALF_BLOCK_SIZE] = C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
}


__global__ void Phase_2(int *dist, int Round, int n) {
    const int i = blockIdx.x; // "i" in n block in one row
    if (i == Round) return;

    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int diagonalOffset = BLOCK_SIZE * Round;

    __shared__ int Diagonal[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int B[BLOCK_SIZE][BLOCK_SIZE];
  
    A[innerI][innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ];
    A[innerI+HALF_BLOCK_SIZE][innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ];
    A[innerI][innerJ+HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ + HALF_BLOCK_SIZE];
    A[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ+HALF_BLOCK_SIZE];

    B[innerI][innerJ] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + innerI*n + innerJ];
    B[innerI+HALF_BLOCK_SIZE][innerJ] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ];
    B[innerI][innerJ+HALF_BLOCK_SIZE] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + innerI*n + innerJ+HALF_BLOCK_SIZE];
    B[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ+HALF_BLOCK_SIZE];

    Diagonal[innerI][innerJ] = dist[diagonalOffset*(n+1) + innerI*n + innerJ]; // diagonalValue
    Diagonal[innerI+HALF_BLOCK_SIZE][innerJ] = dist[diagonalOffset*(n+1) + (innerI+HALF_BLOCK_SIZE)*n + innerJ]; // diagonalValue
    Diagonal[innerI][innerJ+HALF_BLOCK_SIZE] = dist[diagonalOffset*(n+1) + innerI*n + innerJ+HALF_BLOCK_SIZE]; // diagonalValue
    Diagonal[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = dist[diagonalOffset*(n+1) + (innerI+HALF_BLOCK_SIZE)*n + innerJ+HALF_BLOCK_SIZE]; // diagonalValue
  
    __syncthreads();

    #pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {

        A[innerI][innerJ] = (A[innerI][k] + Diagonal[k][innerJ]) < A[innerI][innerJ] ? (A[innerI][k] + Diagonal[k][innerJ]) : A[innerI][innerJ];

        A[innerI+HALF_BLOCK_SIZE][innerJ] = (A[innerI+HALF_BLOCK_SIZE][k] + Diagonal[k][innerJ]) < A[innerI+HALF_BLOCK_SIZE][innerJ] ? (A[innerI+HALF_BLOCK_SIZE][k] + Diagonal[k][innerJ]) : A[innerI+HALF_BLOCK_SIZE][innerJ];

        A[innerI][innerJ+HALF_BLOCK_SIZE] = (A[innerI][k] + Diagonal[k][innerJ+HALF_BLOCK_SIZE]) < A[innerI][innerJ+HALF_BLOCK_SIZE] ? (A[innerI][k] + Diagonal[k][innerJ+HALF_BLOCK_SIZE]) : A[innerI][innerJ+HALF_BLOCK_SIZE];
        
        A[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = (A[innerI+HALF_BLOCK_SIZE][k] + Diagonal[k][innerJ+HALF_BLOCK_SIZE]) < A[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] ? (A[innerI+HALF_BLOCK_SIZE][k] + Diagonal[k][innerJ+HALF_BLOCK_SIZE]) : A[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];

        B[innerI][innerJ] = (Diagonal[innerI][k] + B[k][innerJ]) < B[innerI][innerJ] ? (Diagonal[innerI][k] + B[k][innerJ]) : B[innerI][innerJ];

        B[innerI+HALF_BLOCK_SIZE][innerJ] = (Diagonal[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ]) < B[innerI+HALF_BLOCK_SIZE][innerJ] ? (Diagonal[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ]) : B[innerI+HALF_BLOCK_SIZE][innerJ];

        B[innerI][innerJ+HALF_BLOCK_SIZE] = (Diagonal[innerI][k] + B[k][innerJ+HALF_BLOCK_SIZE]) < B[innerI][innerJ+HALF_BLOCK_SIZE] ? (Diagonal[innerI][k] + B[k][innerJ+HALF_BLOCK_SIZE]) : B[innerI][innerJ+HALF_BLOCK_SIZE];
        
        B[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = (Diagonal[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ+HALF_BLOCK_SIZE]) < B[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] ? (Diagonal[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ+HALF_BLOCK_SIZE]) : B[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
    }

    dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ] = A[innerI][innerJ];
    dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ] = A[innerI+HALF_BLOCK_SIZE][innerJ];
    dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ + HALF_BLOCK_SIZE] = A[innerI][innerJ+HALF_BLOCK_SIZE];
    dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ+HALF_BLOCK_SIZE] = A[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE];

    dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + innerI*n + innerJ] = B[innerI][innerJ];
    dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ] = B[innerI+HALF_BLOCK_SIZE][innerJ];
    dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + innerI*n + innerJ+HALF_BLOCK_SIZE] = B[innerI][innerJ+HALF_BLOCK_SIZE];
    dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ+HALF_BLOCK_SIZE] = B[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE];
}

__global__ void Phase_3(int *dist, int Round, int n, int yOffset) {
    const int j = blockIdx.x;
    const int i = blockIdx.y + yOffset;
    if (i == Round && j == Round) return;

    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;

    __shared__ int A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int B[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int C[BLOCK_SIZE][BLOCK_SIZE];
  
    C[innerI][innerJ] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ];
    C[innerI+HALF_BLOCK_SIZE][innerJ] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ];
    C[innerI][innerJ+HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ+HALF_BLOCK_SIZE];
    C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ+HALF_BLOCK_SIZE];


    A[innerI][innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ];
    A[innerI+HALF_BLOCK_SIZE][innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ];
    A[innerI][innerJ+HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ + HALF_BLOCK_SIZE];
    A[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ + HALF_BLOCK_SIZE];

    B[innerI][innerJ] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ];
    B[innerI+HALF_BLOCK_SIZE][innerJ] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ];
    B[innerI][innerJ+HALF_BLOCK_SIZE] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ+HALF_BLOCK_SIZE];
    B[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ+HALF_BLOCK_SIZE];
  
    __syncthreads();

    #pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        C[innerI][innerJ] = (A[innerI][k] + B[k][innerJ]) < C[innerI][innerJ] ? (A[innerI][k] + B[k][innerJ]) : C[innerI][innerJ];

        C[innerI+HALF_BLOCK_SIZE][innerJ] = (A[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ]) < C[innerI+HALF_BLOCK_SIZE][innerJ] ? (A[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ]) : C[innerI+HALF_BLOCK_SIZE][innerJ];

        C[innerI][innerJ+HALF_BLOCK_SIZE] = (A[innerI][k] + B[k][innerJ+HALF_BLOCK_SIZE]) < C[innerI][innerJ+HALF_BLOCK_SIZE] ? (A[innerI][k] + B[k][innerJ+HALF_BLOCK_SIZE]) : C[innerI][innerJ+HALF_BLOCK_SIZE];
        
        C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = (A[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ+HALF_BLOCK_SIZE]) < C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] ? (A[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ+HALF_BLOCK_SIZE]) : C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
    }

    dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ] = C[innerI][innerJ];
    dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ] = C[innerI+HALF_BLOCK_SIZE][innerJ];
    dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ+HALF_BLOCK_SIZE] = C[innerI][innerJ+HALF_BLOCK_SIZE];
    dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+HALF_BLOCK_SIZE)*n + innerJ+HALF_BLOCK_SIZE] = C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
}

int main(int argc, char *argv[]){
	/******************************* load data *********************************/
    // only two arguments are allowed
    assert(argc == 3);

    int E, V;
    FILE *in_fp;
    in_fp = fopen(argv[1], "r");
    if(in_fp == NULL) printf("Failed on opening file\n");
    // read in data
    fread(&V, sizeof(int), 1, in_fp);
    fread(&E, sizeof(int), 1, in_fp);

    // compensate V to make V % BLOCK_SIZE == 0
	int comp_V = V + (BLOCK_SIZE - ((V-1) % BLOCK_SIZE + 1));

	//allocate memory
    int *adj_mat; 
    size_t sz = comp_V * comp_V * sizeof(int);
	cudaMallocHost((void**) &adj_mat, sz);
	for(int i = 0; i < comp_V; i++){
        for(int j = 0; j < comp_V; j++){
            if(i == j) adj_mat[i*comp_V+j] = 0;
            else adj_mat[i*comp_V+j] = inf;
        }
    }
    // load data to graph
    int src, dst, w;
    while(E--){
        fread(&src, sizeof(int), 1, in_fp);
        fread(&dst, sizeof(int), 1, in_fp);
        fread(&w, sizeof(int), 1, in_fp);
        adj_mat[src*comp_V+dst] = w;
    }
    fclose(in_fp);
    /****************************************************************************/

    int *adj_mat_d[2];
    int round =  ceil((float) comp_V/BLOCK_SIZE);
    const int blocks = (comp_V + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
    // 2D block
    dim3 threads(32, 32);
	dim3 p2(2, round-1);

	#pragma omp parallel num_threads(2)
	{
		int thread_id = omp_get_thread_num();
		cudaSetDevice(thread_id);

        // Malloc memory
        cudaMalloc(&adj_mat_d[thread_id], sz);

        // divide data
		int round_per_thd = round / 2;
		int y_offset = round_per_thd * thread_id;
        if(thread_id == 1) round_per_thd += round % 2;

		dim3 p3(round, round_per_thd);
		
        size_t cp_amount = comp_V * BLOCK_SIZE * round_per_thd * sizeof(int);
        cudaMemcpy(adj_mat_d[thread_id] + y_offset *BLOCK_SIZE * comp_V, adj_mat + y_offset * BLOCK_SIZE * comp_V, cp_amount, cudaMemcpyHostToDevice);

        size_t block_row_sz = BLOCK_SIZE * comp_V * sizeof(int);
		for(int r = 0; r < round; r++) {

            // Every thread has its own y_offset
            if (r >= y_offset && r < (y_offset + round_per_thd)) {
                cudaMemcpy(adj_mat + r * BLOCK_SIZE * comp_V, adj_mat_d[thread_id] + r * BLOCK_SIZE * comp_V, block_row_sz, cudaMemcpyDeviceToHost);
            }

            #pragma omp barrier
            cudaMemcpy(adj_mat_d[thread_id] + r * BLOCK_SIZE * comp_V, adj_mat + r * BLOCK_SIZE * comp_V, block_row_sz, cudaMemcpyHostToDevice);

            Phase_1 <<<1, threads>>>(adj_mat_d[thread_id], r, comp_V);
                
            //cudaDeviceSynchronize();

            Phase_2 <<<blocks, threads>>>(adj_mat_d[thread_id], r, comp_V);

            //cudaDeviceSynchronize();

            Phase_3 <<<p3, threads>>>(adj_mat_d[thread_id], r, comp_V, y_offset);
        }

		cudaMemcpy(adj_mat + y_offset *BLOCK_SIZE * comp_V, adj_mat_d[thread_id] + y_offset *BLOCK_SIZE * comp_V, block_row_sz * round_per_thd, cudaMemcpyDeviceToHost);
		#pragma omp barrier
	}
	
	// output
    FILE *out_fp;
    out_fp = fopen(argv[2], "wb");
    for(int i = 0; i < V; i++){
        for(int j = 0; j < V; j++){
            fwrite(adj_mat+i*comp_V+j, sizeof(int), 1, out_fp);
        }   
    }   
    fclose(out_fp);

	//free memory
	cudaFree(adj_mat_d[0]);
    cudaFree(adj_mat_d[1]);
    cudaFreeHost(adj_mat);
	return 0;
}