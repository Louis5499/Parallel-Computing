#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#define BLOCK_SIZE 64
#define HALF_BLOCK_SIZE 32

const int INF = ((1 << 30) - 1);

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

int original_n, n, m;
int* Dist = NULL;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&original_n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // make n % BLOCK_SIZE == 0
    n = original_n + (BLOCK_SIZE - (original_n%BLOCK_SIZE));

    Dist = (int*) malloc(sizeof(int)*n*n);

    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
            if (i == j) {
                Dist[i*n+j] = 0;
            } else {
                Dist[i*n+j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++ i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);

}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < original_n; ++i) {
		for (int j = 0; j < original_n; ++j) {
            if (Dist[i*n+j] >= INF) Dist[i*n+j] = INF;
        }
		fwrite(&Dist[i*n], sizeof(int), original_n, outfile);
	}
    fclose(outfile);
}

// int main(int argc, char* argv[]) {
// 	input(argv[1]);
// 	block_FW(BLOCK_SIZE);
// 	output(argv[2]);
//     cudaFreeHost(Dist);
// 	return 0;
// }

int ceil(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char *argv[]){
    input(argv[1]);

    size_t matrixSize = n * n * sizeof(int);

    cudaHostRegister(Dist, matrixSize, cudaHostRegisterDefault);

    int *adj_mat_d[2];
    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
    // 2D block
    dim3 block_dim(32, 32);

    int round = ceil(n, BLOCK_SIZE);
	#pragma omp parallel num_threads(2)
	{
		int thread_id = omp_get_thread_num();
		cudaSetDevice(thread_id);

        // Malloc memory
        cudaMalloc(&adj_mat_d[thread_id], matrixSize);

        // divide data
		int round_per_thd = round / 2;
		int y_offset = round_per_thd * thread_id;
        if(thread_id == 1) round_per_thd += round % 2;

		dim3 grid_dim(round, round_per_thd);
		
        size_t cp_amount = n * BLOCK_SIZE * round_per_thd * sizeof(int);
        cudaMemcpy(adj_mat_d[thread_id] + y_offset *BLOCK_SIZE * n, Dist + y_offset * BLOCK_SIZE * n, cp_amount, cudaMemcpyHostToDevice);

        size_t block_row_sz = BLOCK_SIZE * n * sizeof(int);
		for(int r = 0; r < round; r++) {

            // Every thread has its own y_offset
            if (r >= y_offset && r < (y_offset + round_per_thd)) {
                cudaMemcpy(Dist + r * BLOCK_SIZE * n, adj_mat_d[thread_id] + r * BLOCK_SIZE * n, block_row_sz, cudaMemcpyDeviceToHost);
            }

            #pragma omp barrier
            cudaMemcpy(adj_mat_d[thread_id] + r * BLOCK_SIZE * n, Dist + r * BLOCK_SIZE * n, block_row_sz, cudaMemcpyHostToDevice);

            /* Phase 1*/
            Phase_1 <<<1, block_dim>>>(adj_mat_d[thread_id], r, n);

            /* Phase 2*/
            Phase_2 <<<blocks, block_dim>>>(adj_mat_d[thread_id], r, n);

            /* Phase 3*/
            Phase_3 <<<grid_dim, block_dim>>>(adj_mat_d[thread_id], r, n, y_offset);
        }

		cudaMemcpy(Dist + y_offset *BLOCK_SIZE * n, adj_mat_d[thread_id] + y_offset *BLOCK_SIZE * n, block_row_sz * round_per_thd, cudaMemcpyDeviceToHost);
		#pragma omp barrier
	}
	
	output(argv[2]);

	//free memory
	cudaFree(adj_mat_d[0]);
    cudaFree(adj_mat_d[1]);
    cudaFreeHost(Dist);
	return 0;
}