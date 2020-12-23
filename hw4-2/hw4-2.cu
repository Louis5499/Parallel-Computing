#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#define BLOCK_SIZE 64
#define HALF_BLOCK_SIZE 32

using namespace std;

const int INF = ((1 << 30) - 1);
void input(char* infile);
void output(char *outFileName);

void block_FW();
int ceil(int a, int b);
__global__ void Phase1(int *dist, int Round, int n);
__global__ void Phase2(int *dist, int Round, int n);
__global__ void Phase3(int *dist, int Round, int n, int yOffset);

int original_n, n, m;
int* Dist = NULL;

int main(int argc, char *argv[]){
    input(argv[1]);
	block_FW();
	output(argv[2]);
    cudaFreeHost(Dist);
	return 0;
}

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

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW() {
    int *dst[2];

    size_t matrixSize = n * n * sizeof(int);
    cudaHostRegister(Dist, matrixSize, cudaHostRegisterDefault);

    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
    // 2D block
    dim3 block_dim(32, 32);

    int round = ceil(n, BLOCK_SIZE);
	#pragma omp parallel num_threads(2)
	{
		const int cpuThreadId = omp_get_thread_num();
		cudaSetDevice(cpuThreadId);

        cudaMalloc(&dst[cpuThreadId], matrixSize);

		int roundPerThread = round / 2;
        const int yOffset = roundPerThread * cpuThreadId;
        const size_t yOffsetSize = yOffset*BLOCK_SIZE*n;
        if(cpuThreadId == 1) roundPerThread += round % 2;

		dim3 grid_dim(round, roundPerThread);
        
        const size_t rowBlockSize = BLOCK_SIZE * n * sizeof(int);
        const size_t halfBlockSize = rowBlockSize * roundPerThread;
        cudaMemcpy(dst[cpuThreadId] + yOffsetSize, Dist + yOffsetSize, halfBlockSize, cudaMemcpyHostToDevice);

		for(int r = 0; r < round; r++) {
            // Every thread has its own yOffset
            if (r >= yOffset && r < (yOffset + roundPerThread)) {
                cudaMemcpy(Dist + r * BLOCK_SIZE * n, dst[cpuThreadId] + r * BLOCK_SIZE * n, rowBlockSize, cudaMemcpyDeviceToHost);
            }

            #pragma omp barrier
            cudaMemcpy(dst[cpuThreadId] + r * BLOCK_SIZE * n, Dist + r * BLOCK_SIZE * n, rowBlockSize, cudaMemcpyHostToDevice);

            /* Phase 1*/
            Phase1 <<<1, block_dim>>>(dst[cpuThreadId], r, n);

            /* Phase 2*/
            Phase2 <<<blocks, block_dim>>>(dst[cpuThreadId], r, n);

            /* Phase 3*/
            Phase3 <<<grid_dim, block_dim>>>(dst[cpuThreadId], r, n, yOffset);
        }

		cudaMemcpy(Dist + yOffsetSize, dst[cpuThreadId] + yOffsetSize, halfBlockSize, cudaMemcpyDeviceToHost);
		#pragma omp barrier
    }
    cudaFree(dst[0]);
    cudaFree(dst[1]);
}

__global__ void Phase1(int *dist, int Round, int n) {
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


__global__ void Phase2(int *dist, int Round, int n) {
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

__global__ void Phase3(int *dist, int Round, int n, int yOffset) {
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
