#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 64

using namespace std;

const int INF = ((1 << 30) - 1);
// const int V = 50010;
void input(char* infile);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
__global__ void Phase1(int *dist, int Round, int n);
__global__ void Phase2(int *dist, int Round, int n);
__global__ void Phase3(int *dist, int Round, int n);

int original_n, n, m;
int* Dist = NULL;

int main(int argc, char* argv[]) {
	input(argv[1]);
	block_FW(BLOCK_SIZE);
	output(argv[2]);
    cudaFreeHost(Dist);
	return 0;
}

void input(char* infile) {
    cout << "input" << endl;
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

void block_FW(int B) {
    int* dst = NULL;

    const int matrixSize = n * n * sizeof(int);

    cudaHostRegister(Dist, matrixSize, cudaHostRegisterDefault);
    cudaMalloc(&dst, matrixSize);
	cudaMemcpy(dst, Dist, matrixSize, cudaMemcpyHostToDevice);

    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_dim(BLOCK_SIZE, 1, 1);
    dim3 grid_dim(blocks, blocks, 1);

    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        Phase1<<<1, block_dim>>>(dst, r, n);

        /* Phase 2*/
        Phase2<<<blocks, block_dim>>>(dst, r, n);

        /* Phase 3*/
        Phase3<<<grid_dim, block_dim>>>(dst, r, n);
    }

    cudaMemcpy(Dist, dst, matrixSize, cudaMemcpyDeviceToHost);
	cudaFree(dst);
}


inline __device__ void BlockCalc(int* C, int* A, int* B, int innerJ) {
    for (int k = 0; k < BLOCK_SIZE; k++) {
        for (int innerI=0; innerI < BLOCK_SIZE; innerI++) {
            int sum = A[innerI*BLOCK_SIZE + k] + B[k*BLOCK_SIZE + innerJ];
            if (C[innerI*BLOCK_SIZE + innerJ] > sum) {
                C[innerI*BLOCK_SIZE + innerJ] = sum;
            }
        }
        __syncthreads();
    }
    //   printf("New Added Element[%d][%d]: %d   Element[%d][%d]: %d  Combine Value: %d | Original Value: %d\n", bi, k, A[bi*BLOCK_SIZE + k], k, bj, B[k*BLOCK_SIZE + bj], sum, C[bi*BLOCK_SIZE + bj]);
  }

__global__ void Phase1(int *dist, int Round, int n) {
    // const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int offset = BLOCK_SIZE * Round;

    __shared__ int C[BLOCK_SIZE * BLOCK_SIZE];

    // Every thread read its own value
    // how index: blockIndex (to next diagonal block) + innerBlockIndex (every thread has its own index)
    for (int innerI=0; innerI<BLOCK_SIZE; innerI++) C[innerI * BLOCK_SIZE + innerJ] = dist[offset*(n+1) + innerI*n + innerJ];
    __syncthreads();
    BlockCalc(C, C, C, innerJ);
    __syncthreads();
    for (int innerI=0; innerI<BLOCK_SIZE; innerI++) dist[offset*(n+1) + innerI*n + innerJ] = C[innerI * BLOCK_SIZE + innerJ];
}

__global__ void Phase2(int *dist, int Round, int n) {
    const int i = blockIdx.x; // "i" in n block in one row
    if (i == Round) return;

    // const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int diagonalOffset = BLOCK_SIZE * Round;

    __shared__ int Diagonal[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int B[BLOCK_SIZE * BLOCK_SIZE];
  
    for (int innerI=0; innerI < BLOCK_SIZE; innerI++) {
        A[innerI*BLOCK_SIZE + innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ];
        B[innerI*BLOCK_SIZE + innerJ] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + innerI*n + innerJ];
        Diagonal[innerI*BLOCK_SIZE + innerJ] = dist[diagonalOffset*(n+1) + innerI*n + innerJ]; // diagonalValue
    }
  
    __syncthreads();
  
    BlockCalc(A, A, Diagonal, innerJ);
    BlockCalc(B, Diagonal, B, innerJ);

    __syncthreads();
  
    for (int innerI=0; innerI < BLOCK_SIZE; innerI++) {
        dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ] = A[innerI*BLOCK_SIZE + innerJ];
        dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + innerI*n + innerJ] = B[innerI*BLOCK_SIZE + innerJ];
    }
}

__global__ void Phase3(int *dist, int Round, int n) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    if (i == Round && j == Round) return;

    // const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;

    __shared__ int A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int B[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int C[BLOCK_SIZE * BLOCK_SIZE];
  
    for (int innerI=0; innerI < BLOCK_SIZE; innerI++) {
        C[innerI*BLOCK_SIZE + innerJ] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ];
        A[innerI*BLOCK_SIZE + innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + innerI*n + innerJ];
        B[innerI*BLOCK_SIZE + innerJ] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ];
    }
  
    __syncthreads();
  
    BlockCalc(C, A, B, innerJ);
  
    __syncthreads();
  
    for (int innerI=0; innerI < BLOCK_SIZE; innerI++) dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + innerI*n + innerJ] = C[innerI*BLOCK_SIZE + innerJ];
}