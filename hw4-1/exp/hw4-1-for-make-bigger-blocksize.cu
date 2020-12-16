#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 64
#define HALF_BLOCK_SIZE 32

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

struct timespec start, timeEnd;
double io_time=0.0;
double timeDiff(struct timespec start, struct timespec timeEnd){
    // function used to measure time in nano resolution
    float output;
    float nano = 1000000000.0;
    if(timeEnd.tv_nsec < start.tv_nsec) output = ((timeEnd.tv_sec - start.tv_sec -1)+(nano+timeEnd.tv_nsec-start.tv_nsec)/nano);
    else output = ((timeEnd.tv_sec - start.tv_sec)+(timeEnd.tv_nsec-start.tv_nsec)/nano);
    return output;
}

int main(int argc, char* argv[]) {
    clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
    input(argv[1]);
    clock_gettime(CLOCK_MONOTONIC, &timeEnd); // E---------------------------------------------------------------------------------
    io_time += timeDiff(start, timeEnd);
    block_FW(BLOCK_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
    output(argv[2]);
    clock_gettime(CLOCK_MONOTONIC, &timeEnd); // E---------------------------------------------------------------------------------
    io_time += timeDiff(start, timeEnd);
    printf("io time: %f\n", io_time);
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
    cout << "original n: " << original_n << "  n: " << n << endl;

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

    const unsigned long matrixSize = n * n * sizeof(int);

    cudaHostRegister(Dist, matrixSize, cudaHostRegisterDefault);
    cudaMalloc(&dst, matrixSize);
	cudaMemcpy(dst, Dist, matrixSize, cudaMemcpyHostToDevice);

    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_dim(32, 32, 1); // padding (一定要開到 32x)
    dim3 grid_dim(blocks, blocks, 1);

    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);
        // fflush(stdout);
        /* Phase 1*/
        Phase1<<<1, block_dim>>>(dst, r, n);

        /* Phase 2*/
        Phase2<<<blocks, block_dim>>>(dst, r, n);

        /* Phase 3*/
        Phase3<<<grid_dim, block_dim>>>(dst, r, n);
    }

    cudaMemcpy(Dist, dst, matrixSize, cudaMemcpyDeviceToHost);
    cudaFree(dst);
    cout << "cudaGetLastError: " << cudaGetErrorString(cudaGetLastError()) << endl;
}

__global__ void Phase1(int *dist, int Round, int n) {
    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int offset = BLOCK_SIZE * Round;

    __shared__ int C[64][64]; // 2d
    for (int totalOffsetI=0; totalOffsetI<BLOCK_SIZE; totalOffsetI+=64) {
        int matrixInnerI = innerI + totalOffsetI;
        for (int totalOffsetJ=0; totalOffsetJ<BLOCK_SIZE; totalOffsetJ+=64) {
            int matrixInnerJ = innerJ + totalOffsetJ;

            // Every thread read its own value
            // how index: blockIndex (to next diagonal block) + innerBlockIndex (every thread has its own index)
            C[innerI][innerJ] = dist[offset*(n+1) + matrixInnerI*n + matrixInnerJ];
            C[innerI+HALF_BLOCK_SIZE][innerJ] = dist[offset*(n+1) + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ];
            C[innerI][innerJ+HALF_BLOCK_SIZE] = dist[offset*(n+1) + matrixInnerI*n + matrixInnerJ + HALF_BLOCK_SIZE];
            C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = dist[offset*(n+1) + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ + HALF_BLOCK_SIZE];
            __syncthreads();

            for (int k = 0; k < 64; k++) {
                C[innerI][innerJ] = (C[innerI][k] + C[k][innerJ]) < C[innerI][innerJ] ? (C[innerI][k] + C[k][innerJ]) : C[innerI][innerJ];

                C[innerI+HALF_BLOCK_SIZE][innerJ] = (C[innerI+HALF_BLOCK_SIZE][k] + C[k][innerJ]) < C[innerI+HALF_BLOCK_SIZE][innerJ] ? (C[innerI+HALF_BLOCK_SIZE][k] + C[k][innerJ]) : C[innerI+HALF_BLOCK_SIZE][innerJ];

                C[innerI][innerJ+HALF_BLOCK_SIZE] = (C[innerI][k] + C[k][innerJ+HALF_BLOCK_SIZE]) < C[innerI][innerJ+HALF_BLOCK_SIZE] ? (C[innerI][k] + C[k][innerJ+HALF_BLOCK_SIZE]) : C[innerI][innerJ+HALF_BLOCK_SIZE];

                C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = (C[innerI+HALF_BLOCK_SIZE][k] + C[k][innerJ+HALF_BLOCK_SIZE]) < C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] ? (C[innerI+HALF_BLOCK_SIZE][k] + C[k][innerJ+HALF_BLOCK_SIZE]) : C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
                __syncthreads(); // TODO: only phase one
            }

            dist[offset*(n+1) + matrixInnerI*n + matrixInnerJ] = C[innerI][innerJ];
            dist[offset*(n+1) + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ] = C[innerI+HALF_BLOCK_SIZE][innerJ];
            dist[offset*(n+1) + matrixInnerI*n + matrixInnerJ + HALF_BLOCK_SIZE] = C[innerI][innerJ+HALF_BLOCK_SIZE];
            dist[offset*(n+1) + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ + HALF_BLOCK_SIZE] = C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
        }
    }
}

__global__ void Phase2(int *dist, int Round, int n) {
    const int i = blockIdx.x; // "i" in n block in one row
    if (i == Round) return;

    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int diagonalOffset = BLOCK_SIZE * Round;

    __shared__ int Diagonal[64][64];
    __shared__ int A[64][64];
    __shared__ int B[64][64];

    for (int totalOffsetI=0; totalOffsetI<BLOCK_SIZE; totalOffsetI+=64) {
        int matrixInnerI = innerI + totalOffsetI;
        for (int totalOffsetJ=0; totalOffsetJ<BLOCK_SIZE; totalOffsetJ+=64) {
            int matrixInnerJ = innerJ + totalOffsetJ;

            A[innerI][innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ];
            A[innerI+HALF_BLOCK_SIZE][innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ];
            A[innerI][innerJ+HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ + HALF_BLOCK_SIZE];
            A[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ+HALF_BLOCK_SIZE];

            B[innerI][innerJ] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ];
            B[innerI+HALF_BLOCK_SIZE][innerJ] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ];
            B[innerI][innerJ+HALF_BLOCK_SIZE] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ+HALF_BLOCK_SIZE];
            B[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ+HALF_BLOCK_SIZE];

            Diagonal[innerI][innerJ] = dist[diagonalOffset*(n+1) + matrixInnerI*n + matrixInnerJ];
            Diagonal[innerI+HALF_BLOCK_SIZE][innerJ] = dist[diagonalOffset*(n+1) + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ];
            Diagonal[innerI][innerJ+HALF_BLOCK_SIZE] = dist[diagonalOffset*(n+1) + matrixInnerI*n + matrixInnerJ+HALF_BLOCK_SIZE];
            Diagonal[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = dist[diagonalOffset*(n+1) + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ+HALF_BLOCK_SIZE];
        
            __syncthreads();

            for (int k = 0; k < 64; k++) {

                A[innerI][innerJ] = (A[innerI][k] + Diagonal[k][innerJ]) < A[innerI][innerJ] ? (A[innerI][k] + Diagonal[k][innerJ]) : A[innerI][innerJ];

                A[innerI+HALF_BLOCK_SIZE][innerJ] = (A[innerI+HALF_BLOCK_SIZE][k] + Diagonal[k][innerJ]) < A[innerI+HALF_BLOCK_SIZE][innerJ] ? (A[innerI+HALF_BLOCK_SIZE][k] + Diagonal[k][innerJ]) : A[innerI+HALF_BLOCK_SIZE][innerJ];

                A[innerI][innerJ+HALF_BLOCK_SIZE] = (A[innerI][k] + Diagonal[k][innerJ+HALF_BLOCK_SIZE]) < A[innerI][innerJ+HALF_BLOCK_SIZE] ? (A[innerI][k] + Diagonal[k][innerJ+HALF_BLOCK_SIZE]) : A[innerI][innerJ+HALF_BLOCK_SIZE];
                
                A[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = (A[innerI+HALF_BLOCK_SIZE][k] + Diagonal[k][innerJ+HALF_BLOCK_SIZE]) < A[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] ? (A[innerI+HALF_BLOCK_SIZE][k] + Diagonal[k][innerJ+HALF_BLOCK_SIZE]) : A[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];

                B[innerI][innerJ] = (Diagonal[innerI][k] + B[k][innerJ]) < B[innerI][innerJ] ? (Diagonal[innerI][k] + B[k][innerJ]) : B[innerI][innerJ];

                B[innerI+HALF_BLOCK_SIZE][innerJ] = (Diagonal[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ]) < B[innerI+HALF_BLOCK_SIZE][innerJ] ? (Diagonal[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ]) : B[innerI+HALF_BLOCK_SIZE][innerJ];

                B[innerI][innerJ+HALF_BLOCK_SIZE] = (Diagonal[innerI][k] + B[k][innerJ+HALF_BLOCK_SIZE]) < B[innerI][innerJ+HALF_BLOCK_SIZE] ? (Diagonal[innerI][k] + B[k][innerJ+HALF_BLOCK_SIZE]) : B[innerI][innerJ+HALF_BLOCK_SIZE];
                
                B[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = (Diagonal[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ+HALF_BLOCK_SIZE]) < B[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] ? (Diagonal[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ+HALF_BLOCK_SIZE]) : B[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
            }

            dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ] = A[innerI][innerJ];
            dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ] = A[innerI+HALF_BLOCK_SIZE][innerJ];
            dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ + HALF_BLOCK_SIZE] = A[innerI][innerJ+HALF_BLOCK_SIZE];
            dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ+HALF_BLOCK_SIZE] = A[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE];

            dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ] = B[innerI][innerJ];
            dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ] = B[innerI+HALF_BLOCK_SIZE][innerJ];
            dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ+HALF_BLOCK_SIZE] = B[innerI][innerJ+HALF_BLOCK_SIZE];
            dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ+HALF_BLOCK_SIZE] = B[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE];
        }
    }
}

__global__ void Phase3(int *dist, int Round, int n) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    if (i == Round && j == Round) return;

    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;

    __shared__ int A[64][64];
    __shared__ int B[64][64];
    __shared__ int C[64][64];

    for (int totalOffsetI=0; totalOffsetI<BLOCK_SIZE; totalOffsetI+=64) {
        int matrixInnerI = innerI + totalOffsetI;
        for (int totalOffsetJ=0; totalOffsetJ<BLOCK_SIZE; totalOffsetJ+=64) {
            int matrixInnerJ = innerJ + totalOffsetJ;
  
            C[innerI][innerJ] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ];
            C[innerI+HALF_BLOCK_SIZE][innerJ] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ];
            C[innerI][innerJ+HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ+HALF_BLOCK_SIZE];
            C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ+HALF_BLOCK_SIZE];


            A[innerI][innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ];
            A[innerI+HALF_BLOCK_SIZE][innerJ] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ];
            A[innerI][innerJ+HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ + HALF_BLOCK_SIZE];
            A[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ + HALF_BLOCK_SIZE];

            B[innerI][innerJ] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ];
            B[innerI+HALF_BLOCK_SIZE][innerJ] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ];
            B[innerI][innerJ+HALF_BLOCK_SIZE] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ+HALF_BLOCK_SIZE];
            B[innerI + HALF_BLOCK_SIZE][innerJ + HALF_BLOCK_SIZE] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ+HALF_BLOCK_SIZE];
        
            __syncthreads();

            #pragma unroll 32
            for (int k = 0; k < 64; k++) {
                C[innerI][innerJ] = (A[innerI][k] + B[k][innerJ]) < C[innerI][innerJ] ? (A[innerI][k] + B[k][innerJ]) : C[innerI][innerJ];

                C[innerI+HALF_BLOCK_SIZE][innerJ] = (A[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ]) < C[innerI+HALF_BLOCK_SIZE][innerJ] ? (A[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ]) : C[innerI+HALF_BLOCK_SIZE][innerJ];

                C[innerI][innerJ+HALF_BLOCK_SIZE] = (A[innerI][k] + B[k][innerJ+HALF_BLOCK_SIZE]) < C[innerI][innerJ+HALF_BLOCK_SIZE] ? (A[innerI][k] + B[k][innerJ+HALF_BLOCK_SIZE]) : C[innerI][innerJ+HALF_BLOCK_SIZE];
                
                C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] = (A[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ+HALF_BLOCK_SIZE]) < C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE] ? (A[innerI+HALF_BLOCK_SIZE][k] + B[k][innerJ+HALF_BLOCK_SIZE]) : C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
            }

            dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ] = C[innerI][innerJ];
            dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ] = C[innerI+HALF_BLOCK_SIZE][innerJ];
            dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + matrixInnerI*n + matrixInnerJ+HALF_BLOCK_SIZE] = C[innerI][innerJ+HALF_BLOCK_SIZE];
            dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (matrixInnerI+HALF_BLOCK_SIZE)*n + matrixInnerJ+HALF_BLOCK_SIZE] = C[innerI+HALF_BLOCK_SIZE][innerJ+HALF_BLOCK_SIZE];
        }
    }
}