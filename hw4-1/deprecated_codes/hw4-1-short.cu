#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 64
#define HALF_BLOCK_SIZE 32

using namespace std;

const int INF = ((1 << 30) - 1);
const short SMALL_INF = 9999;
// const int V = 50010;
void input(char* infile);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
__global__ void Phase1(short *dist, int Round, int n);
__global__ void Phase2(short *dist, int Round, int n);
__global__ void Phase3(short *dist, int Round, int n);

int original_n, n, m;
short* Dist = NULL;

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
    cout << "intput n: " << original_n << endl;

    // make n % BLOCK_SIZE == 0
    n = original_n + (BLOCK_SIZE - (original_n%BLOCK_SIZE));

    Dist = (short*) malloc(sizeof(short)*n*n);
    // cudaMallocHost((void **)&Dist, sizeof(int)*n*n);

    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
            if (i == j) {
                Dist[i*n+j] = 0;
            } else {
                Dist[i*n+j] = SMALL_INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++ i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n+pair[1]] = short(pair[2]);
    }
    fclose(file);

}

void output(char *outFileName) {
    ofstream fout(outFileName);
    for(int i=0; i<original_n; i++){
      for(int j=0; j<original_n; j++){
          int val = (int) Dist[i*n + j];
          if (val >= SMALL_INF) fout.write((char *)&INF, sizeof(int));
          else fout.write((char *)(&val), sizeof(int));
      }
    }
    fout.close();

    // FILE *outfile = fopen(outFileName, "w");
	// for (int i = 0; i < original_n; ++i) {
	// 	for (int j = 0; j < original_n; ++j) {
    //         if (Dist[i*n+j] >= INF) Dist[i*n+j] = INF;
    //     }
	// 	fwrite(&Dist[i*n], sizeof(int), original_n, outfile);
	// }
    // fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    short* dst = NULL;

    const int matrixSize = n * n * sizeof(short);

    cudaHostRegister(Dist, matrixSize, cudaHostRegisterDefault);
    cudaMalloc(&dst, matrixSize);
	cudaMemcpy(dst, Dist, matrixSize, cudaMemcpyHostToDevice);

    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_dim(32, 32); // TODO: 32*32 -> padding (一定要開到 32x)
    dim3 grid_dim(blocks, blocks);

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
    printf("CUDA Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(Dist, dst, matrixSize, cudaMemcpyDeviceToHost);
	cudaFree(&dst);
}

__global__ void Phase1(short *dist, int Round, int n) {
    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int offset = BLOCK_SIZE * Round;

    __shared__ short C[BLOCK_SIZE][BLOCK_SIZE]; // 2d

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

__global__ void Phase2(short *dist, int Round, int n) {
    const int i = blockIdx.x; // "i" in n block in one row
    if (i == Round) return;

    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int diagonalOffset = BLOCK_SIZE * Round;

    __shared__ short Diagonal[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ short A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ short B[BLOCK_SIZE][BLOCK_SIZE];
  
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

__global__ void Phase3(short *dist, int Round, int n) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    if (i == Round && j == Round) return;

    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;

    __shared__ short A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ short B[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ short C[BLOCK_SIZE][BLOCK_SIZE];
  
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