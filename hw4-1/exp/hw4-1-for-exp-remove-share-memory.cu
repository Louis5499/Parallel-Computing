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
}

__global__ void Phase1(int *dist, int Round, int n) {
    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int offset = BLOCK_SIZE * Round;

    int C[BLOCK_SIZE][BLOCK_SIZE]; // 2d

    // Every thread read its own value
    // how index: blockIndex (to next diagonal block) + innerBlockIndex (every thread has its own index)
    for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
        for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
            C[innerI+iOffset][innerJ+jOffset] = dist[offset*(n+1) + (innerI+iOffset)*n + innerJ+jOffset];
        }
    }
    
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
            for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
                if ((C[innerI+iOffset][k] + C[k][innerJ+jOffset]) < C[innerI+iOffset][innerJ+jOffset]) {
                    C[innerI+iOffset][innerJ+jOffset] = C[innerI+iOffset][k] + C[k][innerJ+jOffset];
                }
            }
        }
        __syncthreads(); // TODO: only phase one
    }

    for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
        for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
            dist[offset*(n+1) + (innerI+iOffset)*n + innerJ+jOffset] = C[innerI+iOffset][innerJ+jOffset];
        }
    }
}

__global__ void Phase2(int *dist, int Round, int n) {
    const int i = blockIdx.x; // "i" in n block in one row
    if (i == Round) return;

    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int diagonalOffset = BLOCK_SIZE * Round;

    int Diagonal[BLOCK_SIZE][BLOCK_SIZE];
    int A[BLOCK_SIZE][BLOCK_SIZE];
    int B[BLOCK_SIZE][BLOCK_SIZE];

    for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
        for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
            A[innerI+iOffset][innerJ+jOffset] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+iOffset)*n + innerJ+jOffset];
            B[innerI+iOffset][innerJ+jOffset] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (innerI+iOffset)*n + innerJ+jOffset];
            Diagonal[innerI+iOffset][innerJ+jOffset] = dist[diagonalOffset*(n+1) + (innerI+iOffset)*n + innerJ+jOffset];
        }
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
            for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
                if ((A[innerI+iOffset][k] + Diagonal[k][innerJ+jOffset]) < A[innerI+iOffset][innerJ+jOffset]) {
                    A[innerI+iOffset][innerJ+jOffset] = A[innerI+iOffset][k] + Diagonal[k][innerJ+jOffset];
                }
                if ((Diagonal[innerI+iOffset][k] + B[k][innerJ+jOffset]) < B[innerI+iOffset][innerJ+jOffset]) {
                    B[innerI+iOffset][innerJ+jOffset] = Diagonal[innerI+iOffset][k] + B[k][innerJ+jOffset];
                }
            }
        }
    }

    for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
        for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
            dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+iOffset)*n + innerJ+jOffset] = A[innerI+iOffset][innerJ+jOffset];
            dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + (innerI+iOffset)*n + innerJ+jOffset] = B[innerI+iOffset][innerJ+jOffset];
        }
    }
}

__global__ void Phase3(int *dist, int Round, int n) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    if (i == Round && j == Round) return;

    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;

    int A[BLOCK_SIZE][BLOCK_SIZE];
    int B[BLOCK_SIZE][BLOCK_SIZE];
    int C[BLOCK_SIZE][BLOCK_SIZE];
  
    for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
        for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
            C[innerI+iOffset][innerJ+jOffset] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+iOffset)*n + innerJ+jOffset];
            A[innerI+iOffset][innerJ+jOffset] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + (innerI+iOffset)*n + innerJ+jOffset];
            B[innerI+iOffset][innerJ+jOffset] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+iOffset)*n + innerJ+jOffset];
        }
    }
  
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
            for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
                if ((A[innerI+iOffset][k] + B[k][innerJ+jOffset]) < C[innerI+iOffset][innerJ+jOffset]) {
                    C[innerI+iOffset][innerJ+jOffset] = A[innerI+iOffset][k] + B[k][innerJ+jOffset];
                }
            }
        }
    }

    for (int iOffset=0; iOffset<BLOCK_SIZE; iOffset+=32) {
        for (int jOffset=0; jOffset<BLOCK_SIZE; jOffset+=32) {
            dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + (innerI+iOffset)*n + innerJ+jOffset] = C[innerI+iOffset][innerJ+jOffset];
        }
    }
}