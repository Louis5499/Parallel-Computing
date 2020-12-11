#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

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
    n = original_n + (BLOCK_SIZE - ((original_n-1) % BLOCK_SIZE + 1));

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

    const int matrixSize =  n * n * sizeof(int);

    cudaHostRegister(Dist, matrixSize, cudaHostRegisterDefault);
    cudaMalloc(&dst, matrixSize);
	cudaMemcpy(dst, Dist, matrixSize, cudaMemcpyHostToDevice);

    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 phase4_grid(blocks, blocks, 1);

    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        Phase1<<<1, block_dim>>>(dst, r, n);

        /* Phase 2*/
        Phase2<<<blocks, block_dim>>>(dst, r, n);

        /* Phase 3*/
        Phase3<<<phase4_grid, block_dim>>>(dst, r, n);
    }

    cudaMemcpy(Dist, dst, matrixSize, cudaMemcpyDeviceToHost);
	cudaFree(dst);
}


inline __device__ void BlockCalc(int* C, int* A, int* B, int bj, int bi) {
    for (int k = 0; k < BLOCK_SIZE; k++) {
      int sum = A[bi*BLOCK_SIZE + k] + B[k*BLOCK_SIZE + bj];
    //   printf("New Added Element[%d][%d]: %d   Element[%d][%d]: %d  Combine Value: %d | Original Value: %d\n", bi, k, A[bi*BLOCK_SIZE + k], k, bj, B[k*BLOCK_SIZE + bj], sum, C[bi*BLOCK_SIZE + bj]);
      if (C[bi*BLOCK_SIZE + bj] > sum) {
        C[bi*BLOCK_SIZE + bj] = sum;
      }
      __syncthreads();
    }
  }

__global__ void Phase1(int *dist, int Round, int n) {
    const int innerI = threadIdx.y;
    const int innerJ = threadIdx.x;
    const int offset = BLOCK_SIZE * Round;

    __shared__ int C[BLOCK_SIZE * BLOCK_SIZE];

    // Every thread read its own value
    // how index: blockIndex (to next diagonal block) + innerBlockIndex (every thread has its own index)
    C[innerI * BLOCK_SIZE + innerJ] = dist[offset*(n+1) + innerI*n + innerJ];
    __syncthreads();
    BlockCalc(C, C, C, innerI, innerJ);
    __syncthreads();
    dist[offset*(n+1) + innerI*n + innerJ] = C[innerI * BLOCK_SIZE + innerJ];
}

__global__ void Phase2(int *dist, int Round, int n) {
    const int i = blockIdx.x; // "i" in n block in one row
    const int bi = threadIdx.y;
    const int bj = threadIdx.x;
    const int diagonalOffset = BLOCK_SIZE * Round;
  
    if (i == Round) return;
  
    __shared__ int A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int B[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int C[BLOCK_SIZE * BLOCK_SIZE];
  
    C[bi*BLOCK_SIZE + bj] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + bi*n + bj];
    B[bi*BLOCK_SIZE + bj] = dist[diagonalOffset*(n+1) + bi*n + bj]; // diagonalValue
  
    __syncthreads();
  
    BlockCalc(C, C, B, bi, bj);
  
    __syncthreads();
  
    dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + bi*n + bj] = C[bi*BLOCK_SIZE + bj];
  
    // Phase 2 1/2
  
    C[bi*BLOCK_SIZE + bj] = dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + bi*n + bj];
    A[bi*BLOCK_SIZE + bj] = dist[diagonalOffset*(n+1) + bi*n + bj]; // diagonalValue
  
    __syncthreads();
  
    BlockCalc(C, A, C, bi, bj);
  
    __syncthreads();
  
    // Block C is the only one that could be changed
    dist[Round*BLOCK_SIZE*n + i*BLOCK_SIZE + bi*n + bj] = C[bi*BLOCK_SIZE + bj];
}

__global__ void Phase3(int *dist, int Round, int n) {
    const unsigned int j = blockIdx.x;
    const unsigned int i = blockIdx.y;
    const unsigned int bi = threadIdx.y;
    const unsigned int bj = threadIdx.x;
  
    if (i == Round && j == Round) return;
    __shared__ int A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int B[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int C[BLOCK_SIZE * BLOCK_SIZE];
  
    C[bi*BLOCK_SIZE + bj] = dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + bi*n + bj];
    A[bi*BLOCK_SIZE + bj] = dist[i*BLOCK_SIZE*n + Round*BLOCK_SIZE + bi*n + bj];
    B[bi*BLOCK_SIZE + bj] = dist[Round*BLOCK_SIZE*n + j*BLOCK_SIZE + bi*n + bj];
  
    __syncthreads();
  
    BlockCalc(C, A, B, bi, bj);
  
    __syncthreads();
  
    dist[i*BLOCK_SIZE*n + j*BLOCK_SIZE + bi*n + bj] = C[bi*BLOCK_SIZE + bj];
}

// __global__ void cal(int *dist, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n) {
//     int block_end_x = block_start_x + block_height;
//     int block_end_y = block_start_y + block_width;

//     for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
//         for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
//             // To calculate B*B elements in the block (b_i, b_j)
//             // For each block, it need to compute B times
//             for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
//                 // To calculate original index of elements in the block (b_i, b_j)
//                 // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
//                 int block_internal_start_x = b_i * B;
//                 int block_internal_end_x = (b_i + 1) * B;
//                 int block_internal_start_y = b_j * B;
//                 int block_internal_end_y = (b_j + 1) * B;

//                 if (block_internal_end_x > n) block_internal_end_x = n;
//                 if (block_internal_end_y > n) block_internal_end_y = n;

//                 for (int i = block_internal_start_x + blockIdx.x; i < block_internal_end_x; i += gridDim.x) {
//                     for (int j = block_internal_start_y + threadIdx.x; j < block_internal_end_y; j += blockDim.x) {
//                         if (dist[i*n + k] + dist[k*n + j] < dist[i*n + j]) {
//                             dist[i*n + j] = dist[i*n + k] + dist[k*n + j];
//                         }
//                         __syncthreads();
//                     }
//                 }
//             }
//         }
//     }
// }