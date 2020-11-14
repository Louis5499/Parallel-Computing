#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <iostream>
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <emmintrin.h>

using namespace std;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

union SsePacket {
    __m128d sseNum;
    double num[2];
};

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int rc, rank, mpiSize;
    rc = MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Distribute Workloads with regard to Height 
    int curHeightNum, avgHeight, modHeight = 0;
    if (rank >= height) {
        avgHeight = 1;
    } else {
        modHeight = height%mpiSize;
        avgHeight = height/mpiSize;
        curHeightNum = (rank < modHeight) ? avgHeight + 1 : avgHeight;
    }
    int frontPadding = 0;
    for (int i=0;i<rank;i++) {
        frontPadding += (i < modHeight) ? avgHeight + 1 : avgHeight;
    }

    /* allocate memory for image */
    int imagesize = width * height;
    int* image = (int*)malloc(imagesize * sizeof(int));
    assert(image);

    int* curImage = (int *)malloc(width*curHeightNum * sizeof(int));

    double y0Offset = ((upper - lower) / height);
    double x0Offset = ((right - left) / width);

    #pragma omp parallel num_threads(ncpus)
    {
        #pragma omp for schedule(dynamic)
        for (int heightIdx=0; heightIdx < curHeightNum; heightIdx++) {
            int curHeight = frontPadding + heightIdx;
            int curImageIndex = heightIdx*width;

            double y0 = curHeight * y0Offset + lower;

            int isEven = (width%2 == 0);
            int SSEWidth = isEven ? width : width - 1;
            const double threshold = 4.0;
            const double yMul = 2.0;
            __m128d yMulSse = _mm_set_pd(yMul, yMul);
            __m128d y0Sse = _mm_set_pd(y0, y0);

            int finish[2] = {1, 1};
            int block[2] = {0, 0};

            int repeats1 = 0;
            int repeats2 = 0;
            SsePacket x0;
            SsePacket x;
            SsePacket y;
            SsePacket length_square;
            int widthIdx = -1;
            int widthIdx1 = 0;
            int widthIdx2 = 0;

            while (!block[0] && !block[1]) {
                if (finish[0] || finish[1]) {
                    widthIdx++;
                    if (finish[0]) {
                        x0.num[0] =  widthIdx * x0Offset + left;
                        x.num[0] = 0;
                        y.num[0] = 0;
                        length_square.num[0] = 0;
                        repeats1 = 0;
                        widthIdx1 = widthIdx;
                        finish[0] = 0;
                    } else if (finish[1]) {
                        x0.num[1] =  widthIdx * x0Offset + left;
                        x.num[1] = 0;
                        y.num[1] = 0;
                        length_square.num[1] = 0;
                        repeats2 = 0;
                        widthIdx2 = widthIdx;
                        finish[1] = 0;
                    }
                }

                __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x.sseNum, x.sseNum), _mm_mul_pd(y.sseNum, y.sseNum)), x0.sseNum);
                y.sseNum = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(yMulSse, x.sseNum), y.sseNum), y0Sse);
                x.sseNum = temp;
                length_square.sseNum = _mm_add_pd(_mm_mul_pd(x.sseNum, x.sseNum),_mm_mul_pd(y.sseNum, y.sseNum));
                ++repeats1;
                ++repeats2;

                if (length_square.num[0] >= threshold || repeats1 >= iters) {
                    // Check two situation:
                    // 1. block -> If one has finished his width calculation, the other hasn't. We need to block finish one to stop calculating back image array.
                    // 2. finish -> If two simultaneously finish their jobs, we need to be careful.
                    // Since only "one" would be distribute next iteration information, the other would not. Hence, we still need to check the current state is final state, not final state + 1 (不要的)。
                    if (!block[0] && finish[0] == 0) {
                        curImage[curImageIndex + widthIdx1] = repeats1;
                        finish[0] = 1;
                    }
                    if (widthIdx + 1 >= width) block[0] = 1;
                }
                if (length_square.num[1] >= threshold || repeats2 >= iters) {
                    if (!block[1] && finish[1] == 0) {
                        curImage[curImageIndex + widthIdx2] = repeats2;
                        finish[1] = 1;
                    }
                    if (widthIdx + 1 >= width) block[1] = 1;
                }
            }

            if (!finish[0]) {
                while (repeats1 < iters && length_square.num[0] < 4) {
                    double temp = x.num[0] * x.num[0] - y.num[0] * y.num[0] + x0.num[0];
                    y.num[0] = 2 * x.num[0] * y.num[0] + y0;
                    x.num[0] = temp;
                    length_square.num[0] = x.num[0] * x.num[0] + y.num[0] * y.num[0];
                    ++repeats1;
                }
                curImage[curImageIndex + widthIdx1] = repeats1;
            }

            if (!finish[1]) {
                while (repeats2 < iters && length_square.num[1] < 4) {
                    double temp = x.num[1] * x.num[1] - y.num[1] * y.num[1] + x0.num[1];
                    y.num[1] = 2 * x.num[1] * y.num[1] + y0;
                    x.num[1] = temp;
                    length_square.num[1] = x.num[1] * x.num[1] + y.num[1] * y.num[1];
                    ++repeats2;
                }
                curImage[curImageIndex + widthIdx2] = repeats2;
            }
        }
    }

    int* revcount = (int*)malloc(mpiSize*sizeof(int));
    int* displs = (int*)malloc(mpiSize*sizeof(int));
    displs[0] = 0;
    for(int i=0; i<mpiSize; i++){
      if (i < modHeight){
        revcount[i] = (avgHeight+1)*width;
        if(i+1 < mpiSize) displs[i+1] = displs[i] + revcount[i];
      }
      else {
        revcount[i] = avgHeight*width;
        if(i+1 < mpiSize) displs[i+1] = displs[i] + revcount[i];
      }
    }
    MPI_Gatherv(curImage, curHeightNum*width, MPI_INT, image, revcount, displs, MPI_INT, 0, MPI_COMM_WORLD);
    // 因為一開始分配是縱向分配，導致需要再轉換。可改成一開始就橫向分配，也許效果較好。
    if(rank==0) {
        // int* ansImage = (int*)malloc(imagesize * sizeof(int));
        // int index = 0;
        // for(int i=0;i<curHeightNum;i++){
        //     int realHeightIdx = 0;
        //     int jumpOffset = curHeightNum;
        //     for(int j=i; j<height && index < imagesize; j+=jumpOffset){
        //         int w0 = j*width;
        //         for(int w=0;w<width;w++){
        //             ansImage[index] = image[w0 + w];
        //             index++;
        //         }
        //         if(realHeightIdx < modHeight) jumpOffset = curHeightNum;
        //         else jumpOffset = avgHeight;
        //         realHeightIdx++;
        //     }
        // }
        // write_png(filename, iters, width, height, ansImage);
        // free(ansImage);
        write_png(filename, iters, width, height, image);
    }

    /* draw and cleanup */
    // write_png(filename, iters, width, height, image);
    free(curImage);
    free(image);
    MPI_Finalize();
    return 0;
}


// TODO: use MPI OPEN FILE TO PARALLELLY WRITE DATA