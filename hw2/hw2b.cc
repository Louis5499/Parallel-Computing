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
    int curHeight, avgHeight, modHeight = 0;
    if (rank >= height) {
        avgHeight = 1;
    } else {
        modHeight = height%mpiSize;
        avgHeight = height/mpiSize;
        curHeight = (rank < modHeight) ? avgHeight + 1 : avgHeight;
    }

    /* allocate memory for image */
    int imagesize = width * height;
    int* image = (int*)malloc(imagesize * sizeof(int));
    assert(image);

    int* curImage = (int *)malloc(width*curHeight * sizeof(int));

    double y0Offset = ((upper - lower) / height);
    double x0Offset = ((right - left) / width);

    // One node calculation
    #pragma omp parallel num_threads(ncpus)
    {
        #pragma omp for schedule(dynamic)
        for (int heightIdx=0; heightIdx < curHeight; heightIdx++) {
            int j = rank + mpiSize*heightIdx;
            double y0 = j*y0Offset + lower;
            int curImageIndex = heightIdx*width;
            for (int i = 0; i < width; ++i) {
                double x0 = i * x0Offset + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                curImage[curImageIndex + i] = repeats;
            }
        }
    }

    int* revcount = (int*)malloc(mpiSize*sizeof(int));
    int* displs = (int*)malloc(mpiSize*sizeof(int));
    displs[0] = 0;
    for(int i=0;i<mpiSize;++i){
      if (i<modHeight){
        revcount[i] = (avgHeight+1)*width;
        if(i+1<mpiSize) displs[i+1] = displs[i] + revcount[i];
      }
      else {
        revcount[i] = avgHeight*width;
        if(i+1<mpiSize) displs[i+1] = displs[i] + revcount[i];
      }
    }
    MPI_Gatherv(curImage, curHeight*width, MPI_INT, image, revcount, displs, MPI_INT, 0, MPI_COMM_WORLD);
    // // TODO: 因為一開始分配是縱向分配，導致需要再轉換。可改成一開始就橫向分配，也許效果較好。
    if(rank==0){
        /* index handle*/
        int* ans_image = (int*)malloc(imagesize * sizeof(int));
        int index = 0;
        for(int i=0;i<curHeight;i++){
            int ri = 0;
            int jplus = curHeight;
            for(int j=i;j<height && index < imagesize ;j+=jplus){
                int w0 = j*width;
                for(int w=0;w<width;w++){
                    ans_image[index] = image[w0 + w];
                    index++;
                }
                if(ri <modHeight)jplus = curHeight;
                else jplus = avgHeight;
                ri++;
            }
        }
        /* draw and cleanup */
        write_png(filename, iters, width, height, ans_image);
        free(ans_image);
    }

    /* draw and cleanup */
    // write_png(filename, iters, width, height, image);
    free(curImage);
    free(image);
    MPI_Finalize();
    return 0;
}


// TODO: use MPI OPEN FILE TO PARALLELLY WRITE DATA