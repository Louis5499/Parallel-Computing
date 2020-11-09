#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <emmintrin.h>

unsigned long long ncpus = 0;

int iters = 0;
int width = 0;
int height = 0;
double upper = 0.0;
double lower = 0.0;
double left = 0.0;
double right = 0.0;
int* image = NULL;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

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

int curHeightIndex = 0;
int getHeightPosition() {
    if (curHeightIndex < height) return curHeightIndex++;
    else return -1;
}

double y0Offset = 0;
double x0Offset = 0;

void* calcPixelValue(void *arg) {
    int* threadid = (int*)arg;
    int realThreadId = *threadid;
    int curHeight = 0;

    while(true) {
        pthread_mutex_lock(&mutex);
        curHeight = getHeightPosition();
        pthread_mutex_unlock(&mutex);
        if (curHeight == -1) break;

        // printf("threadID: %d    curHeight: %d\n", realThreadId, curHeight);
        double y0 = curHeight * y0Offset + lower;
        // for (int i = 0; i < width; ++i) {
        //     double x0 = i * x0Offset + left;

        //     int repeats = 0;
        //     double x = 0;
        //     double y = 0;
        //     double length_squared = 0;
        //     while (repeats < iters && length_squared < 4) {
        //         double temp = x * x - y * y + x0;
        //         y = 2 * x * y + y0;
        //         x = temp;
        //         length_squared = x * x + y * y;
        //         ++repeats;
        //     }

        //     image[curHeight * width + i] = repeats;
        // }

        int isEven = (width%2 == 0);
        int SSEWidth = isEven ? width : width - 1;
        const double threshold = 4.0;
        // __m128d thresholdSse = _mm_set_pd(threshold, threshold);
        const double yMul = 2.0;
        __m128d yMulSse = _mm_set_pd(yMul, yMul);

        __m128d y0Sse = _mm_set_pd(y0, y0);
        for (int i=0; i<SSEWidth; i+=2) {
            double x0_1 = i * x0Offset + left;
            double x0_2 = (i+1) * x0Offset + left;
            __m128d x0Sse = _mm_set_pd(x0_1, x0_2);

            int repeats = 0;
            double x1 = 0;
            double x2 = 0;
            double y1 = 0;
            double y2 = 0;
            double length_squared1 = 0;
            double length_squared2 = 0;
            __m128d lengthSquareSse = _mm_set_pd(length_squared1, length_squared2);
            __m128d xSse = _mm_set_pd(x1, x2);
            __m128d ySse = _mm_set_pd(y1, y2);

            int repeats1 = -1;
            int repeats2 = -1;
            while (repeats < iters && (repeats1 == -1 || repeats2 == -1)) {
                __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(xSse, xSse), _mm_mul_pd(ySse, ySse)), x0Sse);
                ySse = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(yMulSse, xSse), ySse), y0Sse);
                xSse = temp;
                lengthSquareSse = _mm_add_pd(_mm_mul_pd(xSse, xSse),_mm_mul_pd(ySse, ySse));

                length_squared1 = _mm_cvtsd_f64(_mm_unpackhi_pd(lengthSquareSse, lengthSquareSse));
                if (length_squared1 > threshold && repeats1 == -1) repeats1 = repeats; 
                length_squared2 = _mm_cvtsd_f64(lengthSquareSse);
                if (length_squared2 > threshold && repeats2 == -1) repeats2 = repeats;
                ++repeats;
            }

            image[curHeight * width + i] = repeats1;
            image[curHeight * width + i + 1] = repeats2;
        }

        if (!isEven) {
            double x0 = (width-1) * x0Offset + left;

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

            image[curHeight * width + (width-1)] = repeats;
        }
    }

    return NULL;
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    pthread_t threads[ncpus];
    int ID[ncpus];

    pthread_mutex_init(&mutex, 0);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    y0Offset = ((upper - lower) / height);
    x0Offset = ((right - left) / width);

    /* mandelbrot set */
    for (int t=0;t<ncpus;t++) {
        ID[t] = t+1;
        pthread_create(&threads[t], NULL, calcPixelValue, &ID[t]);
    }

    for (int t=0;t<ncpus;t++) {
		pthread_join(threads[t], NULL);
	}

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

    pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);
}
