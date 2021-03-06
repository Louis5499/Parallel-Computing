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
#define BATCH_SIZE 50

unsigned long long ncpus = 0;

int iters = 0;
int width = 0;
int height = 0;
double upper = 0.0;
double lower = 0.0;
double left = 0.0;
double right = 0.0;
int* image = NULL;

// initialize time measurement
struct timespec start, end;
double compute_time=0.0, io_time=0.0, comm_time=0.0;
double heightestComputeTime = 0.0;
double lowestComputeTime = 999.0;

double timeDiff(struct timespec start, struct timespec end){
    // function used to measure time in nano resolution
    float output;
    float nano = 1000000000.0;
    if(end.tv_nsec < start.tv_nsec) output = ((end.tv_sec - start.tv_sec -1)+(nano+end.tv_nsec-start.tv_nsec)/nano);
    else output = ((end.tv_sec - start.tv_sec)+(end.tv_nsec-start.tv_nsec)/nano);
    return output;
}

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

int curLowerWidth = -BATCH_SIZE;
int curHigherWidth = 0;
int curHeightIndex = 0;
int getHeightPosition() {
    if (curHigherWidth + 1 > width) {
        curHeightIndex++;
        if (curHeightIndex >= height) return -1;
        curLowerWidth = 0;
        curHigherWidth = BATCH_SIZE;
        return curHeightIndex;
    } else {
        curLowerWidth += BATCH_SIZE;
        curHigherWidth = ((curHigherWidth + BATCH_SIZE) > width ? width : curHigherWidth + BATCH_SIZE);
        return curHeightIndex;
    }
}

double y0Offset = 0;
double x0Offset = 0;

union SsePacket {
    __m128d sseNum;
    double num[2];
};

void* calcPixelValue(void *arg) {
    int* threadid = (int*)arg;
    int realThreadId = *threadid;
    int curHeight = 0;
    int ourLowerWidth = 0;
    int ourHigherWidth = 0;

    // Fot time calculation
    double threadComputeTime = 0.0;

    while(true) {
        clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
        pthread_mutex_lock(&mutex);
        curHeight = getHeightPosition();
        ourLowerWidth = curLowerWidth;
        ourHigherWidth = curHigherWidth;
        pthread_mutex_unlock(&mutex);
        if (curHeight == -1) break;

        // printf("threadID: %d    curHeight: %d\n", realThreadId, curHeight);
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
        int widthIdx = ourLowerWidth-1;
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
                    image[curHeight * width + widthIdx1] = repeats1;
                    finish[0] = 1;
                }
                if (widthIdx + 1 >= ourHigherWidth) block[0] = 1;
            }
            if (length_square.num[1] >= threshold || repeats2 >= iters) {
                if (!block[1] && finish[1] == 0) {
                    image[curHeight * width + widthIdx2] = repeats2;
                    finish[1] = 1;
                }
                if (widthIdx + 1 >= ourHigherWidth) block[1] = 1;
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
            image[curHeight * width + widthIdx1] = repeats1;
        }

        if (!finish[1]) {
            while (repeats2 < iters && length_square.num[1] < 4) {
                double temp = x.num[1] * x.num[1] - y.num[1] * y.num[1] + x0.num[1];
                y.num[1] = 2 * x.num[1] * y.num[1] + y0;
                x.num[1] = temp;
                length_square.num[1] = x.num[1] * x.num[1] + y.num[1] * y.num[1];
                ++repeats2;
            }
            image[curHeight * width + widthIdx2] = repeats2;
        }
        clock_gettime(CLOCK_MONOTONIC, &end); // S---------------------------------------------------------------------------------
        compute_time += timeDiff(start, end);
        threadComputeTime += timeDiff(start, end);
    }

    if (threadComputeTime > heightestComputeTime) heightestComputeTime = threadComputeTime;
    else if (threadComputeTime < lowestComputeTime) lowestComputeTime = threadComputeTime;

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

    printf("comm_time: %f\nio_time: %f\ncompute_time: %f\n", comm_time, io_time, compute_time);
    printf("avg: %f\nlargest: %f\nlowest: %f\n", compute_time/ncpus, heightestComputeTime, lowestComputeTime);

    pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);
}
