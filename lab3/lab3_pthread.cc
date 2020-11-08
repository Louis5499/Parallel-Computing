#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

struct arg_struct {
    int threadid;
 		unsigned long long r;
		unsigned long long k;
		int totalThread;
};
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

unsigned long long total = 0;
unsigned long long r = 0;
unsigned long long k = 0;
unsigned long long ncpus = 0;

void* hello(void *arg) {
		unsigned long long pixels = 0;
		int *threadid = static_cast<int*>(arg);
		int realValue = *threadid;

    for (unsigned long long x = realValue; x < r; x+=ncpus) {
			unsigned long long y = ceil(sqrtl(r*r - x*x));
			pixels += y;
			// Largely benefit!
			if (pixels % 100000 == 0) pixels %= k;
		}

		pthread_mutex_lock(&mutex);
		total += pixels;
		pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	pthread_mutex_init(&mutex, 0);

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
	pthread_t threads[ncpus];

	int rc;
	int ID[ncpus];
	int t;
	void *midResult;
	for (t = 0; t < ncpus; t++) {
			ID[t] = t;
			// printf("In main: creating thread %d\n", t);
			// We should grab the integer in ID array to pass through pthread_create
			rc = pthread_create(&threads[t], NULL, hello, (void*)&ID[t]);
			if (rc) {
					printf("ERROR; return code from pthread_create() is %d\n", rc);
					exit(-1);
			}
	}

	for (t=0;t<ncpus;t++) {
		pthread_join(threads[t], NULL);
	}

	printf("%llu\n", (4 * total) % k);
	pthread_mutex_destroy(&mutex);

	pthread_exit(NULL);
}
