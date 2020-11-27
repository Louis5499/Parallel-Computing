// C++ Program for Floyd Warshall Algorithm  
// Finding:
// 1. Don't allocate 1d array to store values, use vector: 69s ->47s
// 2. Don't use OpenMP, it would be slower.
#include <bits/stdc++.h> 
#include <omp.h>
#include <pthread.h>
using namespace std;

#define INF 1073741823
pthread_barrier_t barr;

// initialize time measurement
struct timespec start, timeEnd;
struct timespec totalStart, totalEnd;
double total_time=0.0, io_time=0.0, compute_time=0.0, input_time=0.0, output_time=0.0;
double heightestComputeTime = 0.0;
double lowestComputeTime = 999.0;

double timeDiff(struct timespec start, struct timespec timeEnd){
    // function used to measure time in nano resolution
    float output;
    float nano = 1000000000.0;
    if(timeEnd.tv_nsec < start.tv_nsec) output = ((timeEnd.tv_sec - start.tv_sec -1)+(nano+timeEnd.tv_nsec-start.tv_nsec)/nano);
    else output = ((timeEnd.tv_sec - start.tv_sec)+(timeEnd.tv_nsec-start.tv_nsec)/nano);
    return output;
}

using namespace std;

std::runtime_error reprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char* c_str;
    vasprintf(&c_str, fmt, ap);
    va_end(ap);
    std::runtime_error re(c_str);
    free(c_str);
    return re;
}

int V = 0, ncpus;
vector<vector<int>> dist;

void* calculate(void *arg) {
  int* threadid = (int*)arg;
  int realThreadId = *threadid;

  clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
  int iLower = V * realThreadId / ncpus;
  int iUpper = V * (realThreadId + 1) / ncpus;

  for (int k=0; k < V; k++) {
    // For Locality
    for (int i = iLower; i < iUpper; i++) {
      if (i == k || dist[i][k] == INF) continue;
      for (int j = 0; j < V; j++) {
        if (dist[i][k] + dist[k][j] < dist[i][j]) {
          dist[i][j] = dist[i][k] + dist[k][j];
        }
      }
    }
    pthread_barrier_wait(&barr);
  }

  clock_gettime(CLOCK_MONOTONIC, &timeEnd); // E---------------------------------------------------------------------------------
  compute_time += timeDiff(start, timeEnd);
  return NULL;
}

int main(int argc, char** argv) {
  /* detect how many CPUs are available */
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
  ncpus = CPU_COUNT(&cpu_set);
  printf("%d cpus available\n", ncpus);
  pthread_barrier_init(&barr, NULL, (unsigned)ncpus);

  clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
  clock_gettime(CLOCK_MONOTONIC, &totalStart); // S---------------------------------------------------------------------------------

  std::ifstream f(argv[1]);
  if (not f) {
      throw reprintf("failed to open file: %s", argv[1]);
  }
  f.seekg(0, std::ios_base::end);
  f.seekg(0, std::ios_base::beg);
  int E;
  f.read((char*)&V, sizeof V);
  printf("V = %d\n", V);
  f.read((char*)&E, sizeof E);
  printf("E = %d\n", E);

  vector<int> row;
  row.assign(V, INF);//配置一個 row 的大小
  dist.assign(V, row);
  for (int j=0;j<V;j++) {
    dist[j][j] = 0;
  }

  for (int i = 0; i < E; i++) {
    int e[3];
    f.read((char*)e, sizeof e);
    dist[e[0]][e[1]] = e[2];
  }

  clock_gettime(CLOCK_MONOTONIC, &timeEnd); // E---------------------------------------------------------------------------------
  input_time += timeDiff(start, timeEnd);

  pthread_t threads[ncpus];
  int ID[ncpus];

  for (int t=0;t<ncpus;t++) {
    ID[t] = t;
    pthread_create(&threads[t], NULL, calculate, &ID[t]);
  }

  for (int t=0;t<ncpus;t++) {
		pthread_join(threads[t], NULL);
	}

  // Print the result  
  // for (int i = 0; i < V; i++) {  
  //   for (int j = 0; j < V; j++) {  
  //     if (dist[i*V+j] == INF) cout << "INF" << "     ";  
  //     else cout << dist[i*V+j] << "     ";  
  //   }
  //   cout << endl;  
  // }

  clock_gettime(CLOCK_MONOTONIC, &start); // S---------------------------------------------------------------------------------
  std::ofstream fout(argv[2]);
  for(int i=0; i<V; i++){
    for(int j=0; j<V; j++){
        fout.write((char *)(&dist[i][j]), sizeof(int));
    }
  }
  fout.close();
  clock_gettime(CLOCK_MONOTONIC, &timeEnd); // S---------------------------------------------------------------------------------
  clock_gettime(CLOCK_MONOTONIC, &totalEnd); // S---------------------------------------------------------------------------------
  output_time += timeDiff(start, timeEnd);
  io_time = input_time + output_time;
  total_time = timeDiff(totalStart, totalEnd);
  printf("total_time: %f\ninput_time: %f\noutput_time: %f\ncompute_time: %f\navg compute_time: %f\n", total_time, input_time, output_time, compute_time, compute_time/ncpus);

	pthread_exit(NULL);
  return 0;  
}  