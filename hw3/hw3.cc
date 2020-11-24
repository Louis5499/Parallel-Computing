// C++ Program for Floyd Warshall Algorithm  
// Finding:
// 1. Don't allocate 1d array to store values, use vector: 69s ->47s
// 2. Don't use OpenMP, it would be slower.
#include <bits/stdc++.h> 
#include <omp.h>
#include <pthread.h>
using namespace std;

#define INF 99999
pthread_barrier_t barr;

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

  int iLower = V * realThreadId / ncpus;
  int iUpper = V * (realThreadId + 1) / ncpus;

  for (int k=0; k < V; k++) {
    for (int i = iLower; i < iUpper; i++) {
      for (int j = 0; j < V; j++) {
        if (dist[i][k] + dist[k][j] < dist[i][j]) {
          dist[i][j] = dist[i][k] + dist[k][j];
        }
      }
    }
    pthread_barrier_wait(&barr);
  }

  return NULL;
}
  
// Driver code  
int main(int argc, char** argv) {
  /* detect how many CPUs are available */
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
  ncpus = CPU_COUNT(&cpu_set);
  printf("%d cpus available\n", ncpus);
  pthread_barrier_init(&barr, NULL, (unsigned)ncpus);

  std::ifstream f(argv[1]);
  if (not f) {
      throw reprintf("failed to open file: %s", argv[1]);
  }
  f.seekg(0, std::ios_base::end);
  f.seekg(0, std::ios_base::beg);
  int E;
  f.read((char*)&V, sizeof V);
  // printf("V = %d\n", V);
  f.read((char*)&E, sizeof E);
  // printf("E = %d\n", E);

  vector<int> row;
  row.assign(V, 0);//配置一個 row 的大小
  dist.assign(V, row);
  for (int i=0;i<V;i++) {
    for (int j=0;j<V;j++) {
      if (i == j) dist[i][j] = 0;
      else dist[i][j] = INF;
    }
  }

  for (int i = 0; i < E; i++) {
    int e[3];
    f.read((char*)e, sizeof e);
    dist[e[0]][e[1]] = e[2];
  }

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

  std::ofstream fout(argv[2]);
  for(int i=0; i<V; i++){
    for(int j=0; j<V; j++){
        fout.write((char *)(&dist[i][j]), sizeof(int));
    }
  }
  fout.close();
	pthread_exit(NULL);
  return 0;  
}  