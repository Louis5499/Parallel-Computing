// C++ Program for Floyd Warshall Algorithm  
#include <bits/stdc++.h> 
#include <omp.h>
#include <pthread.h>
using namespace std;

#define INF 99999
pthread_barrier_t barr;

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
int* dist = NULL;

void* calculate(void *arg) {
  int* threadid = (int*)arg;
  int realThreadId = *threadid;

  int i_lower = V * realThreadId / ncpus;
  int i_upper = V * (realThreadId + 1) / ncpus;

  for (int k=0; k < V; k++) {
    for (int i = i_lower; i < i_upper; i++) {
      for (int j = 0; j < V; j++) {
        if (dist[i*V + k] + dist[k*V + j] < dist[i*V + j]) {
          dist[i*V + j] = dist[i*V + k] + dist[k*V + j];
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

  dist = new int[V*V];
  for (int i=0;i<V;i++) {
    for (int j=0;j<V;j++) {
      int index = i*V+j;
      if (i == j) dist[index] = 0;
      else dist[index] = INF;
    }
  }

  for (int i = 0; i < E; i++) {
    int e[3];
    f.read((char*)e, sizeof e);
    dist[e[0] * V + e[1]] = e[2];
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
        fout.write((char*)(dist + i*V + j), sizeof(int));
    }
  }
  fout.close();
	pthread_exit(NULL);
  return 0;  
}  