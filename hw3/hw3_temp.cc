// C++ code
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <chrono>
  
#include <limits.h> 
#include <stdio.h> 
#include <iostream>

#define ARRAY_SIZE 600
  
int V = 0;

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

int minDistance(int dist[], bool sptSet[])  { 
  int min = INT_MAX, min_index;

  for (int v = 0; v < V; v++) {
    if (sptSet[v] == false && dist[v] <= min) {
        min = dist[v], min_index = v;
    }
  }

  return min_index; 
}

int main(int argc, char** argv){
    std::ifstream f(argv[1]);
    if (not f) {
        throw reprintf("failed to open file: %s", argv[1]);
    }
    f.seekg(0, std::ios_base::end);
    f.seekg(0, std::ios_base::beg);
    int V;
    int E;
    f.read((char*)&V, sizeof V);
    printf("V = %d\n", V);
    f.read((char*)&E, sizeof E);
    printf("E = %d\n", E);

    int graph[ARRAY_SIZE][ARRAY_SIZE] = { 0 };
    for (int i = 0; i < E; i++) {
      int e[3];
      f.read((char*)e, sizeof e);
      graph[e[0]][e[1]] = e[2];
      std::cout << graph[e[0]][e[1]] << "\n";
      printf("%d   %d:  %d\n", e[0], e[1], graph[e[0]][e[1]]);
    }

    printf("%d", graph[0][0]);

    // int dist[V];
    // bool sptSet[V];

    // for (int i = 0; i < V; i++) {
    //   dist[i] = INT_MAX, sptSet[i] = false;
    // }

    // int src = 0;
    // dist[src] = 0; 

    // for (int count = 0; count < V - 1; count++) { 
    //   int u = minDistance(dist, sptSet);
    //   sptSet[u] = true;
    //   for (int v = 0; v < V; v++) {
    //     if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v]) {
    //       dist[v] = dist[u] + graph[u][v];
    //     }
    //   }
    // }

    // printf("Vertex   Distance from Source\n"); 
    // for (int i = 0; i < V; i++) {
    //   printf("%d tt %d\n", i, dist[i]);
    // }
    return 0;
}