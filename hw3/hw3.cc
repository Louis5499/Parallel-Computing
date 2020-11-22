// C++ Program for Floyd Warshall Algorithm  
#include <bits/stdc++.h> 
#include <omp.h>

using namespace std;

typedef pair<int, int> iPair;
#define INF 99999

void addEdge(vector <pair<int, int> > adj[], int u, int v, int wt) {
    adj[u].push_back(make_pair(v, wt)); 
    adj[v].push_back(make_pair(u, wt)); 
}

int *finalDist;

void shortestPath(vector<pair<int,int> > adj[], int V, int src) { 
    priority_queue< iPair, vector <iPair> , greater<iPair> > pq; 

    vector<int> dist(V, INF); 

    pq.push(make_pair(0, src)); 
    dist[src] = 0; 

    while (!pq.empty()) { 
        int u = pq.top().second; 
        pq.pop(); 
  
        // Get all adjacent of u.  
        for (auto x : adj[u]) { 
            int v = x.first; 
            int weight = x.second; 
  
            // If there is shorted path to v through u. 
            if (dist[v] > dist[u] + weight) 
            { 
                // Updating distance of v 
                dist[v] = dist[u] + weight; 
                pq.push(make_pair(dist[v], v)); 
            } 
        } 
    } 
  
    // Print shortest distances stored in dist[] 
    // printf("Vertex Distance from Source\n"); 
    for (int i = 0; i < V; ++i) 
      finalDist[src*V+i] = dist[i];
        // printf("%d \t\t %d\n", i, dist[i]); 
}

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
  
// Driver code  
int main(int argc, char** argv) {
  /* detect how many CPUs are available */
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
  int ncpus = CPU_COUNT(&cpu_set);
  printf("%d cpus available\n", ncpus);

  std::ifstream f(argv[1]);
  if (not f) {
      throw reprintf("failed to open file: %s", argv[1]);
  }
  f.seekg(0, std::ios_base::end);
  f.seekg(0, std::ios_base::beg);
  int V;
  int E;
  f.read((char*)&V, sizeof V);
  // printf("V = %d\n", V);
  f.read((char*)&E, sizeof E);
  // printf("E = %d\n", E);
  
  finalDist = new int[V*V];

  vector<iPair > adj[V]; 
  for (int i = 0; i < E; i++) {
    int e[3];
    f.read((char*)e, sizeof e);
    // dist[e[0] * V + e[1]] = e[2];
    addEdge(adj, e[0], e[1], e[2]); 
  }
  
  #pragma omp parallel num_threads(ncpus)
  {
    #pragma omp for schedule(dynamic)
    for (int i=0; i<V; i++) {
      shortestPath(adj, V, i);
    }
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
        fout.write((char*)(finalDist + i*V + j), sizeof(int));
    }
  }
  fout.close();
  return 0;  
}  