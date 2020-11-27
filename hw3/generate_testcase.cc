#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

using namespace std;

#define MAX 1073741823

int main(int argc, char** argv) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    int n_threads = CPU_COUNT(&cpu_set);

    string out_filename;
    out_filename = argv[1];
    int n_vertex = 6000;
    int n_edge = 35994000;
    int currentWeight = 1000;
    int maximumWeight = 1000;

    fstream out_file;
    out_file.open(out_filename, ios::out | ios::binary);

    if(!out_file) {
        cout << "Open int_file fail." << endl;
    }

    out_file.write((char*)(&n_vertex), sizeof(int));
    out_file.write((char*)(&n_edge), sizeof(int));
    cout << n_vertex << endl;
    cout << n_edge << endl;

    int src;
    int dst;
    int weight;
    int offset = n_edge/maximumWeight;
    for (int src=0; src<n_vertex; src++) {
      int currentWeight = 1001;
      for (int dst=0; dst<n_vertex; dst++) {
        if (src == dst) continue;
        if (dst%offset == 0) currentWeight--;
        if (currentWeight < 0) currentWeight = 0;
        if (currentWeight >= 1001) currentWeight = 1000;
        // int rowJump = src/offset;
        // int colJump = dst/offset;
        // int decreaseNum = min(rowJump, colJump);
        // int newWeight = currentWeight - decreaseNum;
        // if (newWeight < 0) newWeight = 0;
        out_file.write((char*)(&src), sizeof(int));
        out_file.write((char*)(&dst), sizeof(int));
        out_file.write((char*)(&currentWeight), sizeof(int));
        // cout << src << endl;
        // cout << dst << endl;
        // cout << weight << endl;
      }
    }
    out_file.close();   

    return 0;
}