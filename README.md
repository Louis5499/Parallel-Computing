# Parallel Computing
National Tsing Hua University - CS 542200
Course's Instructor: Jerry Chou

This repo includes 4(+1) homeworks, which all cover on parallel computing techniques, such as MPI, OpenMP, Pthread, and CUDA.

The specific details of each homeworks (Homework Spec, My Report), please refer to each homework directory and find the PDF file prefixed with `PP_2020` & `hw`.

Homework List:
1. HW1: [MPI] Odd-Even Sort
  - If you wants to enhance the performance, please rewrite the logic by `C++` and call the `C++ native libraries`(STL, ...), you will get greater performance than `C`.
  - There're some optimization methods you could try, which is mentioned in the `HW1_Optimization.pdf` file.

2. HW2: [MPI+OpenMP+Vectorization] Mandelbrot Set
  - My code is ranking top 3.
  - Vectorization is Intel-based speedup method.

3. HW3: [Pthread] All-Pairs Shortest Path
  - There're two way get more efficient code than my code:
    1. Implement `Blocked Floyd-Warshall algorithm`, and use `Tenary Expression` to implement logics. (You can check `./hw4-1/seq.cc` to see how to implement `Blocked Floyd-Warshall algorithm` on CPU.)
    2. Implement `Floyd-Warshall algorithm`, and use `Vectorization` to implement logics.

4. HW4: [Cuda] All-Pairs Shortest Path (One GPU or Multiple GPUs)
  - My code is ranking top 3.

There are also 5 labs and how I implemented for your reference.

