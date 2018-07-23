// http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <iostream>


__global__ void add(int *a, int *b, int *c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
  printf("threadIdx.x = %3d\n", threadIdx.x);
}

void random_ints(int *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = rand() % 20;
  }
}

void print_vector(int *a, int n) {
  printf("(");
  for (int i = 0; i < n; i++) {
    printf("%2d, ", a[i]);
  }
  printf(")\n");
}

int main(void) {
  int N = 10;
  // int N = 512;

  int *a, *b, *c; // host memory
  int *d_a, *d_b, *d_c; // device memory
  int size = N * sizeof(int);

  // Allocate space for host memory of a, b, c
  srand(time(NULL));
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size);
  // Allocate space for device memory of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU with N threads
  add<<<1,N>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  std::cout << "a:" << std::endl;
  print_vector(a, N);
  std::cout << "b:" << std::endl;
  print_vector(b, N);
  std::cout << "c:" << std::endl;
  print_vector(c, N);

  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}
