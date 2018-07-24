// http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf

#include <cstdio>
#include <iostream>

#define N 20
#define BLOCK_SIZE 10
#define RADIUS 2


__global__ void stencil_1d(int *in, int *out) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  // printf("blockIdx.x = %3d, threadIdx.x = %3d, blockDim.x=%3d\n", blockIdx.x, threadIdx.x, blockDim.x);
  printf("threadIdx.x = %3d, gindex = %3d, lindex = %3d\n",
         threadIdx.x, gindex, lindex);

  // Read Input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    printf("threadIdx.x = %3d, lindex - RADIUS = %3d, gindex - RADIUS = %3d, lindex + BLOCK_SIZE = %3d, gindex + BLOCK_SIZE = %3d,\nin[gindex - RADIUS] = %3d, in[gindex + BLOCK_SIZE] = %3d\n",
           threadIdx.x, lindex - RADIUS, gindex - RADIUS, lindex + BLOCK_SIZE, gindex + BLOCK_SIZE,
           in[gindex - RADIUS], in[gindex + BLOCK_SIZE]);
  }

  // Synchronize (ensure all the data is available)
  __syncthreads();

  // Apply the stencil
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++) {
    result += temp[lindex + offset];
  }

  // Store the result
  out[gindex] = result;
}

void random_ints(int *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = rand() % 10;
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
  int *a, *b; // host memory
  int *d_a, *d_b; // device memory
  int size = N * sizeof(int);

  // Allocate space for host memory of a, b
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size);
  // Allocate space for device memory of a, b
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

  // Launch stencil_1d() kernel on GPU with N threads
  stencil_1d<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b);

  // Copy result back to host
  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

  std::cout << "a:" << std::endl;
  print_vector(a, N);
  std::cout << "b:" << std::endl;
  print_vector(b, N);

  // Cleanup
  free(a); free(b);
  cudaFree(d_a); cudaFree(d_b);

  return 0;
}
