#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK (1024)
#define BLOCKS_PER_GRID (1024)


#define PRINT_ERR(condition, message) \
if (condition) { \
	fprintf(stderr, message); \
	return 0; \
}

#define CUDA_CHECK(call) \
do { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		return 0; \
	} \
} while(0)


__global__ void kernel(double *arr1, double *arr2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    
    while (idx < n) {
        arr2[idx] = min(arr1[idx], arr2[idx]);
        idx += offset;
    }
}


int main() {
    int n = 0;

	PRINT_ERR(scanf("%d", &n) == EOF, "Error reading n\n");

	PRINT_ERR(n <= 0 || n >= (1 << 25), "Error: n must satisfy 0 < n < 2^25\n");
    
    double *arr1 = (double *)malloc(sizeof(double) * n);
    double *arr2 = (double *)malloc(sizeof(double) * n);
    
	PRINT_ERR(!arr1 || !arr2, "Host memory allocation failed\n");

    for (int i = 0; i < n; i++)
		PRINT_ERR(scanf("%lf", &arr1[i]) == EOF, "Error reading array 1\n");

    for (int i = 0; i < n; i++)
		PRINT_ERR(scanf("%lf", &arr2[i]) == EOF, "Error reading array 2\n");


    double *dev_arr1, *dev_arr2;
	CUDA_CHECK( cudaMalloc(&dev_arr1, sizeof(double) * n) );
	CUDA_CHECK( cudaMalloc(&dev_arr2, sizeof(double) * n) );

	CUDA_CHECK( cudaMalloc(&dev_arr2,  sizeof(double) * n) );
    
	CUDA_CHECK( cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice) );
    
    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_arr1, dev_arr2, n);
    
	CUDA_CHECK( cudaGetLastError() );
    
    CUDA_CHECK( cudaMemcpy(arr2, dev_arr2, sizeof(double) * n, cudaMemcpyDeviceToHost) );

    for (int i = 0; i < n - 1; i++) printf("%.10e ", arr2[i]);
	printf("%.10e", arr2[n-1]);

    printf("\n");

    CUDA_CHECK( cudaFree(dev_arr1) );
    CUDA_CHECK( cudaFree(dev_arr2) );

    free(arr1);
    free(arr2);
    
    return 0;
}
