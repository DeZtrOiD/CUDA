// 7
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>


#define THREADS_PER_BLOCK (dim3(32, 32))
#define BLOCKS_PER_GRID (dim3(16, 16))
#define MAX_PATH_SIZE (4096)
#define MAX_DIMENSION_SIZE (100000000)

#define PRINT_ERR(condition, message) \
do { \
	if (condition) { \
		fprintf(stderr, "%s at %s:%d", message, __FILE__, __LINE__); \
		exit(0); \
	} \
} while(0);

#define CUDA_CHECK(call) \
do { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(0); \
	} \
} while(0);


__global__ void kernel(cudaTextureObject_t tex, uchar4 *output, uint32_t width, uint32_t height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int x, y;
    float neighborhood[3][3];
    for(y = idy; y < (int)height; y += offsety) {
        for(x = idx; x < (int)width; x += offsetx) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uchar4 pixel = tex2D<uchar4>(tex, x + dx, y + dy);
                    neighborhood[dy + 1][dx + 1] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
                }
            }

            float Gx = neighborhood[0][2] + 2.0f * neighborhood[1][2] + neighborhood[2][2]
                - neighborhood[0][0] - 2.0f * neighborhood[1][0] - neighborhood[2][0];

            float Gy = neighborhood[2][0] + 2.0f * neighborhood[2][1] + neighborhood[2][2]
                - neighborhood[0][0] - 2.0f * neighborhood[0][1] - neighborhood[0][2];

            float magnitude = sqrtf(Gx * Gx + Gy * Gy);
            if (magnitude < 0.0f) magnitude = 0.0f;
            if (magnitude > 255.0f) magnitude = 255.0f;

            output[y * width + x] = make_uchar4(
				(unsigned char)(magnitude),
				(unsigned char)(magnitude),
				(unsigned char)(magnitude),
				0
			);
        }
    }
}

int main() {
	char* path = (char*)calloc(sizeof(char), MAX_PATH_SIZE);
	PRINT_ERR(path == NULL, "Host memory allocation failed\n");

	PRINT_ERR(scanf("%4095s", path) == EOF, "Error reading input path\n");
	FILE* in_file = fopen(path, "rb");
	PRINT_ERR(in_file == NULL, "Cannot open input file\n");

	PRINT_ERR(scanf("%4095s", path) == EOF, "Error reading output path\n");
	FILE* out_file = fopen(path, "wb");
	PRINT_ERR(out_file == NULL, "Cannot open output file\n");
	free(path);

    int width, height;
	PRINT_ERR(fread(&width, sizeof(int), 1, in_file) != 1, "Error reading width\n");
	PRINT_ERR(fread(&height, sizeof(int), 1, in_file) != 1, "Error reading height\n");
	PRINT_ERR((width <= 0) || (height <= 0)
		|| ((uint64_t)width * height > MAX_DIMENSION_SIZE), "Incorrect dimensions\n");

	size_t num_pixels = (size_t)width * height;
    size_t data_size = num_pixels * sizeof(uchar4);

	uchar4* h_input = (uchar4*)malloc(data_size);
    PRINT_ERR(h_input == NULL, "Failed to allocate host input memory");
    uchar4* h_output = (uchar4*)malloc(data_size);
    PRINT_ERR(h_output == NULL, "Failed to allocate host output memory");

	PRINT_ERR(fread(h_input, sizeof(uchar4), num_pixels, in_file) != num_pixels, "Error reading pixel data\n");
	fclose(in_file);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cudaArray* cu_array;
    CUDA_CHECK(cudaMallocArray(&cu_array, &channelDesc, width, height));

    CUDA_CHECK(cudaMemcpy2DToArray(cu_array, 0, 0,
			h_input, width * sizeof(uchar4),
			width * sizeof(uchar4), height,
			cudaMemcpyHostToDevice
		)
	);
	free(h_input);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

	uchar4* d_output;
	cudaTextureObject_t tex_obj;
    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMalloc(&d_output, data_size));
    
    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(tex_obj, d_output, width, height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
	cudaDestroyTextureObject(tex_obj);
	cudaFreeArray(cu_array);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost));
	cudaFree(d_output);

	PRINT_ERR(fwrite(&width, sizeof(int), 1, out_file) != 1, "Error writing width\n");
	PRINT_ERR(fwrite(&height, sizeof(int), 1, out_file) != 1, "Error writing height\n");
	PRINT_ERR(fwrite(h_output, sizeof(uchar4), num_pixels, out_file) != num_pixels, "Error writing pixel data\n");
	
	fclose(out_file);
	free(h_output);
    return 0;
}
