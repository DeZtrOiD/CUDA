#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#define MAX_CLASSES (32)
#define MAX_PATH_SIZE (4096)
#define THREADS_PER_BLOCK (256)
#define BLOCKS_PER_GRID (5)
#define MAX_DIMENSION_SIZE (400000000)
#define MAX_NP (524288)

#define PRINT_ERR(condition, message) \
do { \
    if (condition) { \
        fprintf(stderr, "%s at %s:%d\n", message, __FILE__, __LINE__); \
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


__constant__ float d_avg_norm[MAX_CLASSES][3];
__constant__ int d_num_classes;

__global__ void kernel(uchar4* d_input, uchar4* output, uint32_t width, uint32_t height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (size_t pos = idx; pos < (size_t)(width * height); pos += stride) {
        uchar4 pixel = d_input[pos];
        
        float max_score = -FLT_MAX;
        int best_class = 0;
        
        for (int j = 0; j < d_num_classes; j++) {
            // p^T * (avg_j / |avg_j|)
            float score = ((float)pixel.x) * d_avg_norm[j][0]
                + ((float)pixel.y) * d_avg_norm[j][1]
                + ((float)pixel.z) * d_avg_norm[j][2];

            if (score > max_score) {
                max_score = score;
                best_class = j;
            }
        }
        
        output[pos] = make_uchar4(pixel.x, pixel.y, pixel.z, (unsigned char)best_class);
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
    PRINT_ERR((width <= 0) || (height <= 0) || 
        ((uint64_t)width * height > MAX_DIMENSION_SIZE), "Incorrect dimensions\n");

    size_t num_pixels = (size_t)width * height;
    size_t data_size = num_pixels * sizeof(uchar4);

    uchar4* h_input = (uchar4*)malloc(data_size);
    PRINT_ERR(h_input == NULL, "Failed to allocate host input memory\n");
    uchar4* h_output = (uchar4*)malloc(data_size);
    PRINT_ERR(h_output == NULL, "Failed to allocate host output memory\n");

    PRINT_ERR(fread(h_input, sizeof(uchar4), num_pixels, in_file) != num_pixels, "Error reading pixel data\n");
    fclose(in_file);

    int num_classes;
    PRINT_ERR(scanf("%d", &num_classes) == EOF, "Error reading number of classes\n");
    PRINT_ERR(num_classes <= 0 || num_classes > MAX_CLASSES, "Invalid number of classes\n");

    float h_avg[MAX_CLASSES][3] = {0};

    int* class_pixel_counts = (int*)calloc(num_classes, sizeof(int));
    PRINT_ERR(class_pixel_counts == NULL, "Failed to allocate class counts memory");

    int** class_coords = (int**)malloc(num_classes * sizeof(int*));
    PRINT_ERR(class_coords == NULL, "Failed to allocate coords memory");
    
    for (int j = 0; j < num_classes; j++) {
        int np_j;
        PRINT_ERR(scanf("%d", &np_j) == EOF, "Error reading number of pixels for class\n");
        PRINT_ERR(np_j > MAX_NP, "Incorrect np_j\n");

        class_pixel_counts[j] = np_j;

        class_coords[j] = (int*)malloc(np_j * 2 * sizeof(int));
        PRINT_ERR(class_coords[j] == NULL, "Failed to allocate coords for class\n");

        for (int i = 0; i < np_j * 2; i++)
            PRINT_ERR(scanf("%d", &class_coords[j][i]) == EOF, "Error reading coordinate\n");
        // avg_j
        for (int i = 0; i < np_j; i++) {
            int x = class_coords[j][i * 2];
            int y = class_coords[j][i * 2 + 1];
            int idx = y * width + x;
            uchar4 pixel = h_input[idx];
            h_avg[j][0] += pixel.x;
            h_avg[j][1] += pixel.y;
            h_avg[j][2] += pixel.z;
        }
        h_avg[j][0] /= np_j;
        h_avg[j][1] /= np_j;
        h_avg[j][2] /= np_j;
    }
    for (int j = 0; j < num_classes; j++) {
        free(class_coords[j]);
    }
    free(class_coords);
    free(class_pixel_counts);
    // (avg_j / |avg_j|)
    for (int j = 0; j < num_classes; j++) {
        float norm = sqrtf(
            h_avg[j][0] * h_avg[j][0]
            + h_avg[j][1] * h_avg[j][1]
            + h_avg[j][2] * h_avg[j][2]
        );
        if (norm > 1e-10) {
            h_avg[j][0] = h_avg[j][0] / norm;
            h_avg[j][1] = h_avg[j][1] / norm;
            h_avg[j][2] = h_avg[j][2] / norm;
        } else {
            h_avg[j][0] = h_avg[j][1] = h_avg[j][2] = 0.0f;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_avg_norm, h_avg, sizeof(h_avg)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_classes, &num_classes, sizeof(int)));

    uchar4* d_input;
    cudaMalloc(&d_input, data_size);
    cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);
    free(h_input);

    uchar4* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_size));
    
    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_input, d_output, width, height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost));
    cudaFree(d_output);

    PRINT_ERR(fwrite(&width, sizeof(int), 1, out_file) != 1, "Error writing width\n");
    PRINT_ERR(fwrite(&height, sizeof(int), 1, out_file) != 1, "Error writing height\n");
    PRINT_ERR(fwrite(h_output, sizeof(uchar4), num_pixels, out_file) != num_pixels, "Error writing pixel data\n");
    
    fclose(out_file);
    free(h_output);
    
    return 0;
}
