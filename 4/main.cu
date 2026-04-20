#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iomanip> 

const dim3 THREADS_PER_BLOCK(32, 32);
const dim3 BLOCKS_PER_GRID(32, 32);
constexpr int BLOCKS_1D(256);
constexpr int THREADS_1D(256);
constexpr int MAX_N = 10000;
constexpr double EPSILON = 1e-7;

#define PRINT_ERR(condition, message) \
do { \
    if (condition) { \
        std::cerr << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(0); \
    } \
} while(0);

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << "\n" \
                  << cudaGetErrorString(err) << std::endl; \
        std::exit(0); \
    } \
} while(0);

struct abs_less {
    __host__ __device__
    bool operator()(double a, double b) const noexcept {
        return std::abs(a) < std::abs(b);
    }
};

__global__ void swap_rows_kernel(double* d_matrix, int n, int pivot_col, int max_row) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offsetx = blockDim.x * gridDim.x;

    for (int col = pivot_col + idx; col < n; col += offsetx) {
        const int idx1 = col * n + pivot_col;
        const int idx2 = col * n + max_row;

        double tmp = d_matrix[idx1];
        d_matrix[idx1] = d_matrix[idx2];
        d_matrix[idx2] = tmp;
    }
}

__global__ void gaussian_elimination_kernel(
        double* d_matrix, int n, int pivot_col, const double* d_factors
    ) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    for (int row = pivot_col + 1 + idx; row < n; row += offsetx) {
        for (int col = pivot_col + 1 + idy; col < n; col += offsety) {
            const double factor = d_factors[row];

            const int row_idx = col * n + row;
            const int pivot_row_idx = col * n + pivot_col;

            d_matrix[row_idx] -= factor * d_matrix[pivot_row_idx];
        }
    }
}

__global__ void compute_factors_kernel(
        double* d_factors, const double* d_matrix, int n, int pivot_col
    ) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offsetx = blockDim.x * gridDim.x;
    
    for (int row = pivot_col + 1 + idx; row < n; row += offsetx) {
        const int pivot_idx = pivot_col * n + pivot_col;
        const int elem_idx = pivot_col * n + row;

        d_factors[row] = d_matrix[elem_idx] / d_matrix[pivot_idx];
    }
}

int main() {
    int n{};
    PRINT_ERR(!(std::cin >> n), "Error reading n");
    PRINT_ERR(n <= 0 || n > MAX_N, "Invalid matrix size");
    std::vector<double> h_matrix(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val{};
            PRINT_ERR(!(std::cin >> val), "Error reading matrix element");
            h_matrix[static_cast<size_t>(j) * n + i] = val;
        }
    }
    
    double* d_matrix{};
    CUDA_CHECK(cudaMalloc(&d_matrix, static_cast<size_t>(n) * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(
            d_matrix, h_matrix.data(),
            static_cast<size_t>(n) * n * sizeof(double), cudaMemcpyHostToDevice
        )
    );

    double* d_factors{};
    CUDA_CHECK(cudaMalloc(&d_factors, static_cast<size_t>(n) * sizeof(double)));
    
    long double determinant{1.0};
    int swap_count{};
    
    for (int pivot_col = 0; pivot_col < n; ++pivot_col) {
        thrust::device_ptr<double> col_start(d_matrix + static_cast<size_t>(pivot_col) * n + pivot_col);
        thrust::device_ptr<double> col_end(d_matrix + static_cast<size_t>(pivot_col) * n + n);
        
        auto max_iter = thrust::max_element(col_start, col_end, abs_less{});
        const int max_row = pivot_col + (max_iter - col_start); 
        
        double pivot_val{};
        const int pivot_idx = pivot_col * n + max_row;
        CUDA_CHECK(cudaMemcpy(&pivot_val, d_matrix + pivot_idx, sizeof(double), cudaMemcpyDeviceToHost));
        
        if (std::abs(pivot_val) < EPSILON) {
            determinant = 0.0;
            swap_count = 0;
            break;
        }

        determinant *= pivot_val;
        
        if (max_row != pivot_col) {
            swap_rows_kernel<<<BLOCKS_1D, THREADS_1D>>>(d_matrix, n, pivot_col, max_row);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            ++swap_count;
        }
        
        if (pivot_col < n - 1) {
            compute_factors_kernel<<<BLOCKS_1D, THREADS_1D>>>(
                d_factors, d_matrix, n, pivot_col
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            gaussian_elimination_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
                d_matrix, n, pivot_col, d_factors
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    if (swap_count & 1) determinant = -determinant;
    
    std::cout << std::scientific << std::setprecision(10) << determinant << std::endl;
    
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_factors));
    return 0;
}
