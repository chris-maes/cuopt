#include <dual_simplex/barrier.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/user_problem.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <raft/core/handle.hpp>
#include <cublas_v2.h>
#include <cusparse.h>

#include <string>
#include <unistd.h>
#include <numeric>
#include <errno.h>
#include <iostream>

// CUDA test function
__global__ void cuda_test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

bool test_cuda() {
    try {
        // Test basic CUDA operations
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error getting device count: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        if (device_count == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        
        std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
        
        // Test memory allocation
        float* d_data;
        const int n = 1024;
        err = cudaMalloc(&d_data, n * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "CUDA error allocating memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Test kernel launch
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        cuda_test_kernel<<<grid, block>>>(d_data, n);
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in kernel execution: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_data);
            return false;
        }
        
        // Clean up
        cudaFree(d_data);
        
        std::cout << "CUDA test passed successfully!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in CUDA test: " << e.what() << std::endl;
        return false;
    }
}



ssize_t read_all(int fd, void* buf, size_t count) {
    size_t bytes_read = 0;
    while (bytes_read < count) {
        ssize_t result = read(fd, static_cast<char*>(buf) + bytes_read, count - bytes_read);
        if (result == -1) {
            printf("Error reading from file descriptor: %s\n", strerror(errno));
            if (errno == EINTR) continue;
            return -1;
        }
        if (result == 0) return bytes_read;
        bytes_read += result;
        printf("Read %ld/%ld bytes\n", bytes_read, count);
    }
    return bytes_read;
}

ssize_t write_all(int fd, const void* buf, size_t count) {
    size_t bytes_written = 0;
    while (bytes_written < count) {
        ssize_t result = write(fd, static_cast<const char*>(buf) + bytes_written, count - bytes_written);
        if (result == -1) {
            if (errno == EINTR) continue;
            return -1;
        }
        bytes_written += result;
    }
    return bytes_written;
}

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  cublasStatus_t cublas_status = cublasSetPointerMode(handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to set cuBLAS pointer mode");
  }
  
  cusparseStatus_t cusparse_status = cusparseSetPointerMode(handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE);
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to set cuSPARSE pointer mode");
  }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " fd_in fd_out data_size" << std::endl;
        return 1;
    }

    // Test CUDA functionality first
    std::cout << "Testing CUDA functionality..." << std::endl;
    if (!test_cuda()) {
        std::cerr << "CUDA test failed!" << std::endl;
        return 1;
    }

    int fd_in = atoi(argv[1]);
    int fd_out = atoi(argv[2]);
    size_t data_size = static_cast<size_t>(std::stoul(argv[3]));

    printf("Child: FD_IN: %d, FD_OUT: %d, DATA_SIZE: %ld\n", fd_in, fd_out, data_size);

    // Read data from the file descriptor
    char *buffer = new char[data_size];
    if (read_all(fd_in, buffer, data_size) != (ssize_t)(data_size)) {
        std::cout << "Error reading data from pipe." << std::endl;
        return 1;
    }

    namespace dual_simplex = cuopt::linear_programming::dual_simplex;

    raft::handle_t handle;
    
    // Initialize CUDA libraries
    init_handler(&handle);
    
    dual_simplex::user_problem_t<int, double> user_problem(&handle);
    user_problem.deserialize(buffer);

    dual_simplex::lp_solution_t<int, double> solution(user_problem.num_rows, user_problem.num_cols);
    dual_simplex::simplex_solver_settings_t<int, double> simplex_settings;
    dual_simplex::lp_status_t status = dual_simplex::solve_linear_program_with_barrier<int, double>(user_problem, simplex_settings, solution);

    char *result_buffer = new char[solution.bytes_required()];
    solution.serialize(result_buffer, static_cast<int>(status));
    // --- CHILD: RETURN DATA TO PARENT VIA STDOUT ---
    write_all(fd_out, result_buffer, solution.bytes_required());
    delete[] result_buffer;
    delete[] buffer;

    return 0;
}