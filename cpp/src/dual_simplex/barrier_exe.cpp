#include <dual_simplex/barrier.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/user_problem.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <raft/core/handle.hpp>

#include <string>
#include <unistd.h>
#include <numeric>
#include <errno.h>



ssize_t read_all(int fd, void* buf, size_t count) {
    size_t bytes_read = 0;
    while (bytes_read < count) {
        ssize_t result = read(fd, static_cast<char*>(buf) + bytes_read, count - bytes_read);
        if (result == -1) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (result == 0) return bytes_read;
        bytes_read += result;
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


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " fd_in fd_out data_size" << std::endl;
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
    printf("Child: Deserializing user problem\n");
    dual_simplex::user_problem_t<int, double> user_problem(&handle);
    user_problem.deserialize(buffer);
    printf("Child: User problem deserialized\n");
    
    dual_simplex::lp_solution_t<int, double> solution(user_problem.num_rows, user_problem.num_cols);
    dual_simplex::simplex_solver_settings_t<int, double> simplex_settings;
    simplex_settings.log.log = true;
    simplex_settings.log.log_to_console = true;
    simplex_settings.log.enable_log_to_file();
    simplex_settings.log.set_log_file("barrier.log");
    printf("Child: Solving linear program with barrier\n");
    dual_simplex::lp_status_t status = dual_simplex::solve_linear_program_with_barrier<int, double>(user_problem, simplex_settings, solution);
    printf("Child: Linear program with barrier solved\n");
    printf("Child: Serializing solution\n");
    char *result_buffer = new char[solution.bytes_required()];
    solution.serialize(result_buffer, static_cast<int>(status));
    printf("Child: Solution serialized\n");
    // Write data to the file descriptor
    printf("Child: Writing solution to file descriptor\n");
    write_all(fd_out, result_buffer, solution.bytes_required());
    printf("Child: Solution written to file descriptor\n");
    delete[] result_buffer;
    delete[] buffer;

    return 0;
}