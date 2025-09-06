#include <dual_simplex/barrier.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/user_problem.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <string>
#include <unistd.h>
#include <numeric>



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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " data_size" << std::endl;
        return 1;
    }
    size_t data_size = static_cast<size_t>(std::stoul(argv[1]));

    // --- CHILD: RECEIVE DATA FROM PARENT VIA STDIN ---
    char *buffer = new char[data_size];
    if (read_all(STDIN_FILENO, buffer, data_size) != (ssize_t)(data_size)) {
        std::cerr << "Error reading data from pipe." << std::endl;
        return 1;
    }

    raft::handle_t handle;
    dual_simplex::user_problem_t<int, double> user_problem(handle);
    user_problem.deserialize(buffer);

    dual_simplex::lp_solution_t<int, double> solution(user_problem.num_rows, user_problem.num_cols);
    dual_simplex::simplex_solver_settings_t<int, double> simplex_settings;
    dual_simplex::lp_status_t status = dual_simplex::solve_linear_program_with_barrier(user_problem, simplex_settings, solution);

    char *result_buffer = new char[solution.bytes_required()];
    solution.serialize(result_buffer, static_cast<int>(status));
    // --- CHILD: RETURN DATA TO PARENT VIA STDOUT ---
    write_all(STDOUT_FILENO, result_buffer, solution.bytes_required());
    delete[] result_buffer;
    delete[] buffer;

    return 0;
}