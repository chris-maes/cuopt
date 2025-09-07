/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <dual_simplex/solution.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

#include <raft/core/handle.hpp>

#include <string>

namespace cuopt::linear_programming::dual_simplex {

enum class variable_type_t : int8_t {
  CONTINUOUS = 0,
  BINARY     = 1,
  INTEGER    = 2,
};

template <typename i_t, typename f_t>
struct user_problem_t {
  user_problem_t(raft::handle_t const* handle_ptr_)
    : handle_ptr(handle_ptr_), A(1, 1, 1), obj_constant(0.0), obj_scale(1.0)
  {
  }

  size_t bytes_required() const {
    // Minimial bytes required to serialize an LP only
    size_t bytes_required = 0;
    i_t nnz = A.col_start[num_cols];
    bytes_required += 3*sizeof(i_t);  // num_rows, num_cols
    bytes_required +=  num_cols * sizeof(f_t);  // objective
    bytes_required += (num_cols + 1) * sizeof(i_t);  // A.col_start
    bytes_required += nnz * sizeof(i_t);  // A.i
    bytes_required += nnz * sizeof(f_t);  // A.x
    bytes_required += num_rows * sizeof(f_t);  // rhs
    bytes_required += num_rows * sizeof(char);  // row_sense
    bytes_required += num_cols * sizeof(f_t);  // lower
    bytes_required += num_cols * sizeof(f_t);  // upper
    bytes_required += sizeof(f_t); // obj_constant
    bytes_required += sizeof(f_t); // obj_scale
    return bytes_required;
  }

  size_t serialize(char *buffer) const {
    size_t bytes_required = this->bytes_required();
    size_t bytes_written = 0;
    memcpy(buffer, &num_rows, sizeof(i_t));
    bytes_written += sizeof(i_t);
    memcpy(buffer + bytes_written, &num_cols, sizeof(i_t));
    bytes_written += sizeof(i_t);
    i_t nnz = A.col_start[num_cols];
    memcpy(buffer + bytes_written, &nnz, sizeof(i_t));
    bytes_written += sizeof(i_t);
    memcpy(buffer + bytes_written, objective.data(), num_cols * sizeof(f_t));
    bytes_written += num_cols * sizeof(f_t);
    memcpy(buffer + bytes_written, A.col_start.data(), (num_cols + 1) * sizeof(i_t));
    bytes_written += (num_cols + 1) * sizeof(i_t);
    memcpy(buffer + bytes_written, A.i.data(), nnz * sizeof(i_t));
    bytes_written += nnz * sizeof(i_t);
    memcpy(buffer + bytes_written, A.x.data(), nnz * sizeof(f_t));
    bytes_written += nnz * sizeof(f_t);
    memcpy(buffer + bytes_written, rhs.data(), num_rows * sizeof(f_t));
    bytes_written += num_rows * sizeof(f_t);
    memcpy(buffer + bytes_written, row_sense.data(), num_rows * sizeof(char));
    bytes_written += num_rows * sizeof(char);
    memcpy(buffer + bytes_written, lower.data(), num_cols * sizeof(f_t));
    bytes_written += num_cols * sizeof(f_t);
    memcpy(buffer + bytes_written, upper.data(), num_cols * sizeof(f_t));
    bytes_written += num_cols * sizeof(f_t);
    memcpy(buffer + bytes_written, &obj_constant, sizeof(f_t));
    bytes_written += sizeof(f_t);
    memcpy(buffer + bytes_written, &obj_scale, sizeof(f_t));
    bytes_written += sizeof(f_t);
    return bytes_written;
  }

  size_t deserialize(char *buffer) {
    size_t bytes_read = 0;
    memcpy(&num_rows, buffer + bytes_read, sizeof(i_t));
    bytes_read += sizeof(i_t);
    memcpy(&num_cols, buffer + bytes_read, sizeof(i_t));
    bytes_read += sizeof(i_t);
    i_t nnz;
    memcpy(&nnz, buffer + bytes_read, sizeof(i_t));
    bytes_read += sizeof(i_t);

    // Make sure all the data is the correct size 
    objective.resize(num_cols);
    A.m = num_rows;
    A.n = num_cols;
    A.col_start.resize(num_cols + 1);
    A.i.resize(nnz);
    A.x.resize(nnz);
    rhs.resize(num_rows);
    row_sense.resize(num_rows);
    lower.resize(num_cols);
    upper.resize(num_cols);

    memcpy(objective.data(), buffer + bytes_read, num_cols * sizeof(f_t));
    bytes_read += num_cols * sizeof(f_t);
    memcpy(A.col_start.data(), buffer + bytes_read, (num_cols + 1) * sizeof(i_t));
    bytes_read += (num_cols + 1) * sizeof(i_t);
    memcpy(A.i.data(), buffer + bytes_read, nnz * sizeof(i_t));
    bytes_read += nnz * sizeof(i_t);
    memcpy(A.x.data(), buffer + bytes_read, nnz * sizeof(f_t));
    bytes_read += nnz * sizeof(f_t);
    memcpy(rhs.data(), buffer + bytes_read, num_rows * sizeof(f_t));
    bytes_read += num_rows * sizeof(f_t);
    memcpy(row_sense.data(), buffer + bytes_read, num_rows * sizeof(char));
    bytes_read += num_rows * sizeof(char);
    memcpy(lower.data(), buffer + bytes_read, num_cols * sizeof(f_t));
    bytes_read += num_cols * sizeof(f_t);
    memcpy(upper.data(), buffer + bytes_read, num_cols * sizeof(f_t));
    bytes_read += num_cols * sizeof(f_t);
    memcpy(&obj_constant, buffer + bytes_read, sizeof(f_t));
    bytes_read += sizeof(f_t);
    memcpy(&obj_scale, buffer + bytes_read, sizeof(f_t));
    bytes_read += sizeof(f_t);
    return bytes_read;
  }


  raft::handle_t const* handle_ptr;
  i_t num_rows;
  i_t num_cols;
  std::vector<f_t> objective;
  csc_matrix_t<i_t, f_t> A;
  std::vector<f_t> rhs;
  std::vector<char> row_sense;
  std::vector<f_t> lower;
  std::vector<f_t> upper;
  std::vector<i_t> range_rows;
  std::vector<f_t> range_value;
  i_t num_range_rows;
  std::string problem_name;
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  f_t obj_constant;
  f_t obj_scale;  // 1.0 for min, -1.0 for max
  std::vector<variable_type_t> var_types;
};

}  // namespace cuopt::linear_programming::dual_simplex
