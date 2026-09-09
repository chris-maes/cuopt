/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuts/cuts.hpp>
#include <dual_simplex/user_problem.hpp>

#include <utility>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
std::pair<bool, probing_implied_bound_t<i_t, f_t>> compute_fresh_probing_implied_bounds(
  const simplex::user_problem_t<i_t, f_t>& problem,
  f_t absolute_tolerance,
  f_t integrality_tolerance,
  f_t time_limit);

}  // namespace cuopt::mathematical_optimization::mip
