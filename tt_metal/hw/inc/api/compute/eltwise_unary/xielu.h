// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_xielu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * xIELU (Expanded Integral of the Exponential Linear Unit).
 */
ALWI void xielu_tile(uint32_t idst, uint32_t alpha_p, uint32_t alpha_n) {
    MATH(SFPU_UNARY_TWO_PARAM_KERNEL_FN(calculate_xielu, RC, APPROX, idst, alpha_p, alpha_n));
}

ALWI void xielu_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(xielu, sfpu::xielu_init, APPROX)); }

}  // namespace ckernel
