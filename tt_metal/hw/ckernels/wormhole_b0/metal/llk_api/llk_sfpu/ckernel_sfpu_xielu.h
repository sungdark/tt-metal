// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat _sfpu_neg_exp_f32_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    // Step 1: Compute k = round(x / ln(2))
    // z = x / ln(2) = x * (1/ln(2))
    constexpr float INV_LN2 = 1.4426950408889634f;  // 1/ln(2)
    sfpi::vFloat z = val * INV_LN2;

    // Underflow bound
    sfpi::vFloat underflow_bound = UNDERFLOW_THRESHOLD;
    sfpi::vec_min_max(underflow_bound, z);

    // Round z to nearest integer using round-to-nearest-even
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

    // Step 2: Cody-Waite range reduction
    // Compute r = x - k*ln(2) in extended precision
    // r = x - k*LN2_HI - k*LN2_LO
    // This provides better accuracy than simple r = x - k*ln(2)
    // Cody-Waite constants: ln(2) split into high and low parts for extended precision.
    // LN2_HI is chosen so that k*LN2_HI can be computed exactly for integer k in the valid range.
    // LN2_LO contains the remainder: LN2_HI + LN2_LO ≈ -ln(2)

    // We want to do:
    // 1) r_hi = val - k * LN2_HI
    // 2) r = r_hi - k * LN2_LO
    // Since SFPMAD on Wormhole can only do VD = VA * VB + VC,
    // this expression would require additional instructions,
    // To avoid this, we transform the expressions to:
    // 1) r_hi = val + k * (-LN2_HI)
    // 2) r = r_hi + k * (-LN2_LO)
    // Where LN2_HI and LN2_LO are negated.
    // This way, compiler can more easily optimize this expression to a single SFPMAD instruction.
    constexpr float LN2_HI = -0.6931152343750000f;  // High bits of ln(2)
    constexpr float LN2_LO = -3.19461832987e-05f;   // Low bits of ln(2)

    // First subtract k * LN2_HI
    sfpi::vFloat r_hi = k * LN2_HI + val;

    // Then subtract k * LN2_LO
    sfpi::vFloat r = k * LN2_LO + r_hi;

    // Step 3: Polynomial approximation for exp(r) using Taylor series
    // exp(r) ~= 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6! + r⁷/7!
    // Use 7th order polynomial (Taylor series coefficients) for < 1 ULP accuracy
    // Coefficients in ascending order of powers: c0, c1, c2, c3, c4, c5, c6, c7
    sfpi::vFloat p = PolynomialEvaluator::eval(
        r,
        sfpi::vConst1,  // c0 = 1
        sfpi::vConst1,  // c1 = 1
        0.5f,           // c2 = 1/2!
        1.0f / 6.0f,    // c3 = 1/3!
        1.0f / 24.0f,   // c4 = 1/4!
        1.0f / 120.0f,  // c5 = 1/5!
        1.0f / 720.0f,  // c6 = 1/6!
        1.0f / 5040.0f  // c7 = 1/7!
    );

    // Step 4: Scale by 2^k using exponent manipulation
    // ldexp(p, k_int) = p * 2^k
    // We do this by adding k_int to the exponent of p
    // Get the current exponent of p (without bias)
    sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
    // Add k_int to get the new exponent
    sfpi::vInt new_exp = p_exp + k_int;

    // Set the new exponent
    result = sfpi::setexp(p, new_exp);

    return result;
}

/*
 * This function implements the xIELU (Expanded Integral of ELU) activation function,
 * a trainable piecewise function with learnable alpha_p and alpha_n parameters.
 * based on "Deriving Activation Functions Using Integration"
 *
 * @see "Deriving Activation Functions Using Integration" (https://arxiv.org/abs/2411.13010)
 *
 * @param x : The input tensor
 * @param alpha_p : The positive alpha parameter
 * @param alpha_n : The negative alpha parameter
 *
 * Positive input
 * if x > 0 :  alpha_p * x * x + beta * x
 * Negative input
 * if x < 0 : alpha_n * expm1(minimum(x, eps)) - alpha_n * x + beta * x
 *        --> alpha_n * (expm1(minimum(x, eps)) - x) + beta * x
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_xielu(const uint32_t param0, const uint32_t param1) {
    sfpi::vFloat alpha_p = Converter::as_float(param0);
    sfpi::vFloat alpha_n = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat beta_mul_x = sfpi::vConstFloatPrgm0 * x;
        v_if(x > sfpi::vConst0) {
            sfpi::vFloat term_1 = alpha_p * x * x;
            sfpi::dst_reg[0] = term_1 + beta_mul_x;
        }
        v_elseif(x >= sfpi::vConstFloatPrgm1) {
            sfpi::vFloat exp_term = sfpi::vConstFloatPrgm2 - x;
            sfpi::vFloat term_1 = alpha_n * exp_term;
            sfpi::dst_reg[0] = term_1 + beta_mul_x;
        }
        v_else {
            sfpi::vFloat exp_term = _sfpu_neg_exp_f32_(x) - sfpi::vConst1 - x;
            sfpi::vFloat term_1 = alpha_n * exp_term;
            sfpi::dst_reg[0] = term_1 + beta_mul_x;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void xielu_init() {
    sfpi::vConstFloatPrgm0 = 0.5f;                  // beta value
    sfpi::vConstFloatPrgm1 = -1e-6f;                // eps value
    sfpi::vConstFloatPrgm2 = -0.0000009999995427f;  // expm1(eps)
}

}  // namespace sfpu
}  // namespace ckernel
