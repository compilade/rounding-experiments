#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define GGML_ASSERT(x) assert(x)


// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

typedef uint16_t ggml_fp16_t;

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

// ---- From ggml-quants.c ----


#define GROUP_MAX_EPS 1e-15f

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, int rmse_type,
        const float * restrict qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static float make_q3_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return sumlx / suml2;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

static float make_qkx1_quants(int n, int nmax, const float * restrict x, uint8_t * restrict L, float * restrict the_min,
        int ntry, float alpha) {
    float min = x[0];
    float max = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = 0;
        return 0.f;
    }
    if (min > 0) min = 0;
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    for (int itry = 0; itry < ntry; ++itry) {
        float sumlx = 0; int suml2 = 0;
        bool did_change = false;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            if (l != L[i]) {
                L[i] = l;
                did_change = true;
            }
            sumlx += (x[i] - min)*l;
            suml2 += l*l;
        }
        scale = sumlx/suml2;
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += x[i] - scale*L[i];
        }
        min = alpha*min + (1 - alpha)*sum/n;
        if (min > 0) min = 0;
        iscale = 1/scale;
        if (!did_change) break;
    }
    *the_min = -min;
    return scale;
}

static float make_qkx2_quants(int n, int nmax, const float * restrict x, const float * restrict weights,
        uint8_t * restrict L, float * restrict the_min, uint8_t * restrict Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float make_qkx3_quants(int n, int nmax, const float * restrict x, const float * restrict weights,
        uint8_t * restrict L, float * restrict the_min, uint8_t * restrict Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0]*x[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max <= min) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff*diff;
        float w = weights ? weights[i] : x[i]*x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff*diff;
                float w = weights ? weights[i] : x[i]*x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float make_qp_quants(int n, int nmax, const float * restrict x, uint8_t * restrict L, const float * quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    if (!max) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale*L[i];
        float w = quant_weights[i];
        best_mse += w*diff*diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]);
            l = MIN(nmax, l);
            float diff = x[i] - scale_is*l;
            float w = quant_weights[i];
            mse += w*diff*diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return sumlx/suml2;
}

static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size, const float * restrict x,
        ggml_fp16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(q4, 0, super_block_size/2);
    dh[0] = GGML_FP32_TO_FP16(0.f);

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < GROUP_MAX_EPS) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < block_size; ++j) {
            float al = id*xb[j];
            int l = best_index_int8(16, values, al);
            Lb[j] = l;
            float q = values[l];
            float w = weight[j];
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        d = sumqx/sumq2;
        float best = d*sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d * sumqx;
            }
        }
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    if (super_block_size/block_size > 1) {
        int nb = super_block_size/block_size;
        memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
        float d = -max_scale/32;
        dh[0] = GGML_FP32_TO_FP16(d);
        float id = d ? 1/d : 0.f;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            int l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            uint8_t * Lb = L + ib*block_size;
            const float * xb = x + ib*block_size;
            for (int j = 0; j < block_size; ++j) {
                Lb[j] = best_index_int8(16, values, idl*xb[j]);
            }
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            if (ib%2 == 0) scales_l[ib/2] = l_l;
            else scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } else {
        dh[0] = GGML_FP32_TO_FP16(scales[0]);
        if (ntry > 0) {
            float id = scales[0] ? 1/scales[0] : 0;
            for (int j = 0; j < super_block_size; ++j) {
                L[j] = best_index_int8(16, values, id*x[j]);
            }
        }
    }

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

// ---- Custom experiments ----

struct fraction {
    // float frac;
    float numer;
    float denom;
    int i;
};

// The k is the min of the element count and the threshold count
// Or should it always be the threshold count?
// It needs to be as shallow as possible, no?
struct k_heap_cell {
    int id; // identifier of the cell
    int i; // moving index in either k_heap->sorted_p or k_heap->sorted_n
           // depending on the sign of 'fixed'
    float fixed; // fixed
    float prod; // for quick comparison
};

// Needs to be able to let figure out which scale side is the best
// which means...
// The uneven side has to exist for both?
// Why does this seem more complicated than it should be?
// Because it is?
//

struct k_heap {
    bool sorted; // whether it's possible to simply shift the cells
    int top; // index of the top
    int p; // the length of sorted_p
    int n; // the length of sorted_n
    int k; // how many cells are remaining
    const float * sorted_p;
    const float * sorted_n;
    struct k_heap_cell cells[64];
};

// assuming both sorted_x and fixed are sorted descending
static void k_heap_init_sorted(struct k_heap * restrict heap, const float * restrict sorted_n, const float * restrict sorted_k, int n, int k) {
    GGML_ASSERT(k <= 64); // TODO: maybe bump?
    heap->sorted = true;
    heap->top = 0;
    heap->n = n;
    heap->k = k;
    heap->sorted_n = sorted_n;
    float first = sorted_n[0];
    for (int i = 0; i < k; ++i) {
        float fixed = sorted_k[i];
        heap->cells[i] = (struct k_heap_cell){
            .id    = i,
            .i     = 0,
            .fixed = fixed,
            .prod  = first * fixed,
        };
    }
}

static bool k_heap_pop(struct k_heap * restrict heap, int * restrict n_i, int * restrict k_i) {
    if (!heap || heap->k == 0) { return false; }
    int t = heap->top;
    struct k_heap_cell top = heap->cells[t];
    *n_i = top.i;
    *k_i = top.id;
    top.i += 1;
    // if (top.fixed < 0.0f ? top.i < heap->n : top.i < heap->p) {
    if (top.i < heap->n) {
        top.prod = heap->sorted_n[top.i] * top.fixed;
    } else {
        // Remove the top element, there are no more sub-elements inside
        heap->k -= 1;
        // Pretty much only maybe true for ternary and/or at first
        if (heap->sorted) {
            heap->top += 1;
            return true;
        }
        // Handle removal by putting the last item at the top
        top = heap->cells[heap->k];
    }
    bool valid = false;
    int i = 0;
    // Is it still the top?
    while (!valid) {
        if (2*i + 2 < heap->k) {
            float first = heap->cells[t + 2*i + 1].prod;
            float second = heap->cells[t + 2*i + 2].prod;
            bool second_max = first < second;
            if (top.prod < (second_max ? second : first)) {
                // swap
                int next = 2*i + (second_max ? 2 : 1);
                heap->cells[t + i] = heap->cells[t + next];
                i = next;
            } else {
                valid = true;
            }
        } else if (2*i + 1 < heap->k) {
            // Only a single leaf to compare with
            if (top.prod < heap->cells[t + 2*i + 1].prod) {
                // swap
                int next = 2*i + 1;
                heap->cells[t + i] = heap->cells[t + next];
                i = next;
            } else {
                valid = true;
            }
        } else {
            // Got to a leaf
            valid = true;
        }
    }
    if (i != 0) { heap->sorted = false; }
    heap->cells[t + i] = top;
    return true;
}

// comparator function for sorting fractions in make_qkxs_quants
static inline int compare_fractions_desc(const void * a, const void * b) {
    const struct fraction * f_a = (const struct fraction *) a;
    const struct fraction * f_b = (const struct fraction *) b;
    float na = f_a->numer;
    float da = f_a->denom;
    float nb = f_b->numer;
    float db = f_b->denom;

    // Stable sort
    // a - b sorts ascending, which means
    // 1 swaps, -1 stays
    if (da == db) { // equal denominators
        return (na == nb) ? ((a > b) ? 1 : -1) : (na < nb) ? 1 : -1;
    }
    if (na == nb) { // equal numerators
        return (da > db) ? 1 : -1;
    }
    float ab = na * db;
    float ba = nb * da;
    return (ab == ba) ? ((a > b) ? 1 : -1) : (ab < ba) ? 1 : -1;
}

// exhaustive search with cumulative sums
// Need Faux to have room for n*(max(abs(nmin), abs(nmax))) fractions
static float make_qkxs_quants(int n, int nmin, int nmax, const float * restrict x, const float * restrict weights, int8_t * restrict L, struct fraction * restrict Faux, bool signed_scale) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    bool negative_scale = false;
    if (signed_scale && -nmin != nmax) {
        // the max side should have the biggest range
        if ((max < 0.0f) == (-nmin < nmax)) {
            // [-4, 3] ==> [-3, 4]
            int tmp = nmin;
            nmin = -nmax;
            nmax = -tmp;
            negative_scale = true;
        }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.0f;
    }
    int n_frac = 0;
    for (int i = 0; i < n; ++i) {
        // assuming nmin <= nmax
        const int odd_max = MAX(0, x[i] < 0 ? -nmin : nmax);
        const int odd_min = MAX(0, x[i] < 0 ? -nmax : nmin);
        const float v = fabsf(x[i]);
        // fprintf(stderr, "%s: i=%d, odd_min=%d, odd_max=%d\n", __func__, i, odd_min, odd_max);
        for (int j = odd_min; j < odd_max; ++j) {
            const float odd = 2*j + 1;
            Faux[n_frac++] = (struct fraction){
                .numer=v,
                .denom=odd,
                .i=i,
            };
        }
    }

    qsort(Faux, n_frac, sizeof(struct fraction), compare_fractions_desc);

    float iscale = 0.0f;
    {
        float sumlx = 0.0f;
        float suml2 = 0.0f;
        float best = 0.0f;
        float best_denom = 1.0f;
        for (int i = 0; i < n_frac; ++i) {
            // maximize the weighted cosine
            const int ii = Faux[i].i;
            const float w = weights ? weights[ii] : x[ii] * x[ii];
            sumlx += w * Faux[i].numer;
            suml2 += w * Faux[i].denom;
            const float current = sumlx * sumlx;
            // fprintf(stderr, "%s: Faux[%d]=(%f/%f) * %f, square(sumlx)=%f, suml2=%f, k*cos2=%f\n", __func__, i, Faux[i].numer, Faux[i].denom, Faux[i].weight, current, suml2, current / suml2);
            // use the last in case of equality
            // FIXME: > or >= ?? Why does [0, 0, 1] rounds to [0, 0, 0] with >= ?
            if (suml2 > 0.0f && Faux[i].numer > 0.0f && current * best_denom > best * suml2) {
                best = current;
                best_denom = suml2;
                iscale = Faux[i].denom / (2.0f * Faux[i].numer);
                if (!isfinite(iscale)) {
                    fprintf(stderr, "%s: iscale is not finite, %f/(2*%f)\n", __func__, Faux[i].denom, Faux[i].numer);
                }
            }
        }
    }
    // (very) small fudging necessary because floats otherwise round to nearest even
    iscale = iscale * ((float)((1 << 23) + 1) / (float)(1 << 23));

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        // Rounding away from zero is assumed by the search algorithm above.
        int l = MAX(nmin, MIN(lroundf(x[i] * iscale), nmax));
        if (negative_scale) {
            l = -l;
        }
        L[i] = l;
        float w = weights ? weights[i] : x[i] * x[i];
        // weighted projection scale
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}


// non-linear exhaustive search with cumulative sums
// Need Faux to have room for n*k fractions
static float make_qkxs_nl_quants(int n, int k, const float * restrict x, const float * restrict weights, const int8_t * restrict kvalues, uint8_t * restrict L, uint8_t * restrict Laux, struct fraction * restrict Faux) {
    float max = 0.0f;
    float amax = 0.0f;
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    int kmin = abs(kvalues[0]);
    int koff = 0;
    for (int i = 1; i < k; ++i) {
        int ak = abs(kvalues[i]);
        if (ak < kmin) {
            kmin = ak;
            koff = i;
        }
    }
    kmin = kvalues[koff];
    for (int i = 0; i < n; ++i) {
        Laux[i] = koff;
        L[i] = koff;
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
        float w = weights ? weights[i] : x[i] * x[i];
        sumlx += w * x[i] * kmin;
        suml2 += w * kmin * kmin;
    }

    int n_frac = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.0f) {
            for (int j = koff; 0 < j; --j) {
                const float threshold = kvalues[j] + kvalues[j - 1];
                const float step = kvalues[j - 1] - kvalues[j]; // this should be negative
                Faux[n_frac++] = (struct fraction){
                    // This should always be positive or else
                    // the fraction comparison function won't work properly
                    // FIXME: think about signs
                    .numer=x[i] * step,
                    // It's amazing how this is still the difference of consecutive squares
                    .denom=threshold * step,
                    .i=i,
                };
            }
        } else {
            for (int j = koff + 1; j < k; ++j) {
                const float threshold = kvalues[j] + kvalues[j - 1];
                const float step = kvalues[j] - kvalues[j - 1];
                Faux[n_frac++] = (struct fraction){
                    .numer=x[i] * step,
                    .denom=threshold * step,
                    .i=i,
                };
            }
        }
    }

    qsort(Faux, n_frac, sizeof(struct fraction), compare_fractions_desc);

    float best = 0.0f;
    float best_sumlx = 0.0f;
    float best_suml2 = 1.0f;
    int best_i = -1;
    for (int i = 0; i < n_frac; ++i) {
        const int ii = Faux[i].i;
        const float w = weights ? weights[ii] : x[ii] * x[ii];
        sumlx += w * Faux[i].numer;
        suml2 += w * Faux[i].denom;
        const float current = sumlx * sumlx;
        Laux[ii] += x[ii] < 0.0f ? -1 : 1;
        // use the last in case of equality
        if (suml2 > 0.0f && current * best_suml2 >= best * suml2) {
            best = current;
            best_sumlx = sumlx;
            best_suml2 = suml2;
            if (i == best_i + 1) {
                // reduce copies for consecutive bests
                L[ii] += x[ii] < 0.0f ? -1 : 1;
            } else {
                for (int j = 0; j < n; ++j) {
                    L[j] = Laux[j];
                }
            }
            best_i = i;
        }
    }

    return best_suml2 != 0.0f ? best_sumlx / best_suml2 : 0.0f;
}

static void merge_sorted(struct fraction * restrict dest, const struct fraction * restrict a, const struct fraction * restrict b, int k) {
    struct fraction * max = compare_fractions_desc(a, b) > 0 ? b : a;
    int b_i = 0;
    for (int i = 0; i < 2*k; ++i) {

    }
}

static int compare_qkxsm_hepler(const void * a, const void * b) {
    float a_f = *(const float *) a;
    float b_f = *(const float *) b;

    // Stable descending sort
    // a - b sorts ascending, which means
    // 1 swaps, -1 stays
    return (a_f == b_f) ? (a > b ? 1 : -1) : (a_f < b_f ? 1 : -1);
}

#define FIXED_POINT_EPS (1.0f/(1 << 20))

// exhaustive search with cumulative sums, and a min
// Need Faux to have room for n*(nmax + 1) fractions
static float make_qkxcm_quants(int n, int nmax, const float * restrict x, const float * restrict weights, uint8_t * restrict L, float * restrict the_min, struct fraction * restrict Faux, uint8_t * restrict Laux) { // , float * restrict aux, uint8_t * restrict Laux) {
    // const bool trace = fabsf(x[0] - (-0.125f + 512.f/1024.f*0.25f)) < 1e-6 && fabsf(x[1] - (-0.125f + 205.f/1024.f*0.25f)) < 1e-6 && x[2] == 1;
    // const bool trace = fabsf(x[0] - (-1.f + 512.f/1024.f*2.f)) < 1e-6 && fabsf(x[1] - (-1.f + 205.f/1024.f*2.f)) < 1e-6 && x[2] == 1;
    const bool trace = fabsf(x[0] - (-1.f + 2047.f/2047.f*2.f)) < 1e-6 && fabsf(x[1] - (-1.f + 61.f/2047.f*2.f)) < 1e-6 && x[2] == 1;
    float max = x[0];
    float min = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    float sum_x2 = sum_w * x[0] * x[0];
    L[0] = 0;
    Laux[0] = 0;
    // fprintf(stderr, "%s: x=[%f", __func__, x[0]);
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) { min = x[i]; }
        if (x[i] > max) { max = x[i]; }
        const float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
        sum_x2 += w * x[i] * x[i];
        L[i] = 0;
        Laux[i] = 0;
        // fprintf(stderr, ", %f", x[i]);
    }
    // fprintf(stderr, "]\n");
    // sum_x2 -= sum_x * min;
    // sum_x -= min * sum_w;
    // What about negating the min? Can't, the scale isn't applied on the min.
    // The min needs to be strictly negative because then can be quantized unsigned.
    // Could a negative superblock min scale be used? Maybe, but that's out of scope here.
    if (max == min) {
        if (min < 0.0f) {
            for (int i = 0; i < n; ++i) { L[i] = 0; }
            *the_min = -min;
            return 0.0f;
        } else {
            for (int i = 0; i < n; ++i) { L[i] = 1; }
            *the_min = 0.0f;
            return min;
        }
    }
    if (min > 0.0f) { min = 0.0f; }
    if (sum_w <= 0.0f) {
        // should not happen?
        fprintf(stderr, "%s: should not happen, sum_w is %f\n", __func__, sum_w);
        sum_w = 1.0f;
    }

    int n_frac = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nmax; ++j) {
            // All the 0.5 thresholds -> 0.5 * (1, 3, 5, 7, 9, ...)
            // WARNING: THE ORDER IS IMPORTANT SOMEHOW (or not?)
            // const float odd = 2*(nmax - j) - 1;
            const float odd = 2*j + 1;
            // All possible scales
            Faux[n_frac++] = (struct fraction){x[i] - min, odd, i};
        }
    }
    qsort(Faux, n_frac, sizeof(struct fraction), compare_fractions_desc);

    const float mean_x = sum_x / sum_w;
    const float sum_x2m = sum_x2 - sum_x * mean_x;
    const float correction = (sum_x * sum_x) / (sum_x2 * sum_w);
    // We need to only project onto the hyperplane when it's possible
    // That means only in one direction
    // Does this mean the mean needs to be positive, or negative?
    // Does it depend on the mean?
    // Which one, the quantized mean or the full-precision mean?
    // Both?
    // How, exactly?
    float scale = 0.0f;
    float this_min = 0.0f;
    {
        float best_proj = 0.0f;
        float best_norm = 1.0f;
        float sumlx = 0.0f;
        float suml2 = 0.0f;
        float suml = 0.0f;
        float sumxm = 0.0f;
        float suml2m = 0.0f;
        int max_l = 0;
        for (int i = 0; i < n_frac; ++i) {
            const int ii = Faux[i].i;
            const float w = weights[ii];
            // fprintf(stderr, "%s: Faux[%d]: %f / %f = %f, w: %f\n", __func__, i, Faux[i].numer, Faux[i].denom, Faux[i].numer / Faux[i].denom, w);
            sumlx += w * x[ii];
            suml2 += w * Faux[i].denom;
            suml += w;

            float D = sum_w * suml2 - suml * suml;
            Laux[ii] += 1;
            if (Laux[ii] > max_l) { max_l = Laux[ii]; }

            float proj = sumlx * sumlx;
            float norm = suml2 * sum_x2;
            if (D > 0.0f) {
                float proto_scale = (sum_w * sumlx - sum_x * suml);
                float proto_min = (suml2 * sum_x - suml * sumlx) - (nmax - max_l)*proto_scale;

                if (proto_min < 0.0f) {
                    // FIXME: there's some precision issues sometimes. It's more noticeable with weights.

                    // Project the vectors onto the hyperplane normal to 1
                    // const float sumlxm = sumlx - sumxm;
                    const float sumlxm = sumlx - (suml * sum_x / sum_w); // same as (proto_scale / sum_w)
                    // the squared norm also needs to be projected
                    const float suml2m = suml2 - (suml * suml / sum_w); // same as (D / sum_w)
                    // const float suml2m = D / sum_w; // same as (D / sum_w)

                    const float normm = suml2m * sum_x2m;
                    const float projm_centered = sumlxm * sumlxm;
                    // FIXME: this isn't precise enough
                    const float projm = projm_centered + correction * (normm - projm_centered);

                    // TODO: make the algorithm more stable and remove the small fudging
                    // What about the squared error instead? Can that be calculated cumulatively too?
                    if (trace) {
                        fprintf(stderr, "%s: [%d] projm / normm = %g / %g = %f, proj / norm = %g / %g = %f\n", __func__, i, projm, normm, projm / normm, proj, norm, proj / norm);
                    }
                    if (normm > FIXED_POINT_EPS && normm >= projm && projm * norm > proj * normm) {
                        if (trace) {
                            fprintf(stderr, "%s: [%d] (projm * norm = %g * %g = %g) > (proj * normm = %g * %g = %g)\n", __func__, i, projm, norm, projm * norm, proj, normm, proj * normm);
                        }
                        proj = projm;
                        norm = normm;
                    } else {
                        proto_min = 0.0f;
                    }
                }
                // maximize the weighted correlation
                if (norm > 0.0f && proj * best_norm > best_proj * norm) {
                    best_proj = proj;
                    best_norm = norm;
                    if (proto_min < 0.0f) {
                        scale = proto_scale / D;
                        this_min = proto_min / D;
                        // if (this_min == -1.0f) { fprintf(stderr, "%s: found\n", __func__); }
                        for (int j = 0; j < n; ++j) {
                            L[j] = Laux[j] + (nmax - max_l);
                        }
                    } else {
                        scale = sumlx / suml2;
                        this_min = 0.0f;
                        for (int j = 0; j < n; ++j) {
                            L[j] = Laux[j];
                        }
                    }
                    // scale = proto_min < 0.0f ? proto_scale / D : sumlx / suml2;
                    // this_min = proto_min < 0.0f ? proto_min / D : 0.0f;
                    // fprintf(stderr, "%s: [%d] min = %f, best = %f / %f = %f\n", __func__, i, this_min, best_proj, best_norm, best_proj / best_norm);
                    // for (int j = 0; j < n; ++j) {
                    //     L[j] = Laux[j] + (nmax - max_l);
                    //     // fprintf(stderr, " %i,", L[j]);
                    // }
                    // // fprintf(stderr, "]\n");
                }
            } else {
                if (norm > 0.0f && proj * best_norm > best_proj * norm) {
                    best_proj = proj;
                    best_norm = norm;
                    scale = sumlx / suml2;
                    this_min = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        L[j] = Laux[j];
                    }
                }
            }
            if (trace) {
                fprintf(stderr, "%s: [%i] proj=%g, norm=%g, scale=%g, min=%g, L=[%i", __func__, i, best_proj, best_norm, scale, this_min, L[0]);
                for (int j = 1; j < n; ++j) {
                    fprintf(stderr, ", %i", L[j]);
                }
                fprintf(stderr, "], v=[%g", L[0] * scale + this_min);
                for (int j = 1; j < n; ++j) {
                    fprintf(stderr, ", %g", L[j] * scale + this_min);
                }
                fprintf(stderr, "]\n");
            }
        }
    }

    {
        float sumvx = 0.0f;
        float sumv2 = 0.0f;
        float sumx2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            float w = weights[i];
            float v = L[i] * scale + this_min;
            sumvx += w * v * x[i];
            sumv2 += w * v * v;
            sumx2 += w * x[i] * x[i];
        }
        float cos = sumvx / sqrtf(sumv2 * sumx2);
        if (trace && (cos < 0.5 || sumv2 * sumx2 == 0.0f)) {
            fprintf(stderr, "%s: small cos=%f (%f / sqrt(%f * %f)), scale=%f, min=%f\n", __func__, cos, sumvx, sumv2, sumx2, scale, -this_min);
            for (int i = 0; i < n; ++i) {
                float w = weights[i];
                float v = L[i] * scale + this_min;
                fprintf(stderr, "x[%i]=%.8f,\tv[%i]=%f,\tw[%i]=%f\n", i, x[i], i, v, i, w);
            }
        }
    }

    *the_min = -this_min;

    return scale;
}

static float make_qkxchm_quants(int n, int nmax, const float * restrict x, const float * restrict weights, uint8_t * restrict L, float * restrict the_min, struct k_heap * restrict k_heap, float * restrict aux, float * restrict Kaux) { // , float * restrict aux, uint8_t * restrict Laux) {
    float max = x[0];
    float min = x[0];
    float sum_w = weights[0];
    float sum_x = weights[0] * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) { min = x[i]; }
        if (x[i] > max) { max = x[i]; }
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    // if (sum_w < 0) {
    //     fprintf(stderr, "%s: warning, weird sum_w: %f\n", __func__, sum_w);
    //     sum_w = 1;
    // }
    float mean_x = sum_x / sum_w;
    // What about negating the min? Can't, the scale isn't applied on the min.
    // The min needs to be strictly negative because then can be quantized unsigned.
    // Could a negative superblock min scale be used? Maybe, but that's out of scope here.
    if (min > 0) { min = 0; }
    if (max == min) {
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        *the_min = -min;
        return 0.f;
    }

    for (int i = 0; i <= nmax; ++i) {
        Kaux[i] = 1.0f/(2*i + 1);
    }
    for (int i = 0; i < n; ++i) {
        aux[i] = x[i] - min;
    }

    // sort then use the heap
    qsort(aux, n, sizeof(float), compare_qkxsm_hepler);
    // FIXME: argsort the weights!?

    float iscale = 0;
    {
        int x_i = 0;
        int k_i = 0;
        int * n_ptr;
        int * k_ptr;
        if (nmax <= n) {
            k_heap_init_sorted(k_heap, aux, Kaux, n, nmax + 1);
            n_ptr = &x_i;
            k_ptr = &k_i;
        } else {
            k_heap_init_sorted(k_heap, Kaux, aux, nmax + 1, n);
            n_ptr = &k_i;
            k_ptr = &x_i;
        }

        float best = 0;
        float sumlx = 0;
        float suml2 = 0;
        float sq = 0;
        while(k_heap_pop(k_heap, n_ptr, k_ptr)) {
            float w = weights[x_i];
            // project onto the hyperplane normal to [1,1,1,1,...]
            sumlx += w * ((aux[x_i] + min) - mean_x);
            // FIXME: use w correctly
            sq += w; // consecutive squares
            // the squared norm also needs to be projected
            suml2 += w * (float)(2*k_i + 1);
            float norm = suml2 - (sq * sq) / sum_w;
            // maximize the cosine similarity
            float current = norm > 0 ? sumlx * sumlx / norm : 0;
            // Use the last scale
            if (current >= best) {
                best = current;
                iscale = (2*k_i + 1) / (2.0f * aux[x_i]);
            }
            // fprintf(stderr, "%s: suml: %d, i: %d, prod: %f\n", __func__, suml, i, aux[x_i] * Kaux[k_i]);
        }
    }

    // (very) small fudging necessary because floats otherwise round to nearest even
    iscale = iscale * ((float)((1 << 23) + 1) / (float)(1 << 23));

    // FIXME: how do weights interact with the mean?
    float sum_l = 0;
    for (int i = 0; i < n; ++i) {
        // Rounding away from zero is assumed by the search algorithm above.
        int l = MAX(0, MIN(lroundf((x[i] - min) * iscale), nmax));
        float w = weights[i];
        L[i] = l;
        sum_l += w * l;
    }
    float sumlx = 0;
    float suml2 = 0;
    float mean_l = sum_l / sum_w;
    for (int i = 0; i < n; ++i) {
        float l_m = L[i] - mean_l;
        float w = weights[i];
        sumlx += w * x[i] * l_m;
        suml2 += w * l_m * l_m;
    }
    // Calculate the best scale and min from that rounding
    float scale = suml2 > 0 ? sumlx / suml2 : 0;
    // float scale = suml2 / sumlx;

    float this_min = (scale * mean_l) - mean_x;
    if (this_min < 0) {
        // FIXME: change the scale to make the min less bad
        // fprintf(stderr, "%s: min clamped to zero, was %f, while real min was %f\n", __func__, this_min, min);
        this_min = 0;
    }
    *the_min = this_min;

    sumlx = 0;
    suml2 = 0;
    float sumx2 = 0;
    for (int i = 0; i < n; ++i) {
        float l = L[i] * scale - this_min;
        sumlx += l * x[i];
        suml2 += l * l;
        sumx2 += x[i] * x[i];
    }
    // fprintf(stderr, "%s: cos: %f\n", __func__, sumlx / sqrtf(suml2 * sumx2));

    return scale;
}

// ---- helper to be called from Python ----

void anyrize_qx(const float * x, const float * w, float * v, int ne0, int ne1, int nmax) {
    int8_t L[ne0];
    for (int i = 0; i < ne1; ++i) {
        float scale = make_qx_quants(ne0, nmax, x + ne0*i, L, 1, w ? w + i*ne0 : NULL);
        for (int j = 0; j < ne0; ++j) {
            v[i*ne0 + j] = (L[j] - nmax) * scale;
        }
    }
}

void anyrize_qkxs(const float * x, const float * w, float * v, int ne0, int ne1, int nmin, int nmax, bool signed_scale) {
    struct fraction Faux[ne0 * MAX(abs(nmin), abs(nmax))];
    int8_t L[ne0];
    for (int i = 0; i < ne1; ++i) {
        float scale = make_qkxs_quants(ne0, nmin, nmax, x + ne0*i, w ? w + i*ne0 : NULL, L, Faux, signed_scale);
        for (int j = 0; j < ne0; ++j) {
            v[i*ne0 + j] = L[j] * scale;
            if (!isfinite(v[i*ne0 + j])) {
                fprintf(stderr, "%s: invalid value? %f spotted\n", __func__, v[i*ne0 + j]);
            }
        }
    }
}

void anyrize_q3(const float * x, float * v, int ne0, int ne1, int nmax) {
    int8_t L[ne0];
    for (int i = 0; i < ne1; ++i) {
        float scale = make_q3_quants(ne0, nmax, x + ne0*i, L, true);
        for (int j = 0; j < ne0; ++j) {
            v[i*ne0 + j] = (L[j] - nmax) * scale;
            if (!isfinite(v[i*ne0 + j])) {
                fprintf(stderr, "%s: invalid value? %f spotted\n", __func__, v[i*ne0 + j]);
            }
        }
    }
}

void anyrize_qp(const float * x, const float * w, float * v, int ne0, int ne1, int nmax) {
    uint8_t L[ne0];
    float weights[ne0];
    for (int i = 0; i < ne1; ++i) {
        for (int j = 0; j < ne0; ++j) {
            weights[j] = w ? w[i*ne0 + j] : 1.0f;
        }
        float scale = make_qp_quants(ne0, nmax, x + ne0*i, L, weights);
        for (int j = 0; j < ne0; ++j) {
            v[i*ne0 + j] = L[j] * scale;
            if (!isfinite(v[i*ne0 + j])) {
                fprintf(stderr, "%s: invalid value? %f spotted\n", __func__, v[i*ne0 + j]);
            }
        }
    }
}

void anyrize_qkx2_q4_k(const float * x, float * v, int ne0, int ne1, int nmax) {
    uint8_t L[ne0];
    uint8_t Laux[ne0];
    float weights[ne0];
    float the_min;
    for (int i = 0; i < ne1; ++i) {
        float sum_x2 = 0;
        for (int l = 0; l < ne0; ++l) {
            sum_x2 += x[ne0*i + l] * x[ne0*i + l];
        }
        float av_x = sqrtf(sum_x2/ne0);
        for (int l = 0; l < ne0; ++l) {
            weights[l] = av_x + fabsf(x[ne0*i + l]);
            weights[l] = x[ne0*i + l] * x[ne0*i + l];
            // weights[l] = 1;
        }
        float scale = make_qkx2_quants(ne0, nmax, x + i*ne0, weights, L, &the_min, Laux, -1.f, 0.1f, 20, false);
        for (int j = 0; j < ne0; ++j) {
            v[i*ne0 + j] = L[j] * scale - the_min;
        }
    }
}

void anyrize_qkxcm_q4_k(const float * x, float * v, int ne0, int ne1, int nmax) {
    uint8_t L[ne0];
    uint8_t Laux[ne0];
    struct fraction Faux[ne0 * (nmax + 1)];
    float weights[ne0];
    float the_min = 0.0f;
    for (int i = 0; i < ne1; ++i) {
        float sum_x2 = 0;
        for (int l = 0; l < ne0; ++l) {
            sum_x2 += x[ne0*i + l] * x[ne0*i + l];
        }
        float av_x = sqrtf(sum_x2/ne0);
        for (int l = 0; l < ne0; ++l) {
            weights[l] = av_x + fabsf(x[ne0*i + l]);
            weights[l] = x[ne0*i + l] * x[ne0*i + l];
            // weights[l] = 1;
        }
        float scale = make_qkxcm_quants(ne0, nmax, x + i*ne0, weights, L, &the_min, Faux, Laux);
        for (int j = 0; j < ne0; ++j) {
            if (!isfinite(scale) || !isfinite(the_min)) {
                fprintf(stderr, "%s: scale is %f, min is %f\n", __func__, scale, the_min);
            }
            v[i*ne0 + j] = L[j] * scale - the_min;
        }
    }
}

void anyrize_qkx3_q4_k(const float * x, const float * w, float * v, int ne0, int ne1, int nmax) {
    uint8_t L[ne0];
    uint8_t Laux[ne0];
    float the_min;
    for (int i = 0; i < ne1; ++i) {
        // FIXME: preparation for weights
        float scale = make_qkx3_quants(ne0, nmax, x + i*ne0, w ? w + i*ne0 : NULL, L, &the_min, Laux, -0.9f, 0.05f, 36, false);
        for (int j = 0; j < ne0; ++j) {
            v[i*ne0 + j] = L[j] * scale - the_min;
        }
    }
}

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

void anyrize_iq4nl(const float * x, const float * w, float * v, int ne0, int ne1) {
    uint8_t L[ne0];
    uint8_t Laux[ne0];
    ggml_fp16_t unused_dh;
    uint8_t unused_q4[ne0];
    uint16_t unused_h;
    uint8_t * unused_l = NULL;
    float weight[ne0];
    for (int i = 0; i < ne1; ++i) {
        float scale = 0.0f;
        quantize_row_iq4_nl_impl(ne0, ne0, x + i*ne0, &unused_dh, unused_q4, &unused_h, unused_l, &scale, weight, L, kvalues_iq4nl, w ? w + i*ne0 : NULL, 7);
        for (int j = 0; j < ne0; ++j) {
            v[i*ne0 + j] = kvalues_iq4nl[L[j]] * scale;
        }
    }
}

void anyrize_qkxs_iq4nl(const float * x, const float * w, float * v, int ne0, int ne1) {
    uint8_t L[ne0];
    uint8_t Laux[ne0];
    struct fraction Faux[ne0 * 16];
    for (int i = 0; i < ne1; ++i) {
        float scale = make_qkxs_nl_quants(ne0, 16, x + i*ne0, w ? w + i*ne0 : NULL, kvalues_iq4nl, L, Laux, Faux);
        for (int j = 0; j < ne0; ++j) {
            v[i*ne0 + j] = kvalues_iq4nl[L[j]] * scale;
        }
    }
}

void anyrize_qkxs_iq4nl_signed(const float * x, const float * w, float * v, int ne0, int ne1) {
    uint8_t Lp[ne0];
    uint8_t Ln[ne0];
    uint8_t Laux[ne0];
    float neg_x[ne0];
    struct fraction Faux[ne0 * 16];
    for (int i = 0; i < ne1; ++i) {
        for (int j = 0; j < ne0; ++j) {
            neg_x[j] = -x[i*ne0 + j];
        }
        float scale_p = make_qkxs_nl_quants(ne0, 16, x + i*ne0, w ? w + i*ne0 : NULL, kvalues_iq4nl, Lp, Laux, Faux);
        float scale_n = make_qkxs_nl_quants(ne0, 16, neg_x, w ? w + i*ne0 : NULL, kvalues_iq4nl, Ln, Laux, Faux);
        float sumlx_p = 0.0f;
        float sumlx_n = 0.0f;
        for (int j = 0; j < ne0; ++j) {
            float ww = w ? w[i*ne0 + j] : x[i*ne0 + j] * x[i*ne0 + j];
            sumlx_p += ww * x[i*ne0 + j] * kvalues_iq4nl[Lp[j]];
            sumlx_n += ww * -x[i*ne0 + j] * kvalues_iq4nl[Ln[j]];
        }
        if (sumlx_n * scale_n > sumlx_p * scale_p) {
            for (int j = 0; j < ne0; ++j) {
                v[i*ne0 + j] = kvalues_iq4nl[Ln[j]] * -scale_n;
            }
        } else {
            for (int j = 0; j < ne0; ++j) {
                v[i*ne0 + j] = kvalues_iq4nl[Lp[j]] * scale_p;
            }
        }
    }
}
