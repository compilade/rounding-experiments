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

// ---- Custom experiments ----

struct fraction {
    // float frac;
    float numer;
    float denom;
    float weight;
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
// Need Faux to have room for n*nmax fractions
static float make_qkxs_quants(int n, int nmin, int nmax, const float * restrict x, const float * restrict weights, uint8_t * restrict L, struct k_heap * restrict k_heap, float * restrict aux, float * restrict Kaux) {
    int n_k = nmax - nmin; // assuming this is positive
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
    for (int i = 0; i < n; ++i) {
        aux[i] = fabsf(x[i]);
    }
    for (int i = 0; i < n_k; ++i) {
        
    }
    if (nmax % 2 == 0) {
        // asymmetrical rounding
    }
    // TODO: how to handle the min?
    // What does it mean geometrically?

    // The scale means we only care about the angle.
    // The min means everything can slide along the [1,1,1,...] diagonal
    // Can that be used?
    // Yes...
    // How?

    // sort (using qsort)
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

// exhaustive search with cumulative sums, and a min
// Need Faux to have room for n*(nmax + 1) fractions
static float make_qkxcm_quants(int n, int nmax, const float * restrict x, const float * restrict weights, uint8_t * restrict L, float * restrict the_min, struct fraction * restrict Faux) { // , float * restrict aux, uint8_t * restrict Laux) {
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
    if (sum_w <= 0) {
        // should not happen?
        fprintf(stderr, "%s: should not happen, sum_w is %f\n", __func__, sum_w);
        sum_w = 1;
    }
    // What about negating the min? Can't, the scale isn't applied on the min.
    // The min needs to be strictly negative because then can be quantized unsigned.
    // Could a negative superblock min scale be used? Maybe, but that's out of scope here.
    if (min > 0) { min = 0; }
    if (max == min) {
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        *the_min = -min;
        return 0.f;
    }

    // TODO: why nmax + 1 odd numbers?
    // because
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= nmax; ++j) {
            // All the 0.5 thresholds -> 0.5 * (1, 3, 5, 7, 9, ...)
            float odd = 2*j + 1;
            // All possible scales
            Faux[i*(nmax + 1) + j] = (struct fraction){x[i] - min, odd, weights[i]};
        }
    }
    qsort(Faux, n*(nmax + 1), sizeof(struct fraction), compare_fractions_desc);

    const float mean_x = sum_x / sum_w;
    float iscale = 0;
    {
        float best = 0;
        float current = 0;
        float sumlx = 0;
        float suml2 = 0;
        float sq = 0;
        for (int i = 0; i < n*(nmax + 1); ++i) {
            // fprintf(stderr, "%s: Faux[%d]: %f / %f = %f, w: %f\n", __func__, i, Faux[i].numer, Faux[i].denom, Faux[i].numer / Faux[i].denom, Faux[i].weight);
            // project onto the hyperplane normal to [1,1,1,1,...]
            sumlx += Faux[i].weight * (Faux[i].numer + min - mean_x);
            sq += Faux[i].weight;
            suml2 += Faux[i].weight * Faux[i].denom;
            // the squared norm also needs to be projected
            // TODO: consecutive squares without multiplying
            float norm = suml2 - (sq * sq) / sum_w;
            // maximize the weighted correlation
            current = norm > 0 ? sumlx * sumlx / norm : 0;
            // Use the last scale
            if (current >= best) {
                best = current;
                iscale = Faux[i].denom / (2.0f * Faux[i].numer);
            }
        }
    }
    // (very) small fudging necessary because floats otherwise round to nearest even
    iscale = iscale * ((float)((1 << 23) + 1) / (float)(1 << 23));

    float sum_l = 0;
    for (int i = 0; i < n; ++i) {
        // Rounding away from zero is assumed by the search algorithm above.
        int l = MAX(0, MIN(lroundf((x[i] - min) * iscale), nmax));
        L[i] = l;
        sum_l += weights[i] * l;
    }

    float sum_xl = 0;
    float sum_l2 = 0;
    // float sum_lm = 0;
    // float sum_xm = 0;
    const float mean_l = sum_l / sum_w;
    for (int i = 0; i < n; ++i) {
        const float w = weights[i];
        const float l_m = L[i] - mean_l;
        const float x_m = x[i] - mean_x;
        sum_xl = w * x_m * l_m;
        sum_l2 = w * l_m * l_m;
        // sum_lm = w * l_m;
        // sum_xm = w * x_m;
    }
    // float scale = 0;
    // // TODO: find somewhere this is formally proven
    // float D = sum_w * sum_l2 - sum_lm * sum_lm;
    // if (D > 0) {
    //     scale = (sum_w * sum_xl - sum_xm * sum_lm)/D;
    //     min   = (sum_l2 * sum_xm - sum_lm * sum_xl)/D;
    //     if (min > 0) {
    //         min = 0;
    //         scale = sum_xl / sum_l2;
    //     }
    // } else if (D < 0) {
    //     fprintf(stderr, "%s: should not happen, D is %f, sum_w is %f, sum_l2 is %f, sum_l is %f\n", __func__, D, sum_w, sum_l2, sum_l);
    // }
    // Calculate the best scale and min from that rounding
    float scale = sum_l2 > 0 ? sum_xl / sum_l2 : 0;

    float this_min = (scale * mean_l) - mean_x;
    if (this_min < 0) {
        // fprintf(stderr, "%s: min clamped to zero, was %f, while real min was %f\n", __func__, this_min, min);
        this_min = 0;
    }
    *the_min = this_min;

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
