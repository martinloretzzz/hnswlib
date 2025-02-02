#pragma once
#include "hnswlib.h"
#include <vector>
#include <cstddef>
#include <array>

namespace hnswlib {

template <unsigned BatchSize, unsigned b = 0>
struct BatchUnroll {
    static inline void add(const void *batch_ptr, float val1, size_t i, float* res) {
        const float* vec2 = static_cast<const float*>(((const float**)batch_ptr)[b]);
        res[b] += val1 * vec2[i];
        BatchUnroll<BatchSize, b + 1>::add(batch_ptr, val1, i, res);
    }
};

template <unsigned BatchSize>
struct BatchUnroll<BatchSize, BatchSize> {
    static inline void add(const void*, float, size_t, float*) { }
};

template <unsigned BatchSize>
static std::array<float, BatchSize> BatchedInnerProduct(const void *pVect1, void **batch_ptr, void *qty_ptr)
{
    size_t dim = *static_cast<size_t*>(qty_ptr);
    std::array<float, BatchSize> res{};

    for (size_t i = 0; i < dim; i++) {
        float val1 = static_cast<const float*>(pVect1)[i];
        BatchUnroll<BatchSize>::add(batch_ptr, val1, i, res.data());
    }

    for (unsigned b = 0; b < BatchSize; b++) {
        res[b] = 1.0f - res[b];
    }
    return res;
}

static std::vector<float> BatchedInnerProductForFixedSize(size_t batch_size, const void *pVect1, void **batch_ptr, void *qty_ptr)
{
    switch (batch_size) {
        case 0:
            return {};
#define HANDLE_CASE(N) \
    case N: { \
        auto tmp = BatchedInnerProduct<N>(pVect1, batch_ptr, qty_ptr); \
        return std::vector<float>(tmp.begin(), tmp.end()); \
    }
        HANDLE_CASE(1)   HANDLE_CASE(2)   HANDLE_CASE(3)   HANDLE_CASE(4)
        HANDLE_CASE(5)   HANDLE_CASE(6)   HANDLE_CASE(7)   HANDLE_CASE(8)
        HANDLE_CASE(9)   HANDLE_CASE(10)  HANDLE_CASE(11)  HANDLE_CASE(12)
        HANDLE_CASE(13)  HANDLE_CASE(14)  HANDLE_CASE(15)  HANDLE_CASE(16)
        HANDLE_CASE(17)  HANDLE_CASE(18)  HANDLE_CASE(19)  HANDLE_CASE(20)
        HANDLE_CASE(21)  HANDLE_CASE(22)  HANDLE_CASE(23)  HANDLE_CASE(24)
        HANDLE_CASE(25)  HANDLE_CASE(26)  HANDLE_CASE(27)  HANDLE_CASE(28)
        HANDLE_CASE(29)  HANDLE_CASE(30)  HANDLE_CASE(31)  HANDLE_CASE(32)
        HANDLE_CASE(33)  HANDLE_CASE(34)  HANDLE_CASE(35)  HANDLE_CASE(36)
        HANDLE_CASE(37)  HANDLE_CASE(38)  HANDLE_CASE(39)  HANDLE_CASE(40)
        HANDLE_CASE(41)  HANDLE_CASE(42)  HANDLE_CASE(43)  HANDLE_CASE(44)
        HANDLE_CASE(45)  HANDLE_CASE(46)  HANDLE_CASE(47)  HANDLE_CASE(48)
        HANDLE_CASE(49)  HANDLE_CASE(50)  HANDLE_CASE(51)  HANDLE_CASE(52)
        HANDLE_CASE(53)  HANDLE_CASE(54)  HANDLE_CASE(55)  HANDLE_CASE(56)
        HANDLE_CASE(57)  HANDLE_CASE(58)  HANDLE_CASE(59)  HANDLE_CASE(60)
        HANDLE_CASE(61)  HANDLE_CASE(62)  HANDLE_CASE(63)  HANDLE_CASE(64)
#undef HANDLE_CASE
        default:
            printf("Can't handle batched size of %d\n", (int)batch_size);
            return {};
    }
}

// Main function that handles arbitrary batch sizes by processing in chunks.
static std::vector<float> BatchedInnerProductForSize(size_t batch_size, const void *pVect1, void **batch_ptr, void *qty_ptr)
{
    if (batch_size <= 64)
        return BatchedInnerProductForFixedSize(batch_size, pVect1, batch_ptr, qty_ptr);

    printf("Batch bigger than 64!");

    return {};

    std::vector<float> result;
    size_t processed = 0;
    
    while (processed < batch_size) {
        // TODO is the right batch pointer provided?
        size_t curBatch = std::min(batch_size - processed, size_t(64));
        auto partial = BatchedInnerProductForFixedSize(curBatch, pVect1, batch_ptr, qty_ptr);
        result.insert(result.end(), partial.begin(), partial.end());
        processed += curBatch;
    }
    
    return result;
}

static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return res;
}

static float
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

static std::array<float, 2> BatchedInnerProductBatch2(const float* vec1, const float**batch, size_t dim)
{
    std::array<float, 2> res{};
    for (size_t i = 0; i < dim; i++) {
        float val1 = vec1[i];
        res[0] += val1 * batch[0][i];
        res[1] += val1 * batch[1][i];
    }
    return res;
}

static std::array<float, 4> BatchedInnerProductBatch4(const float* vec1, const float**batch, size_t dim)
{
    std::array<float, 4> res{};
    for (size_t i = 0; i < dim; i++) {
        float val1 = vec1[i];
        res[0] += val1 * batch[0][i];
        res[1] += val1 * batch[1][i];
        res[2] += val1 * batch[2][i];
        res[3] += val1 * batch[3][i];
    }
    return res;
}

static std::array<float, 8> BatchedInnerProductBatch8(const float* vec1, const float**batch, size_t dim)
{
    std::array<float, 8> res{};
    for (size_t i = 0; i < dim; i++) {
        float val1 = vec1[i];
        res[0] += val1 * batch[0][i];
        res[1] += val1 * batch[1][i];
        res[2] += val1 * batch[2][i];
        res[3] += val1 * batch[3][i];
        res[4] += val1 * batch[4][i];
        res[5] += val1 * batch[5][i];
        res[6] += val1 * batch[6][i];
        res[7] += val1 * batch[7][i];
    }
    return res;
}

static std::array<float, 16> BatchedInnerProductBatch16(const float* vec1, const float**batch, size_t dim)
{
    std::array<float, 16> res{};
    for (size_t i = 0; i < dim; i++) {
        float val1 = vec1[i];
        res[0] += val1 * batch[0][i];
        res[1] += val1 * batch[1][i];
        res[2] += val1 * batch[2][i];
        res[3] += val1 * batch[3][i];
        res[4] += val1 * batch[4][i];
        res[5] += val1 * batch[5][i];
        res[6] += val1 * batch[6][i];
        res[7] += val1 * batch[7][i];
        res[8] += val1 * batch[8][i];
        res[9] += val1 * batch[9][i];
        res[10] += val1 * batch[10][i];
        res[11] += val1 * batch[11][i];
        res[12] += val1 * batch[12][i];
        res[13] += val1 * batch[13][i];
        res[14] += val1 * batch[14][i];
        res[15] += val1 * batch[15][i];
    }
    return res;
}

static std::vector<float> BatchedInnerProductNoTemplate(size_t batch_size, const void *pVect1, void **pBatch, void *qty_ptr)
{
    size_t dim = *static_cast<size_t*>(qty_ptr);
    const float** batch = ((const float**) pBatch);
    const float* vec1 = static_cast<const float*>(pVect1);

    size_t processed = 0;
    std::vector<float> result;
    result.reserve(batch_size);

    while (processed < batch_size) {
        if (batch_size - processed >= 16) {
            auto batchResult = BatchedInnerProductBatch16(vec1, batch + processed, dim);
            result.insert(result.end(), batchResult.begin(), batchResult.end());
            processed += 16;
        } else if (batch_size - processed >= 8) {
            auto batchResult = BatchedInnerProductBatch8(vec1, batch + processed, dim);
            result.insert(result.end(), batchResult.begin(), batchResult.end());
            processed += 8;
        } else if (batch_size - processed >= 4) {
            auto batchResult = BatchedInnerProductBatch4(vec1, batch + processed, dim);
            result.insert(result.end(), batchResult.begin(), batchResult.end());
            processed += 4;
        } else if (batch_size - processed >= 2) {
            auto batchResult = BatchedInnerProductBatch2(vec1, batch + processed, dim);
            result.insert(result.end(), batchResult.begin(), batchResult.end());
            processed += 2;
        } else {
            float dist = InnerProduct(pVect1, batch[processed], qty_ptr);
            result.push_back(dist);
            processed += 1;
        }
    }
    
    for (unsigned b = 0; b < batch_size; b++) {
        result[b] = 1.0f - result[b];
    }

    return result;
}

#if defined(USE_SSE)

static std::array<float, 4>
BatchedInnerProductSIMD4(const void *pVec0, void **pBatch, void *qty_ptr) {
    float *pv0 = (float *) pVec0;

    const float** batch = ((const float**)pBatch);
    const float* pv1 = (const float*) batch[0];
    const float* pv2 = (const float*) batch[1];
    const float* pv3 = (const float*) batch[2];
    const float* pv4 = (const float*) batch[3];

    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty / 4;
    const float *pEnd0 = pv0 + 4 * qty4;

    __m128 v0, vx;

    __m128 s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps();
    __m128 s3 = _mm_setzero_ps();
    __m128 s4 = _mm_setzero_ps();

    for (; pv0 < pEnd0; pv0 += 4, pv1 += 4, pv2 += 4, pv3 += 4, pv4 += 4) {
        __m128 v0 = _mm_loadu_ps(pv0);
        s1 = _mm_add_ps(s1, _mm_mul_ps(v0, _mm_loadu_ps(pv1)));
        s2 = _mm_add_ps(s2, _mm_mul_ps(v0, _mm_loadu_ps(pv2)));
        s3 = _mm_add_ps(s3, _mm_mul_ps(v0, _mm_loadu_ps(pv3)));
        s4 = _mm_add_ps(s4, _mm_mul_ps(v0, _mm_loadu_ps(pv4)));
    }

    float PORTABLE_ALIGN32 tmp[8];
    std::array<float, 4> res{};

    _mm_store_ps(tmp, s1);
    res[0] = 1.0f - (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
    _mm_store_ps(tmp, s2);
    res[1] = 1.0f - (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
    _mm_store_ps(tmp, s3);
    res[2] = 1.0f - (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
    _mm_store_ps(tmp, s4);
    res[3] = 1.0f - (tmp[0] + tmp[1] + tmp[2] + tmp[3]);

    return res;
}


static std::vector<float> BatchedInnerProductSIMD(size_t batch_size, const void *pVect1, void **pBatch, void *qty_ptr)
{
    const float** batch = ((const float**)pBatch);
    std::vector<float> result;
    size_t processed = 0;
    result.reserve(batch_size);
    
    while (processed < batch_size) {
        void **batch_ptr = (void**) (batch + processed);
        size_t currentBatch = (batch_size - processed >= 4) ? 4 : (batch_size - processed);
        if (currentBatch == 4) {
            auto simdRes = BatchedInnerProductSIMD4(pVect1, batch_ptr, qty_ptr);
            result.insert(result.end(), simdRes.begin(), simdRes.end());
        } else {
            auto partial = BatchedInnerProductForFixedSize(currentBatch, pVect1, batch_ptr, qty_ptr);
            result.insert(result.end(), partial.begin(), partial.end());
        }
        processed += currentBatch;
    }
    
    return result;
}

#endif


#if defined(USE_AVX)

// Favor using AVX if available.
static float
InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128 v1, v2;
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return sum;
}

static float
InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif


#if defined(USE_AVX512)

static float
InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    size_t loop = qty16 / 4;
    
    while (loop--) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v3 = _mm512_loadu_ps(pVect1);
        __m512 v4 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v5 = _mm512_loadu_ps(pVect1);
        __m512 v6 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v7 = _mm512_loadu_ps(pVect1);
        __m512 v8 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        sum512 = _mm512_fmadd_ps(v3, v4, sum512);
        sum512 = _mm512_fmadd_ps(v5, v6, sum512);
        sum512 = _mm512_fmadd_ps(v7, v8, sum512);
    }

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
    }

    float sum = _mm512_reduce_add_ps(sum512);
    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_AVX)

static float
InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
static DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

static float
InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float
InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif


static std::vector<float> BatchedInnerProductForSizeSimple(size_t batch_size, const void *pVect1, void **batch_ptr, void *qty_ptr)
{
        size_t dim = *((size_t *) qty_ptr);
       auto fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        }
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif

    std::vector<float> result;
    for (int i = 0; i<batch_size; ++i) {
        float* vec2 = ((float**) batch_ptr)[i];
        float dist = fstdistfunc_(pVect1, vec2, qty_ptr);
        result.push_back(dist);
    }

    return result;
}


class InnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    InnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        }
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    BATCHEDDISTFUNC<float> get_dist_func_batched() {
        BATCHEDDISTFUNC<float> distfunc = BatchedInnerProductNoTemplate;
#if defined(USE_SSE)
        distfunc = BatchedInnerProductSIMD;
#endif
        return distfunc;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

~InnerProductSpace() {}
};

}  // namespace hnswlib
