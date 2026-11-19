// FP32-only standalone MLAS bridge, using ONNX Runtime v1.30.0 MIT sources.
// This bridge has no model, tokenizer, schema, RoPE or local attention-mask dependency.
#include "mlas.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <omp.h>

// Private pinned-v1.30.0 entry point. Its ABI is verified by strict linkage;
// use the upstream partition operation to preserve packed-B column offsets.
void MlasSgemmThreaded(ptrdiff_t,ptrdiff_t,CBLAS_TRANSPOSE,CBLAS_TRANSPOSE,
                       size_t,size_t,size_t,const MLAS_SGEMM_DATA_PARAMS*,ptrdiff_t);

namespace {
std::atomic<int> threads{1};
std::atomic<bool> upstream_partition{false};
void run(int ta, int tb, size_t m, size_t n, size_t k, const float* a,
         size_t lda, const float* b, size_t ldb, bool packed, float alpha,
         float beta, float* c, size_t ldc) {
    const auto operation = [&](size_t start, size_t count) {
        MLAS_SGEMM_DATA_PARAMS data;
        data.A = a + start * (ta == 111 ? lda : 1);
        data.lda = lda; data.B = b; data.ldb = ldb;
        data.C = c + start * ldc; data.ldc = ldc;
        data.alpha = alpha; data.beta = beta; data.BIsPacked = packed;
        MlasGemm(static_cast<CBLAS_TRANSPOSE>(ta), static_cast<CBLAS_TRANSPOSE>(tb),
                 count, n, k, data, nullptr, nullptr);
    };
    const int budget = threads.load();
    if (upstream_partition.load() && budget>1 && m*n*k>=1048576) {
        // Same M-vs-N choice as upstream MlasGemmBatch, with the explicit budget.
        const size_t available=n>m ? (n+15)/16 : m;
        const int active=std::min<size_t>(budget,available);
        MLAS_SGEMM_DATA_PARAMS data;
        data.A=a;data.lda=lda;data.B=b;data.ldb=ldb;data.C=c;data.ldc=ldc;
        data.alpha=alpha;data.beta=beta;data.BIsPacked=packed;
        #pragma omp parallel num_threads(active)
        {
            const int actual=omp_get_num_threads();
            MlasSgemmThreaded(n>m?1:actual,n>m?actual:1,
                static_cast<CBLAS_TRANSPOSE>(ta),static_cast<CBLAS_TRANSPOSE>(tb),
                m,n,k,&data,omp_get_thread_num());
        }
        return;
    }
    const size_t blocks = (m + 15) / 16;
    const int active = std::min<int>(budget, static_cast<int>(blocks));
    if (active <= 1 || m * n * k < 1048576) {
        operation(0, m);
        return;
    }
    // Each task calls serial standalone MLAS; no nested MLAS thread pool exists.
    #pragma omp parallel num_threads(active)
    {
        const int index = omp_get_thread_num();
        const int actual = omp_get_num_threads();
        const size_t first_block = blocks * index / actual;
        const size_t end_block = blocks * (index + 1) / actual;
        const size_t start = first_block * 16;
        const size_t end = std::min(m, end_block * 16);
        if (end > start) operation(start, end - start);
    }
}
}
extern "C" {
void candle_mlas_set_num_threads(int n) {
    if (n < 1 || n > 1024) std::abort();
    threads.store(n); omp_set_dynamic(0); omp_set_num_threads(n);
}
int candle_mlas_get_num_threads() { return threads.load(); }
void candle_mlas_set_partition(int mode) {
    if (mode != 0 && mode != 1) std::abort();
    upstream_partition.store(mode == 1);
}
int candle_mlas_get_partition() { return upstream_partition.load() ? 1 : 0; }
const char* candle_mlas_get_config() {
    return "Standalone MLAS FP32 SGEMM; ORT v1.30.0 f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7; M-independent B packing; OpenMP row tasks and batched heads";
}
void cblas_sgemm(int layout, int ta, int tb, int m, int n, int k, float alpha,
                 const float* a, int lda, const float* b, int ldb, float beta,
                 float* c, int ldc) {
    if (layout != 101 || m < 0 || n < 0 || k < 0) std::abort();
    run(ta,tb,m,n,k,a,lda,b,ldb,false,alpha,beta,c,ldc);
}
size_t cblas_sgemm_pack_get_size(int identifier, int, int n, int k) {
    if (identifier != 162 || n <= 0 || k <= 0) std::abort();
    return MlasGemmPackBSize(CblasNoTrans,CblasTrans,n,k,nullptr);
}
void cblas_sgemm_pack(int layout, int identifier, int trans, int, int n, int k,
                      float alpha, const float* source, int ld, float* dest) {
    if (layout != 101 || identifier != 162 || trans != 112 || alpha != 1.f) std::abort();
    MlasGemmPackB(CblasNoTrans,CblasTrans,n,k,source,ld,dest,nullptr);
}
void cblas_sgemm_compute(int layout, int ta, int tb, int m, int n, int k,
                         const float* a, int lda, const float* b, int ldb,
                         float beta, float* c, int ldc) {
    if (layout != 101 || ta != 111 || tb != 151) std::abort();
    run(ta,112,m,n,k,a,lda,b,ldb,true,1.f,beta,c,ldc);
}
void candle_mlas_geglu(const float* input, float* output, size_t rows, size_t width) {
    const int budget=threads.load();
    #pragma omp parallel for num_threads(budget) schedule(static)
    for (size_t r=0;r<rows;++r) {
        const float* gate=input+r*2*width;
        float* dest=output+r*width;
        MlasComputeGeluErf(gate,dest,width);
        for (size_t j=0;j<width;++j) dest[j]*=gate[width+j];
    }
}
void candle_mlas_sgemm_batch(int ta,int tb,size_t batch,size_t m,size_t n,size_t k,
                           const float* a,size_t lda,size_t stride_a,
                           const float* b,size_t ldb,size_t stride_b,
                           float* c,size_t ldc,size_t stride_c) {
    const int budget=threads.load();
    #pragma omp parallel for num_threads(budget) schedule(static)
    for(size_t i=0;i<batch;++i) {
        MLAS_SGEMM_DATA_PARAMS data;
        data.A=a+i*stride_a;data.lda=lda;data.B=b+i*stride_b;data.ldb=ldb;
        data.C=c+i*stride_c;data.ldc=ldc;
        MlasGemm(static_cast<CBLAS_TRANSPOSE>(ta),static_cast<CBLAS_TRANSPOSE>(tb),m,n,k,data,nullptr,nullptr);
    }
}
void candle_mlas_layer_norm(const float* input,const float* weight,const float* bias,float* output,
                          size_t rows,size_t width,float epsilon) {
    const int budget=threads.load();
    #pragma omp parallel for num_threads(budget) schedule(static) if(rows*width>=16384)
    for(size_t row=0;row<rows;++row) {
        const float* x=input+row*width;float* y=output+row*width;
        float mean,inv;
        if(!MlasLayerNormF32(x,weight,bias,y,&mean,&inv,width,epsilon,false)) {
            double average=0.,variance=0.;
            for(size_t j=0;j<width;++j) average+=x[j];
            average/=width;
            for(size_t j=0;j<width;++j) {const double z=x[j]-average;variance+=z*z;}
            const double denominator=std::sqrt(variance/width+epsilon);
            for(size_t j=0;j<width;++j) y[j]=static_cast<float>((x[j]-average)/denominator)*weight[j]+bias[j];
        }
    }
}
uint32_t candle_mlas_abi_version() { return 2; }
void candle_mlas_softmax_f32(const float* input,float* output,size_t rows,size_t width) {
    MlasComputeSoftmax(input,output,rows,width,false,false,0.f,nullptr);
}

}
