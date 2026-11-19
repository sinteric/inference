// Vela-specific fused layouts and padding/local attention masks.
#include "mlas.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
extern "C" int candle_mlas_get_num_threads();
extern "C" void candle_mlas_set_num_threads(int);
extern "C" const char* candle_mlas_get_config();
extern "C" void candle_mlas_softmax_f32(const float*,float*,size_t,size_t);
extern "C" {
// Compatibility metadata names for the Vela frontend; the backend owns the budget.
int vela_mlas_get_num_threads() { return candle_mlas_get_num_threads(); }
void vela_mlas_set_num_threads(int n) { candle_mlas_set_num_threads(n); }
const char* vela_mlas_get_config() { return candle_mlas_get_config(); }
void vela_mlas_qkv_rope(const float* input,const float* cosine,const float* sine,
                         float* output,size_t batch,size_t seq,size_t heads,size_t dim,float scale) {
    const size_t hidden=heads*dim;const size_t block=batch*seq*hidden;
    const int budget=candle_mlas_get_num_threads();
    #pragma omp parallel for collapse(2) num_threads(budget) schedule(static)
    for(size_t b=0;b<batch;++b) for(size_t h=0;h<heads;++h) {
        for(size_t p=0;p<seq;++p) {
            const float* source=input+(b*seq+p)*3*hidden+h*dim;
            const size_t dest=((b*heads+h)*seq+p)*dim;
            const float* c=cosine+p*(dim/2);const float* s=sine+p*(dim/2);
            for(size_t j=0;j<dim/2;++j) {
                const float q1=source[j],q2=source[j+dim/2];
                const float k1=source[hidden+j],k2=source[hidden+j+dim/2];
                output[dest+j]=(q1*c[j]-q2*s[j])*scale;output[dest+j+dim/2]=(q1*s[j]+q2*c[j])*scale;
                output[block+dest+j]=k1*c[j]-k2*s[j];output[block+dest+j+dim/2]=k1*s[j]+k2*c[j];
            }
            std::copy_n(source+2*hidden,dim,output+2*block+dest);
        }
    }
}
void vela_mlas_masked_softmax(const float* input,const float* mask,float* output,
                              size_t batch,size_t heads,size_t queries,size_t keys) {
    const size_t rows=batch*heads*queries;const int budget=candle_mlas_get_num_threads();
    #pragma omp parallel for num_threads(budget) schedule(static)
    for(size_t row=0;row<rows;++row) {
        const size_t b=row/(heads*queries),q=row%queries;
        const float* m=mask+(b*queries+q)*keys;float* dest=output+row*keys;
        for(size_t j=0;j<keys;++j) dest[j]=input[row*keys+j]+m[j];
        // Same FP32 softmax flags as the official ORT CPU kernel.
        candle_mlas_softmax_f32(dest,dest,1,keys);
    }
}
void vela_mlas_unpack_heads(const float* input,float* output,size_t batch,size_t heads,size_t seq,size_t dim) {
    const int budget=candle_mlas_get_num_threads();
    #pragma omp parallel for collapse(2) num_threads(budget) schedule(static)
    for(size_t b=0;b<batch;++b) for(size_t p=0;p<seq;++p) {
        for(size_t h=0;h<heads;++h) std::copy_n(input+((b*heads+h)*seq+p)*dim,dim,output+((b*seq+p)*heads+h)*dim);
    }
}
void vela_mlas_masked_softmax_u32(const float* input,const uint32_t* mask,float* output,
                                  size_t batch,size_t heads,size_t queries,size_t keys,
                                  size_t seq,size_t query_start,size_t key_start,int radius) {
    const size_t rows=batch*heads*queries;const int budget=candle_mlas_get_num_threads();
    const float denied=std::numeric_limits<float>::lowest();
    #pragma omp parallel for num_threads(budget) schedule(static)
    for(size_t row=0;row<rows;++row) {
        const size_t b=row/(heads*queries),q=row%queries,qpos=query_start+q;
        const uint32_t* padding=mask+b*seq+key_start;float* dest=output+row*keys;
        size_t lo=0,hi=keys;
        if(radius>=0) {
            const size_t r=static_cast<size_t>(radius);
            lo=std::max(key_start,qpos>r?qpos-r:0)-key_start;
            hi=std::min(key_start+keys,qpos+r+1)-key_start;
        }
        for(size_t j=0;j<lo;++j) dest[j]=input[row*keys+j]+denied;
        for(size_t j=lo;j<hi;++j) dest[j]=input[row*keys+j]+(padding[j]?0.f:denied);
        for(size_t j=hi;j<keys;++j) dest[j]=input[row*keys+j]+denied;
        candle_mlas_softmax_f32(dest,dest,1,keys);
    }
}
}
