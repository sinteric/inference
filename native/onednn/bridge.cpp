// RETIRED. Production binds oneDNN's C API from Rust (crates/onednn-sys,
// crates/rt-onednn). This file is kept as the numeric reference for the
// primitives the C API path must match: strict fpmath/accumulation, user
// scratchpads, kind-7 GeGLU (GELU_ERF + BINARY_MUL), packed weight reorder.
// Rebuild with native/onednn/build.sh only for A/B diffs — not linked by default.
#include <oneapi/dnnl/dnnl.hpp>
#include <omp.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
using namespace dnnl;
namespace {
engine &cpu(){static engine e(engine::kind::cpu,0);return e;}
primitive_attr attrs(bool sum=false){primitive_attr a;a.set_fpmath_mode(fpmath_mode::strict);a.set_accumulation_mode(accumulation_mode::strict);a.set_scratchpad_mode(scratchpad_mode::user);if(sum){post_ops p;p.append_sum(1.f);a.set_post_ops(p);}return a;}
void err(char*dst,size_t cap,const char*s){if(dst&&cap){std::strncpy(dst,s,cap-1);dst[cap-1]=0;}}
struct Weight {memory::desc md,raw;reorder::primitive_desc pd;reorder op;int threads;
 static memory::desc select(int64_t n,int64_t k){memory::desc a({64,k},memory::data_type::f32,memory::dims{k,1}),b({k,n},memory::data_type::f32,memory::format_tag::any),c({64,n},memory::data_type::f32,memory::dims{n,1});return matmul::primitive_desc(cpu(),a,b,c,attrs()).weights_desc();}
 Weight(int64_t n,int64_t k,int t):md(select(n,k)),raw({k,n},memory::data_type::f32,memory::dims{1,k}),pd(cpu(),raw,cpu(),md,attrs()),op(pd),threads(t){}
};
struct Op {primitive prim;memory::desc a,b,c,scratch,scale,shift,gate;int kind,threads;std::string impl;stream execution_stream{cpu()};std::unordered_map<int,memory> args;};
}
extern "C" {
unsigned vela_dnnl_abi(){auto*v=dnnl_version();return v&&v->major==3&&v->minor==13&&v->patch==4?1:0;}
const char*vela_dnnl_revision(){return "oneDNN3.13.4 c531fe1a5cd631b7f60277cdf6675c588cadbc28; standalone Rust-owned data; public primitives; FP32 strict";}
int vela_dnnl_configure(){return (int)dnnl_set_primitive_cache_capacity(0);}
void*vela_dnnl_weight_create(int64_t n,int64_t k,int threads,size_t*bytes,size_t*scratch,char*e,size_t cap){try{if(n<=0||k<=0||threads<=0||!bytes||!scratch)throw std::invalid_argument("weight shape");omp_set_dynamic(0);omp_set_num_threads(threads);auto w=std::make_unique<Weight>(n,k,threads);*bytes=w->md.get_size();*scratch=w->pd.scratchpad_desc().get_size();return w.release();}catch(const std::exception&x){err(e,cap,x.what());return nullptr;}}
void vela_dnnl_weight_destroy(void*p){delete static_cast<Weight*>(p);}
int vela_dnnl_weight_reorder(void*p,const float*src,void*dst,void*scratch,size_t bytes,char*e,size_t cap){try{if(!p||!src||!dst||!scratch)throw std::invalid_argument("reorder pointer");auto&w=*static_cast<Weight*>(p);if(bytes<w.pd.scratchpad_desc().get_size())throw std::invalid_argument("scratch too small");omp_set_num_threads(w.threads);stream s(cpu());w.op.execute(s,{{DNNL_ARG_FROM,memory(w.raw,cpu(),const_cast<float*>(src))},{DNNL_ARG_TO,memory(w.md,cpu(),dst)},{DNNL_ARG_SCRATCHPAD,memory(w.pd.scratchpad_desc(),cpu(),scratch)}});s.wait();return 0;}catch(const std::exception&x){err(e,cap,x.what());return 1;}}
// kind:0 matmul,1 layernorm,2 softmax,3 GELU-erf,4 binary-mul,5 ReLU,7 MatMul+GELU+gate.
// Matmul strides are in FP32 elements; g=1 uses 2D, otherwise 3D batching.
void*vela_dnnl_op_create(int kind,int64_t g,int64_t m,int64_t n,int64_t k,int64_t ab,int64_t ar,int64_t bb,int64_t br,int trans,float parameter,int sum,int threads,const void*weight,size_t*scratch,char*e,size_t cap){try{
 if(g<=0||m<=0||n<=0||threads<=0||!scratch)throw std::invalid_argument("operator shape");omp_set_num_threads(threads);auto o=std::make_unique<Op>();o->kind=kind;o->threads=threads;
 if(kind==0||kind==7){if(k<=0)throw std::invalid_argument("matmul K");
  o->a=g==1?memory::desc({m,k},memory::data_type::f32,memory::dims{ar,1}):memory::desc({g,m,k},memory::data_type::f32,memory::dims{ab,ar,1});
  if(weight){if(g!=1)throw std::invalid_argument("batched packed weight");o->b=static_cast<const Weight*>(weight)->md;}
  else if(g==1)o->b=memory::desc({k,n},memory::data_type::f32,trans?memory::dims{1,br}:memory::dims{br,1});
  else o->b=memory::desc({g,k,n},memory::data_type::f32,trans?memory::dims{bb,1,br}:memory::dims{bb,br,1});
  o->c=g==1?memory::desc({m,n},memory::data_type::f32,memory::dims{n,1}):memory::desc({g,m,n},memory::data_type::f32,memory::dims{m*n,n,1});
  auto attr=attrs(sum);if(kind==7){if(sum)throw std::invalid_argument("gated projection sum");o->gate=o->c;post_ops post;post.append_eltwise(algorithm::eltwise_gelu_erf,0.f,0.f);post.append_binary(algorithm::binary_mul,o->gate);attr.set_post_ops(post);}auto pd=matmul::primitive_desc(cpu(),o->a,o->b,o->c,attr);o->scratch=pd.scratchpad_desc();o->impl=pd.impl_info_str();o->prim=matmul(pd);
 }else{ o->a=memory::desc({m,n},memory::data_type::f32,memory::dims{n,1});o->c=o->a;
  if(kind==1){o->scale=memory::desc({n},memory::data_type::f32,memory::dims{1});o->shift=o->scale;auto pd=layer_normalization_forward::primitive_desc(cpu(),prop_kind::forward_inference,o->a,o->c,parameter,normalization_flags::use_scale|normalization_flags::use_shift,attrs());o->scratch=pd.scratchpad_desc();o->impl=pd.impl_info_str();o->prim=layer_normalization_forward(pd);}
  else if(kind==2){auto pd=softmax_forward::primitive_desc(cpu(),prop_kind::forward_inference,algorithm::softmax_accurate,o->a,o->c,1,attrs());o->scratch=pd.scratchpad_desc();o->impl=pd.impl_info_str();o->prim=softmax_forward(pd);}
  else if(kind==3||kind==5){auto pd=eltwise_forward::primitive_desc(cpu(),prop_kind::forward_inference,kind==3?algorithm::eltwise_gelu_erf:algorithm::eltwise_relu,o->a,o->c,0.f,0.f,attrs());o->scratch=pd.scratchpad_desc();o->impl=pd.impl_info_str();o->prim=eltwise_forward(pd);}
  else if(kind==4){o->b=o->a;auto pd=binary::primitive_desc(cpu(),algorithm::binary_mul,o->a,o->b,o->c,attrs());o->scratch=pd.scratchpad_desc();o->impl=pd.impl_info_str();o->prim=binary(pd);}
  else throw std::invalid_argument("operator kind");
 }
 o->args.emplace(kind==4?DNNL_ARG_SRC_0:DNNL_ARG_SRC,memory(o->a,cpu(),DNNL_MEMORY_NONE));
 o->args.emplace(DNNL_ARG_DST,memory(o->c,cpu(),DNNL_MEMORY_NONE));
 o->args.emplace(DNNL_ARG_SCRATCHPAD,memory(o->scratch,cpu(),DNNL_MEMORY_NONE));
 if(kind==0||kind==7)o->args.emplace(DNNL_ARG_WEIGHTS,memory(o->b,cpu(),DNNL_MEMORY_NONE));
 else if(kind==1){o->args.emplace(DNNL_ARG_SCALE,memory(o->scale,cpu(),DNNL_MEMORY_NONE));o->args.emplace(DNNL_ARG_SHIFT,memory(o->shift,cpu(),DNNL_MEMORY_NONE));}
 else if(kind==4)o->args.emplace(DNNL_ARG_SRC_1,memory(o->b,cpu(),DNNL_MEMORY_NONE));
 if(kind==7)o->args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(1)|DNNL_ARG_SRC_1,memory(o->gate,cpu(),DNNL_MEMORY_NONE));
 *scratch=o->scratch.get_size();return o.release();
 }catch(const std::exception&x){err(e,cap,x.what());return nullptr;}}
void vela_dnnl_op_destroy(void*p){delete static_cast<Op*>(p);}
const char*vela_dnnl_op_impl(void*p){return p?static_cast<Op*>(p)->impl.c_str():"null";}
int vela_dnnl_execute(void*p,const float*a,const float*b,const float*shift,float*c,void*scratch,size_t bytes,char*e,size_t cap){try{if(!p||!a||!c||!scratch)throw std::invalid_argument("execution pointer");auto&o=*static_cast<Op*>(p);if(bytes<o.scratch.get_size())throw std::invalid_argument("scratch too small");omp_set_num_threads(o.threads);
 o.args.at(o.kind==4?DNNL_ARG_SRC_0:DNNL_ARG_SRC).set_data_handle(const_cast<float*>(a));o.args.at(DNNL_ARG_DST).set_data_handle(c);o.args.at(DNNL_ARG_SCRATCHPAD).set_data_handle(scratch);
 if(o.kind==0||o.kind==7){if(!b)throw std::invalid_argument("matmul weight pointer");o.args.at(DNNL_ARG_WEIGHTS).set_data_handle(const_cast<float*>(b));}
 else if(o.kind==1){if(!b||!shift)throw std::invalid_argument("norm scale/shift");o.args.at(DNNL_ARG_SCALE).set_data_handle(const_cast<float*>(b));o.args.at(DNNL_ARG_SHIFT).set_data_handle(const_cast<float*>(shift));}
 else if(o.kind==4){if(!b)throw std::invalid_argument("binary pointer");o.args.at(DNNL_ARG_SRC_1).set_data_handle(const_cast<float*>(b));}
 if(o.kind==7){if(!shift)throw std::invalid_argument("gate input");o.args.at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(1)|DNNL_ARG_SRC_1).set_data_handle(const_cast<float*>(shift));}
 o.prim.execute(o.execution_stream,o.args);o.execution_stream.wait();return 0;
 }catch(const std::exception&x){err(e,cap,x.what());return 1;}}
}
