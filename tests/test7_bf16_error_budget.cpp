// test7_bf16_error_budget.cpp
// Measures bf16 quantization error between explicit-inject (Path A) vs bake (Path B).
//
// Path A: base_bf16 @ X_bf16  +  delta_bf16 @ X_h_bf16  (two separate GEMMs, f32 accumulate)
// Path B: bake_bf16 @ X_bf16  where bake_bf16 = round_f32_bf16(base_f32 + delta_f32)  (one GEMM)
//
// Quantifies the bf16 non-associativity error across the full FFN.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_CHK(c) do{hipError_t e=c;if(e!=hipSuccess){fprintf(stderr,"HIP%d L%d\n",e,__LINE__);exit(1);}}while(0)
#define RB_CHK(c)  do{rocblas_status s=c;if(s!=rocblas_status_success){fprintf(stderr,"RB%d L%d\n",s,__LINE__);exit(1);}}while(0)

static uint16_t f2bf(float f){uint32_t b;memcpy(&b,&f,4);b+=((b>>16)&1)+0x7FFF;return(uint16_t)(b>>16);}
static float bf2f(uint16_t v){uint32_t b=(uint32_t)v<<16;float o;memcpy(&o,&b,4);return o;}

static void h2d_bf(const float*h,int n,uint16_t**d){
    std::vector<uint16_t> bf(n); for(int i=0;i<n;i++)bf[i]=f2bf(h[i]);
    HIP_CHK(hipMalloc(d,n*sizeof(uint16_t))); HIP_CHK(hipMemcpy(*d,bf.data(),n*sizeof(uint16_t),hipMemcpyHostToDevice));
}
static void h2d_f32(const float*h,int n,float**d){
    HIP_CHK(hipMalloc(d,n*sizeof(float))); HIP_CHK(hipMemcpy(*d,h,n*sizeof(float),hipMemcpyHostToDevice));
}

static void gemm_bf16xf32(rocblas_handle h,rocblas_operation ta,rocblas_operation tb,
                          int m,int n,int k,const uint16_t*A,int lda,const uint16_t*B,int ldb,
                          float*C,int ldc,float alpha=1.f,float beta=0.f){
    RB_CHK(rocblas_gemm_ex(h,ta,tb,m,n,k,&alpha,
        A,rocblas_datatype_bf16_r,lda,B,rocblas_datatype_bf16_r,ldb,
        &beta,C,rocblas_datatype_f32_r,ldc,C,rocblas_datatype_f32_r,ldc,
        rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
}

// expand fwd: out[M,N] = in[M,K] @ w[N,K]^T
static void expand_bf16(rocblas_handle h,int M,int K,int N,
                        const uint16_t*w,const uint16_t*in,int in_s,float*out){
    gemm_bf16xf32(h,rocblas_operation_transpose,rocblas_operation_none,N,M,K,w,K,in,in_s,out,N);
}
// contract fwd via col-major trick: out[M,H] = in[M,S] @ w[S,H] (storage [S,H])
static void contract_bf16(rocblas_handle h,int M,int S,int H,
                          const uint16_t*w,const uint16_t*in,int in_s,float*out){
    gemm_bf16xf32(h,rocblas_operation_none,rocblas_operation_none,H,M,S,w,H,in,in_s,out,H);
}

struct ErrorStats { float max_abs,mean_abs,max_rel; double cosine; int n; };

static ErrorStats measure(const float*a,const float*b,int n){
    ErrorStats s={0,0,0,0,n}; double dot=0,na=0,nb=0;
    for(int i=0;i<n;i++){
        float d=fabsf(a[i]-b[i]); s.max_abs=fmaxf(s.max_abs,d); s.mean_abs+=d;
        float mv=fmaxf(fabsf(a[i]),fabsf(b[i]));
        if(mv>1e-30f){float rd=d/mv;s.max_rel=fmaxf(s.max_rel,rd);}
        dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i];
    }
    s.mean_abs/=n; s.cosine=dot/(sqrt(na)*sqrt(nb)+1e-30);
    return s;
}

static void report(const char*nm,const ErrorStats&s){
    printf("  %-20s max_abs=%.2e  mean_abs=%.2e  max_rel=%.2e  cos=%.8f\n",
           nm,(double)s.max_abs,(double)s.mean_abs,(double)s.max_rel,s.cosine);
}

// ================================================================
void run_tiny_test(rocblas_handle rh,hipStream_t st){
    const int S=3,H=2,D=4,F=6,M=2;
    printf("=== TINY SENTINEL (S=%d H=%d D=%d F=%d M=%d) ===\n",S,H,D,F,M);

    std::vector<float> gate0(F*D),up0(F*D),down0(D*F);
    for(int i=0;i<F*D;i++){gate0[i]=(i+1)*0.1f; up0[i]=(i+1)*0.07f;}
    for(int i=0;i<D*F;i++)down0[i]=(i+1)*0.05f;
    float hx[M*D]={1,2,3,4,5,6,7,8};

    auto mk=[&](float b,float stp){std::vector<float>v(S*H);for(int i=0;i<S;i++)for(int j=0;j<H;j++)v[i*H+j]=b+i*stp+j;return v;};
    auto g0=mk(0.5f,0.1f),g1=mk(0.3f,0.2f);
    auto u0=mk(0.9f,0.15f),u1=mk(0.7f,0.05f);
    auto d0=mk(1.2f,0.08f),d1=mk(1.0f,0.12f);
    std::vector<float> d0t(H*S),d1t(H*S);
    for(int i=0;i<H;i++)for(int j=0;j<S;j++)d0t[i*S+j]=d0[j*H+i];
    for(int i=0;i<H;i++)for(int j=0;j<S;j++)d1t[i*S+j]=d1[j*H+i];

    // — path A: base + delta separate, bf16 GEMMs —
    uint16_t *d_gate_bf,*d_up_bf,*d_down_bf,*d_x_bf;
    h2d_bf(gate0.data(),F*D,&d_gate_bf); h2d_bf(up0.data(),F*D,&d_up_bf);
    h2d_bf(down0.data(),D*F,&d_down_bf); h2d_bf(hx,M*D,&d_x_bf);

    float *d_gpA,*d_upA,*d_intA,*d_foA,*d_tmp;
    HIP_CHK(hipMalloc(&d_gpA,M*F*sizeof(float))); HIP_CHK(hipMalloc(&d_upA,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_intA,M*F*sizeof(float))); HIP_CHK(hipMalloc(&d_foA,M*D*sizeof(float)));
    HIP_CHK(hipMalloc(&d_tmp,M*S*sizeof(float)));

    HIP_CHK(hipMemset(d_gpA,0,M*F*sizeof(float)));
    expand_bf16(rh,M,D,F,d_gate_bf,d_x_bf,D,d_gpA);
    {  // inject gate deltas
        uint16_t *dg; h2d_bf(g0.data(),S*H,&dg);
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_bf16(rh,M,H,S,dg,d_x_bf,D,d_tmp);
        std::vector<float> hgp(M*F),ht(M*S);
        HIP_CHK(hipMemcpy(hgp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hgp[i*F+j]+=ht[i*S+j];
        HIP_CHK(hipFree(dg));
        h2d_bf(g1.data(),S*H,&dg);
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_bf16(rh,M,H,S,dg,d_x_bf+H,D,d_tmp);
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hgp[i*F+S+j]+=ht[i*S+j];
        HIP_CHK(hipMemcpy(d_gpA,hgp.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(dg));
    }

    HIP_CHK(hipMemset(d_upA,0,M*F*sizeof(float)));
    expand_bf16(rh,M,D,F,d_up_bf,d_x_bf,D,d_upA);
    {  // inject up deltas
        uint16_t *du; h2d_bf(u0.data(),S*H,&du);
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_bf16(rh,M,H,S,du,d_x_bf,D,d_tmp);
        std::vector<float> hup(M*F),ht(M*S);
        HIP_CHK(hipMemcpy(hup.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hup[i*F+j]+=ht[i*S+j];
        HIP_CHK(hipFree(du));
        h2d_bf(u1.data(),S*H,&du);
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_bf16(rh,M,H,S,du,d_x_bf+H,D,d_tmp);
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hup[i*F+S+j]+=ht[i*S+j];
        HIP_CHK(hipMemcpy(d_upA,hup.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(du));
    }

    {   // silu(gate)*up
        std::vector<float> hgp(M*F),hup(M*F),hint(M*F);
        HIP_CHK(hipMemcpy(hgp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(hup.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M*F;i++){float gp=hgp[i],sg=1.f/(1.f+expf(-gp));hint[i]=gp*sg*hup[i];}
        HIP_CHK(hipMemcpy(d_intA,hint.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
    }
    {   // ffn_out = int @ down^T + inject down deltas
        std::vector<float> hint(M*F);
        HIP_CHK(hipMemcpy(hint.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        uint16_t *d_ibf; h2d_bf(hint.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_foA,0,M*D*sizeof(float)));
        gemm_bf16xf32(rh,rocblas_operation_transpose,rocblas_operation_none,
                      D,M,F,d_down_bf,F,d_ibf,F,d_foA,D);
        HIP_CHK(hipFree(d_ibf));
        // inject down deltas
        uint16_t *dd; h2d_bf(d0.data(),S*H,&dd);
        h2d_bf(hint.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_tmp,0,M*H*sizeof(float)));
        contract_bf16(rh,M,S,H,dd,d_ibf,F,d_tmp);
        std::vector<float> hfo(M*D),ht(M*H);
        HIP_CHK(hipMemcpy(hfo.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*H*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<H;j++)hfo[i*D+j]+=ht[i*H+j];
        HIP_CHK(hipFree(dd)); HIP_CHK(hipFree(d_ibf));
        h2d_bf(d1.data(),S*H,&dd);
        h2d_bf(hint.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_tmp,0,M*H*sizeof(float)));
        contract_bf16(rh,M,S,H,dd,d_ibf+S,F,d_tmp);
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*H*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<H;j++)hfo[i*D+H+j]+=ht[i*H+j];
        HIP_CHK(hipMemcpy(d_foA,hfo.data(),M*D*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(dd)); HIP_CHK(hipFree(d_ibf));
    }

    // — path B: bake bf16, one GEMM —
    std::vector<float> gb(gate0),ub(up0),db(down0);
    for(int i=0;i<S;i++)for(int j=0;j<H;j++)gb[i*D+j]+=g0[i*H+j];
    for(int i=0;i<S;i++)for(int j=0;j<H;j++)gb[(S+i)*D+(H+j)]+=g1[i*H+j];
    for(int i=0;i<S;i++)for(int j=0;j<H;j++)ub[i*D+j]+=u0[i*H+j];
    for(int i=0;i<S;i++)for(int j=0;j<H;j++)ub[(S+i)*D+(H+j)]+=u1[i*H+j];
    for(int i=0;i<H;i++)for(int j=0;j<S;j++)db[i*F+j]+=d0t[i*S+j];
    for(int i=0;i<H;i++)for(int j=0;j<S;j++)db[(H+i)*F+(S+j)]+=d1t[i*S+j];

    uint16_t *d_gb,*d_ub,*d_db;
    h2d_bf(gb.data(),F*D,&d_gb); h2d_bf(ub.data(),F*D,&d_ub); h2d_bf(db.data(),D*F,&d_db);

    float *d_gpB,*d_upB,*d_intB,*d_foB;
    HIP_CHK(hipMalloc(&d_gpB,M*F*sizeof(float))); HIP_CHK(hipMalloc(&d_upB,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_intB,M*F*sizeof(float))); HIP_CHK(hipMalloc(&d_foB,M*D*sizeof(float)));

    HIP_CHK(hipMemset(d_gpB,0,M*F*sizeof(float)));
    expand_bf16(rh,M,D,F,d_gb,d_x_bf,D,d_gpB);
    HIP_CHK(hipMemset(d_upB,0,M*F*sizeof(float)));
    expand_bf16(rh,M,D,F,d_ub,d_x_bf,D,d_upB);
    {   std::vector<float> hgp(M*F),hup(M*F),hint(M*F);
        HIP_CHK(hipMemcpy(hgp.data(),d_gpB,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(hup.data(),d_upB,M*F*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M*F;i++){float gp=hgp[i],sg=1.f/(1.f+expf(-gp));hint[i]=gp*sg*hup[i];}
        HIP_CHK(hipMemcpy(d_intB,hint.data(),M*F*sizeof(float),hipMemcpyHostToDevice));}
    {   std::vector<float> hint(M*F);
        HIP_CHK(hipMemcpy(hint.data(),d_intB,M*F*sizeof(float),hipMemcpyDeviceToHost));
        uint16_t *d_ibf; h2d_bf(hint.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_foB,0,M*D*sizeof(float)));
        gemm_bf16xf32(rh,rocblas_operation_transpose,rocblas_operation_none,
                      D,M,F,d_db,F,d_ibf,F,d_foB,D);
        HIP_CHK(hipFree(d_ibf));}

    HIP_CHK(hipStreamSynchronize(st));

    // readback and measure
    std::vector<float> va_gp(M*F),vb_gp(M*F),va_up(M*F),vb_up(M*F),va_int(M*F),vb_int(M*F),va_fo(M*D),vb_fo(M*D);
    HIP_CHK(hipMemcpy(va_gp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vb_gp.data(),d_gpB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(va_up.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vb_up.data(),d_upB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(va_int.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vb_int.data(),d_intB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(va_fo.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vb_fo.data(),d_foB,M*D*sizeof(float),hipMemcpyDeviceToHost));

    report("gate_pre",   measure(va_gp.data(),vb_gp.data(),M*F));
    report("up_pre",     measure(va_up.data(),vb_up.data(),M*F));
    report("intermediate",measure(va_int.data(),vb_int.data(),M*F));
    report("ffn_out",    measure(va_fo.data(),vb_fo.data(),M*D));

    // per-projection gate error
    for(int d=0;d<2;d++){
        std::vector<float> ga(M*S),gb(M*S);
        for(int i=0;i<M;i++)for(int j=0;j<S;j++){ga[i*S+j]=va_gp[i*F+d*S+j];gb[i*S+j]=vb_gp[i*F+d*S+j];}
        char buf[32];snprintf(buf,sizeof(buf),"gate D%d",d);
        report(buf,measure(ga.data(),gb.data(),M*S));
    }
    for(int d=0;d<2;d++){
        std::vector<float> fa(M*H),fb(M*H);
        for(int i=0;i<M;i++)for(int j=0;j<H;j++){fa[i*H+j]=va_fo[i*D+d*H+j];fb[i*H+j]=vb_fo[i*D+d*H+j];}
        char buf[32];snprintf(buf,sizeof(buf),"down D%d",d);
        report(buf,measure(fa.data(),fb.data(),M*H));
    }

    printf("  (bf16 rounding expected: ~1/256 ≈ 3.9e-3 relative per quantization)\n\n");

    // cleanup
    HIP_CHK(hipFree(d_gate_bf));HIP_CHK(hipFree(d_up_bf));HIP_CHK(hipFree(d_down_bf));HIP_CHK(hipFree(d_x_bf));
    HIP_CHK(hipFree(d_gpA));HIP_CHK(hipFree(d_upA));HIP_CHK(hipFree(d_intA));HIP_CHK(hipFree(d_foA));HIP_CHK(hipFree(d_tmp));
    HIP_CHK(hipFree(d_gpB));HIP_CHK(hipFree(d_upB));HIP_CHK(hipFree(d_intB));HIP_CHK(hipFree(d_foB));
    HIP_CHK(hipFree(d_gb));HIP_CHK(hipFree(d_ub));HIP_CHK(hipFree(d_db));
}

// ================================================================
void run_layer_test(rocblas_handle rh,hipStream_t st){
    // Real layer shape (SmolLM3-3B, 4 domains): S=2752, H=512, D=2048, F=11008
    // Use M=32 for speed
    const int S=2752,H=512,D=2048,F=11008,M=32;
    printf("=== REAL LAYER (S=%d H=%d D=%d F=%d M=%d) ===\n",S,H,D,F,M);

    std::vector<float> gate0(F*D),up0(F*D),down0(D*F),hx(M*D);
    // deterministic fill
    for(int i=0;i<F*D;i++){gate0[i]=((i*7919+104729)%10007)*0.001f-5.f; up0[i]=((i*6271+224737)%10007)*0.001f-5.f;}
    for(int i=0;i<D*F;i++)down0[i]=((i*4903+346627)%10007)*0.001f-5.f;
    for(int i=0;i<M*D;i++)hx[i]=((i*919+591623)%10007)*0.001f-5.f;

    // delta: small sentinel within random weight range
    std::vector<float> g0(S*H),g1(S*H),u0(S*H),u1(S*H),d0(S*H),d1(S*H);
    for(int i=0;i<S*H;i++){g0[i]=((i*241+1000003)%10007)*0.2e-4f; g1[i]=((i*503+2000033)%10007)*0.2e-4f;
                            u0[i]=((i*751+3000071)%10007)*0.2e-4f; u1[i]=((i*997+4000077)%10007)*0.2e-4f;
                            d0[i]=((i*1231+5000099)%10007)*0.2e-4f; d1[i]=((i*1481+6000037)%10007)*0.2e-4f;}
    std::vector<float> d0t(H*S),d1t(H*S);
    for(int i=0;i<H;i++)for(int j=0;j<S;j++){d0t[i*S+j]=d0[j*H+i]; d1t[i*S+j]=d1[j*H+i];}

    uint16_t *d_gate_bf,*d_up_bf,*d_down_bf,*d_x_bf;
    h2d_bf(gate0.data(),F*D,&d_gate_bf); h2d_bf(up0.data(),F*D,&d_up_bf);
    h2d_bf(down0.data(),D*F,&d_down_bf); h2d_bf(hx.data(),M*D,&d_x_bf);

    float *d_gpA,*d_upA,*d_intA,*d_foA,*d_tmpS,*d_tmpH;
    HIP_CHK(hipMalloc(&d_gpA,M*F*sizeof(float))); HIP_CHK(hipMalloc(&d_upA,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_intA,M*F*sizeof(float))); HIP_CHK(hipMalloc(&d_foA,M*D*sizeof(float)));
    HIP_CHK(hipMalloc(&d_tmpS,M*S*sizeof(float))); HIP_CHK(hipMalloc(&d_tmpH,M*H*sizeof(float)));

    // — PATH A —
    HIP_CHK(hipMemset(d_gpA,0,M*F*sizeof(float)));
    expand_bf16(rh,M,D,F,d_gate_bf,d_x_bf,D,d_gpA);
    {uint16_t *dg; h2d_bf(g0.data(),S*H,&dg);HIP_CHK(hipMemset(d_tmpS,0,M*S*sizeof(float)));
        expand_bf16(rh,M,H,S,dg,d_x_bf,D,d_tmpS);
        std::vector<float> hgp(M*F),ht(M*S);HIP_CHK(hipMemcpy(hgp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmpS,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hgp[i*F+j]+=ht[i*S+j];
        HIP_CHK(hipFree(dg));h2d_bf(g1.data(),S*H,&dg);
        HIP_CHK(hipMemset(d_tmpS,0,M*S*sizeof(float)));expand_bf16(rh,M,H,S,dg,d_x_bf+H,D,d_tmpS);
        HIP_CHK(hipMemcpy(ht.data(),d_tmpS,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hgp[i*F+S+j]+=ht[i*S+j];
        HIP_CHK(hipMemcpy(d_gpA,hgp.data(),M*F*sizeof(float),hipMemcpyHostToDevice));HIP_CHK(hipFree(dg));}

    HIP_CHK(hipMemset(d_upA,0,M*F*sizeof(float)));
    expand_bf16(rh,M,D,F,d_up_bf,d_x_bf,D,d_upA);
    {uint16_t *du; h2d_bf(u0.data(),S*H,&du);HIP_CHK(hipMemset(d_tmpS,0,M*S*sizeof(float)));
        expand_bf16(rh,M,H,S,du,d_x_bf,D,d_tmpS);
        std::vector<float> hup(M*F),ht(M*S);HIP_CHK(hipMemcpy(hup.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmpS,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hup[i*F+j]+=ht[i*S+j];
        HIP_CHK(hipFree(du));h2d_bf(u1.data(),S*H,&du);
        HIP_CHK(hipMemset(d_tmpS,0,M*S*sizeof(float)));expand_bf16(rh,M,H,S,du,d_x_bf+H,D,d_tmpS);
        HIP_CHK(hipMemcpy(ht.data(),d_tmpS,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hup[i*F+S+j]+=ht[i*S+j];
        HIP_CHK(hipMemcpy(d_upA,hup.data(),M*F*sizeof(float),hipMemcpyHostToDevice));HIP_CHK(hipFree(du));}

    {std::vector<float> hgp(M*F),hup(M*F),hint(M*F);
        HIP_CHK(hipMemcpy(hgp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(hup.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M*F;i++){float gp=hgp[i],sg=1.f/(1.f+expf(-gp));hint[i]=gp*sg*hup[i];}
        HIP_CHK(hipMemcpy(d_intA,hint.data(),M*F*sizeof(float),hipMemcpyHostToDevice));}

    {std::vector<float> hint(M*F);HIP_CHK(hipMemcpy(hint.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        uint16_t *d_ibf;h2d_bf(hint.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_foA,0,M*D*sizeof(float)));
        gemm_bf16xf32(rh,rocblas_operation_transpose,rocblas_operation_none,D,M,F,d_down_bf,F,d_ibf,F,d_foA,D);
        HIP_CHK(hipFree(d_ibf));
        uint16_t *dd;h2d_bf(d0.data(),S*H,&dd);h2d_bf(hint.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_tmpH,0,M*H*sizeof(float)));contract_bf16(rh,M,S,H,dd,d_ibf,F,d_tmpH);
        std::vector<float> hfo(M*D),ht(M*H);HIP_CHK(hipMemcpy(hfo.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmpH,M*H*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<H;j++)hfo[i*D+j]+=ht[i*H+j];
        HIP_CHK(hipFree(dd));HIP_CHK(hipFree(d_ibf));
        h2d_bf(d1.data(),S*H,&dd);h2d_bf(hint.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_tmpH,0,M*H*sizeof(float)));contract_bf16(rh,M,S,H,dd,d_ibf+S,F,d_tmpH);
        HIP_CHK(hipMemcpy(ht.data(),d_tmpH,M*H*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<H;j++)hfo[i*D+H+j]+=ht[i*H+j];
        HIP_CHK(hipMemcpy(d_foA,hfo.data(),M*D*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(dd));HIP_CHK(hipFree(d_ibf));}

    // — PATH B —
    std::vector<float> gb(gate0),ub(up0),db(down0);
    for(int i=0;i<S;i++)for(int j=0;j<H;j++){gb[i*D+j]+=g0[i*H+j];gb[(S+i)*D+(H+j)]+=g1[i*H+j];}
    for(int i=0;i<S;i++)for(int j=0;j<H;j++){ub[i*D+j]+=u0[i*H+j];ub[(S+i)*D+(H+j)]+=u1[i*H+j];}
    for(int i=0;i<H;i++)for(int j=0;j<S;j++){db[i*F+j]+=d0t[i*S+j];db[(H+i)*F+(S+j)]+=d1t[i*S+j];}

    uint16_t *d_gb,*d_ub,*d_db;
    h2d_bf(gb.data(),F*D,&d_gb);h2d_bf(ub.data(),F*D,&d_ub);h2d_bf(db.data(),D*F,&d_db);
    float *d_gpB,*d_upB,*d_intB,*d_foB;
    HIP_CHK(hipMalloc(&d_gpB,M*F*sizeof(float)));HIP_CHK(hipMalloc(&d_upB,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_intB,M*F*sizeof(float)));HIP_CHK(hipMalloc(&d_foB,M*D*sizeof(float)));

    HIP_CHK(hipMemset(d_gpB,0,M*F*sizeof(float)));expand_bf16(rh,M,D,F,d_gb,d_x_bf,D,d_gpB);
    HIP_CHK(hipMemset(d_upB,0,M*F*sizeof(float)));expand_bf16(rh,M,D,F,d_ub,d_x_bf,D,d_upB);
    {std::vector<float> hgp(M*F),hup(M*F),hint(M*F);
        HIP_CHK(hipMemcpy(hgp.data(),d_gpB,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(hup.data(),d_upB,M*F*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M*F;i++){float gp=hgp[i],sg=1.f/(1.f+expf(-gp));hint[i]=gp*sg*hup[i];}
        HIP_CHK(hipMemcpy(d_intB,hint.data(),M*F*sizeof(float),hipMemcpyHostToDevice));}
    {std::vector<float> hint(M*F);HIP_CHK(hipMemcpy(hint.data(),d_intB,M*F*sizeof(float),hipMemcpyDeviceToHost));
        uint16_t *d_ibf;h2d_bf(hint.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_foB,0,M*D*sizeof(float)));
        gemm_bf16xf32(rh,rocblas_operation_transpose,rocblas_operation_none,D,M,F,d_db,F,d_ibf,F,d_foB,D);
        HIP_CHK(hipFree(d_ibf));}

    HIP_CHK(hipStreamSynchronize(st));

    std::vector<float> va_gp(M*F),vb_gp(M*F),va_up(M*F),vb_up(M*F),va_int(M*F),vb_int(M*F),va_fo(M*D),vb_fo(M*D);
    HIP_CHK(hipMemcpy(va_gp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vb_gp.data(),d_gpB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(va_up.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vb_up.data(),d_upB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(va_int.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vb_int.data(),d_intB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(va_fo.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vb_fo.data(),d_foB,M*D*sizeof(float),hipMemcpyDeviceToHost));

    report("gate_pre",   measure(va_gp.data(),vb_gp.data(),M*F));
    report("up_pre",     measure(va_up.data(),vb_up.data(),M*F));
    report("intermediate",measure(va_int.data(),vb_int.data(),M*F));
    report("ffn_out",    measure(va_fo.data(),vb_fo.data(),M*D));
    for(int d=0;d<2;d++){
        std::vector<float> ga(M*S),gb(M*S);
        for(int i=0;i<M;i++)for(int j=0;j<S;j++){ga[i*S+j]=va_gp[i*F+d*S+j];gb[i*S+j]=vb_gp[i*F+d*S+j];}
        char buf[32];snprintf(buf,sizeof(buf),"gate D%d",d);report(buf,measure(ga.data(),gb.data(),M*S));}
    for(int d=0;d<2;d++){
        std::vector<float> fa(M*H),fb(M*H);
        for(int i=0;i<M;i++)for(int j=0;j<H;j++){fa[i*H+j]=va_fo[i*D+d*H+j];fb[i*H+j]=vb_fo[i*D+d*H+j];}
        char buf[32];snprintf(buf,sizeof(buf),"down D%d",d);report(buf,measure(fa.data(),fb.data(),M*H));}

    printf("  (bf16 rounding expected: ~1/256 ≈ 3.9e-3 relative per quantization)\n\n");

    HIP_CHK(hipFree(d_gate_bf));HIP_CHK(hipFree(d_up_bf));HIP_CHK(hipFree(d_down_bf));HIP_CHK(hipFree(d_x_bf));
    HIP_CHK(hipFree(d_gpA));HIP_CHK(hipFree(d_upA));HIP_CHK(hipFree(d_intA));HIP_CHK(hipFree(d_foA));
    HIP_CHK(hipFree(d_tmpS));HIP_CHK(hipFree(d_tmpH));
    HIP_CHK(hipFree(d_gpB));HIP_CHK(hipFree(d_upB));HIP_CHK(hipFree(d_intB));HIP_CHK(hipFree(d_foB));
    HIP_CHK(hipFree(d_gb));HIP_CHK(hipFree(d_ub));HIP_CHK(hipFree(d_db));
}

int main(){
    hipStream_t st;HIP_CHK(hipStreamCreate(&st));
    rocblas_handle rh;RB_CHK(rocblas_create_handle(&rh));RB_CHK(rocblas_set_stream(rh,st));

    run_tiny_test(rh,st);
    run_layer_test(rh,st);

    RB_CHK(rocblas_destroy_handle(rh));HIP_CHK(hipStreamDestroy(st));
    printf("Test 7: DONE\n");
    return 0;
}
