// test6_stack_vs_bake.cpp
// Compares explicit graft injection (Path A) vs baked-weight FFN (Path B)
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

static float bf2f(uint16_t v){uint32_t b=(uint32_t)v<<16;float o;memcpy(&o,&b,4);return o;}
static uint16_t f2bf(float f){uint32_t b;memcpy(&b,&f,4);b+=((b>>16)&1)+0x7FFF;return(uint16_t)(b>>16);}

static void gemm(rocblas_handle h,rocblas_operation ta,rocblas_operation tb,
                 int m,int n,int k,const void*A,int lda,const void*B,int ldb,
                 float*C,int ldc,float alpha=1.f,float beta=0.f){
    RB_CHK(rocblas_gemm_ex(h,ta,tb,m,n,k,&alpha,
        A,rocblas_datatype_bf16_r,lda,B,rocblas_datatype_bf16_r,ldb,
        &beta,C,rocblas_datatype_f32_r,ldc,C,rocblas_datatype_f32_r,ldc,
        rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
}

static void expand_fwd(rocblas_handle h,int M,int K,int N,
                       const void*w,const void*in,int in_stride,float*out){
    gemm(h,rocblas_operation_transpose,rocblas_operation_none,N,M,K,w,K,in,in_stride,out,N);
}
static void contract_fwd(rocblas_handle h,int M,int S,int H,
                         const void*w,const void*in,int in_stride,float*out){
    gemm(h,rocblas_operation_none,rocblas_operation_none,H,M,S,w,H,in,in_stride,out,H);
}

// helper: host f32 -> device bf16 -> device pointer
static void hf2dbf(hipStream_t s,const float* h,int n,uint16_t** d){
    std::vector<uint16_t> hv(n);
    for(int i=0;i<n;i++)hv[i]=f2bf(h[i]);
    HIP_CHK(hipMalloc(d,n*sizeof(uint16_t)));
    HIP_CHK(hipMemcpy(*d,hv.data(),n*sizeof(uint16_t),hipMemcpyHostToDevice));
}

int main(){
    const int S=3,H=2,D=4,F=6,M=2;
    int fail=0;

    hipStream_t st; HIP_CHK(hipStreamCreate(&st));
    rocblas_handle rh; RB_CHK(rocblas_create_handle(&rh));
    RB_CHK(rocblas_set_stream(rh,st));

    // ---- host data ----
    std::vector<float> h_gate(F*D), h_up(F*D), h_down(D*F);
    for(int i=0;i<F*D;i++){h_gate[i]=(i+1)*0.1f; h_up[i]=(i+1)*0.07f;}
    for(int i=0;i<D*F;i++)h_down[i]=(i+1)*0.05f;

    float h_x[M*D]={1,2,3,4,5,6,7,8};

    auto mk=[&](float base,float step){
        std::vector<float> v(S*H);
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)v[i*H+j]=base+i*step+j;
        return v;
    };
    auto g0=mk(0.5f,0.1f),g1=mk(0.3f,0.2f);
    auto u0=mk(0.9f,0.15f),u1=mk(0.7f,0.05f);
    auto d0hip=mk(1.2f,0.08f),d1hip=mk(1.0f,0.12f);

    std::vector<float> d0tphs(H*S),d1tphs(H*S);
    for(int i=0;i<H;i++)for(int j=0;j<S;j++)d0tphs[i*S+j]=d0hip[j*H+i];
    for(int i=0;i<H;i++)for(int j=0;j<S;j++)d1tphs[i*S+j]=d1hip[j*H+i];

    // ---- allocate device buffers ----
    uint16_t *d_gate,*d_up,*d_down,*d_x;
    float *d_gpA,*d_upA,*d_intA,*d_foA;
    float *d_gpB,*d_upB,*d_intB,*d_foB;

    hf2dbf(st,h_gate.data(),F*D,&d_gate);
    hf2dbf(st,h_up.data(),  F*D,&d_up);
    hf2dbf(st,h_down.data(),D*F,&d_down);
    hf2dbf(st,h_x,M*D,&d_x);

    HIP_CHK(hipMalloc(&d_gpA, M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_upA, M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_intA,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_foA, M*D*sizeof(float)));
    HIP_CHK(hipMalloc(&d_gpB, M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_upB, M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_intB,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_foB, M*D*sizeof(float)));

    float *d_tmp; HIP_CHK(hipMalloc(&d_tmp,M*S*sizeof(float)));

    // ======================================================
    // PATH A: explicit injection (mimics forward_layer)
    // ======================================================

    // gate_pre = X @ gate^T
    HIP_CHK(hipMemset(d_gpA,0,M*F*sizeof(float)));
    gemm(rh,rocblas_operation_transpose,rocblas_operation_none,F,M,D,
         d_gate,D,d_x,D,d_gpA,F);

    // Inject gate deltas
    {
        uint16_t *dg0,*dg1;
        hf2dbf(st,g0.data(),S*H,&dg0);
        hf2dbf(st,g1.data(),S*H,&dg1);
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_fwd(rh,M,H,S,dg0,d_x,D,d_tmp);
        // gate_pre[:,0:S] += d_tmp  (on host after download, simpler)
        std::vector<float> h_gp(M*F),h_tmp(M*S);
        HIP_CHK(hipMemcpy(h_gp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(h_tmp.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)h_gp[i*F+j]+=h_tmp[i*S+j];
        // second domain
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_fwd(rh,M,H,S,dg1,d_x+H,D,d_tmp);
        HIP_CHK(hipMemcpy(h_tmp.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)h_gp[i*F+S+j]+=h_tmp[i*S+j];
        HIP_CHK(hipMemcpy(d_gpA,h_gp.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(dg0));HIP_CHK(hipFree(dg1));
    }

    // up = X @ up^T
    HIP_CHK(hipMemset(d_upA,0,M*F*sizeof(float)));
    gemm(rh,rocblas_operation_transpose,rocblas_operation_none,F,M,D,
         d_up,D,d_x,D,d_upA,F);

    // Inject up deltas
    {
        uint16_t *du0,*du1;
        hf2dbf(st,u0.data(),S*H,&du0);
        hf2dbf(st,u1.data(),S*H,&du1);
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_fwd(rh,M,H,S,du0,d_x,D,d_tmp);
        std::vector<float> h_up(M*F),h_tmp(M*S);
        HIP_CHK(hipMemcpy(h_up.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(h_tmp.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)h_up[i*F+j]+=h_tmp[i*S+j];
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_fwd(rh,M,H,S,du1,d_x+H,D,d_tmp);
        HIP_CHK(hipMemcpy(h_tmp.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)h_up[i*F+S+j]+=h_tmp[i*S+j];
        HIP_CHK(hipMemcpy(d_upA,h_up.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(du0));HIP_CHK(hipFree(du1));
    }

    // intermediate = silu(gate_pre) * up (on host for simplicity)
    {
        std::vector<float> h_gp(M*F),h_up(M*F),h_int(M*F);
        HIP_CHK(hipMemcpy(h_gp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(h_up.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M*F;i++){
            float gp=h_gp[i],sg=1.f/(1.f+expf(-gp));
            h_int[i]=gp*sg*h_up[i];
        }
        HIP_CHK(hipMemcpy(d_intA,h_int.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
    }

    // ffn_out = intermediate @ down^T
    {
        std::vector<float> h_int(M*F);
        HIP_CHK(hipMemcpy(h_int.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        uint16_t *d_ibf; hf2dbf(st,h_int.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_foA,0,M*D*sizeof(float)));
        gemm(rh,rocblas_operation_transpose,rocblas_operation_none,D,M,F,
             d_down,F,d_ibf,F,d_foA,D);
        HIP_CHK(hipFree(d_ibf));
    }

    // Inject down deltas: ffn_out[:,h_slice] += intermediate_i @ down_tphs^T
    {
        uint16_t *dd0,*dd1;
        hf2dbf(st,d0hip.data(),S*H,&dd0);
        hf2dbf(st,d1hip.data(),S*H,&dd1);
        uint16_t *d_ibf;
        {
            std::vector<float> h_int(M*F);
            HIP_CHK(hipMemcpy(h_int.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
            hf2dbf(st,h_int.data(),M*F,&d_ibf);
        }
        // domain 0
        HIP_CHK(hipMemset(d_tmp,0,M*H*sizeof(float)));
        contract_fwd(rh,M,S,H,dd0,d_ibf,F,d_tmp);
        {
            std::vector<float> h_fo(M*D),h_tmp(M*H);
            HIP_CHK(hipMemcpy(h_fo.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
            HIP_CHK(hipMemcpy(h_tmp.data(),d_tmp,M*H*sizeof(float),hipMemcpyDeviceToHost));
            for(int i=0;i<M;i++)for(int j=0;j<H;j++)h_fo[i*D+j]+=h_tmp[i*H+j];
            HIP_CHK(hipMemcpy(d_foA,h_fo.data(),M*D*sizeof(float),hipMemcpyHostToDevice));
        }
        // domain 1
        HIP_CHK(hipMemset(d_tmp,0,M*H*sizeof(float)));
        contract_fwd(rh,M,S,H,dd1,d_ibf+S,F,d_tmp);
        {
            std::vector<float> h_fo(M*D),h_tmp(M*H);
            HIP_CHK(hipMemcpy(h_fo.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
            HIP_CHK(hipMemcpy(h_tmp.data(),d_tmp,M*H*sizeof(float),hipMemcpyDeviceToHost));
            for(int i=0;i<M;i++)for(int j=0;j<H;j++)h_fo[i*D+H+j]+=h_tmp[i*H+j];
            HIP_CHK(hipMemcpy(d_foA,h_fo.data(),M*D*sizeof(float),hipMemcpyHostToDevice));
        }
        HIP_CHK(hipFree(dd0));HIP_CHK(hipFree(dd1));HIP_CHK(hipFree(d_ibf));
    }

    // ======================================================
    // PATH B: bake + single forward
    // ======================================================
    {
        std::vector<float> gb(h_gate), ub(h_up), db(h_down);
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)gb[i*D+j]+=g0[i*H+j];
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)gb[(S+i)*D+(H+j)]+=g1[i*H+j];
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)ub[i*D+j]+=u0[i*H+j];
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)ub[(S+i)*D+(H+j)]+=u1[i*H+j];
        for(int i=0;i<H;i++)for(int j=0;j<S;j++)db[i*F+j]+=d0tphs[i*S+j];
        for(int i=0;i<H;i++)for(int j=0;j<S;j++)db[(H+i)*F+(S+j)]+=d1tphs[i*S+j];

        uint16_t *d_gb,*d_ub,*d_db;
        hf2dbf(st,gb.data(),F*D,&d_gb);
        hf2dbf(st,ub.data(),F*D,&d_ub);
        hf2dbf(st,db.data(),D*F,&d_db);

        HIP_CHK(hipMemset(d_gpB,0,M*F*sizeof(float)));
        gemm(rh,rocblas_operation_transpose,rocblas_operation_none,F,M,D,
             d_gb,D,d_x,D,d_gpB,F);

        HIP_CHK(hipMemset(d_upB,0,M*F*sizeof(float)));
        gemm(rh,rocblas_operation_transpose,rocblas_operation_none,F,M,D,
             d_ub,D,d_x,D,d_upB,F);

        {
            std::vector<float> h_gp(M*F),h_up(M*F),h_int(M*F);
            HIP_CHK(hipMemcpy(h_gp.data(),d_gpB,M*F*sizeof(float),hipMemcpyDeviceToHost));
            HIP_CHK(hipMemcpy(h_up.data(),d_upB,M*F*sizeof(float),hipMemcpyDeviceToHost));
            for(int i=0;i<M*F;i++){
                float gp=h_gp[i],sg=1.f/(1.f+expf(-gp));
                h_int[i]=gp*sg*h_up[i];
            }
            HIP_CHK(hipMemcpy(d_intB,h_int.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
        }

        {
            std::vector<float> h_int(M*F);
            HIP_CHK(hipMemcpy(h_int.data(),d_intB,M*F*sizeof(float),hipMemcpyDeviceToHost));
            uint16_t *d_ibf; hf2dbf(st,h_int.data(),M*F,&d_ibf);
            HIP_CHK(hipMemset(d_foB,0,M*D*sizeof(float)));
            gemm(rh,rocblas_operation_transpose,rocblas_operation_none,D,M,F,
                 d_db,F,d_ibf,F,d_foB,D);
            HIP_CHK(hipFree(d_ibf));
        }

        HIP_CHK(hipFree(d_gb));HIP_CHK(hipFree(d_ub));HIP_CHK(hipFree(d_db));
    }

    HIP_CHK(hipStreamSynchronize(st));

    // ======================================================
    // Compare
    // ======================================================
    std::vector<float> h_gpA(M*F),h_gpB(M*F),h_upA(M*F),h_upB(M*F);
    std::vector<float> h_intA(M*F),h_intB(M*F),h_foA(M*D),h_foB(M*D);
    HIP_CHK(hipMemcpy(h_gpA.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(h_gpB.data(),d_gpB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(h_upA.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(h_upB.data(),d_upB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(h_intA.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(h_intB.data(),d_intB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(h_foA.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(h_foB.data(),d_foB,M*D*sizeof(float),hipMemcpyDeviceToHost));

    auto cmp=[&](const char*name,const float*a,const float*b,int n){
        float md=0;int mi=0;
        for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>md){md=d;mi=i;}}
        bool ok=md<1e-4f;
        printf("%-24s max|d|=%.2e @%d  A=%.6f B=%.6f  %s\n",
               name,(double)md,mi,(double)a[mi],(double)b[mi],ok?"OK":"FAIL");
        return ok;
    };

    printf("=== Path A (explicit) vs Path B (baked) ===\n");
    fail+=!cmp("gate_pre",     h_gpA.data(),h_gpB.data(),M*F);
    fail+=!cmp("up_pre",       h_upA.data(),h_upB.data(),M*F);
    fail+=!cmp("intermediate", h_intA.data(),h_intB.data(),M*F);
    fail+=!cmp("ffn_out",      h_foA.data(),h_foB.data(),M*D);

    // Slice-level check
    printf("\n=== Down h-slice diffs ===\n");
    for(int d=0;d<2;d++){
        int off=d*H;float md=0;
        for(int i=0;i<M;i++)for(int j=0;j<H;j++){
            float dd=fabsf(h_foA[i*D+off+j]-h_foB[i*D+off+j]);
            if(dd>md)md=dd;
        }
        printf("  domain%d ffn_out h-slice max|d|=%.2e %s\n",d,(double)md,md<1e-4f?"OK":"FAIL");
        if(md>=1e-4f)fail++;
    }
    printf("\n=== Gate i-slice diffs ===\n");
    for(int d=0;d<2;d++){
        int off=d*S;float md=0;
        for(int i=0;i<M;i++)for(int j=0;j<S;j++){
            float dd=fabsf(h_gpA[i*F+off+j]-h_gpB[i*F+off+j]);
            if(dd>md)md=dd;
        }
        printf("  domain%d gate_pre i-slice max|d|=%.2e %s\n",d,(double)md,md<1e-4f?"OK":"FAIL");
        if(md>=1e-4f)fail++;
    }

    // Print full ffn_out
    printf("\n=== ffn_out A ===\n");
    for(int i=0;i<M;i++){printf("  t%d: ",i);for(int j=0;j<D;j++)printf("%.4f ",(double)h_foA[i*D+j]);printf("\n");}
    printf("=== ffn_out B ===\n");
    for(int i=0;i<M;i++){printf("  t%d: ",i);for(int j=0;j<D;j++)printf("%.4f ",(double)h_foB[i*D+j]);printf("\n");}

    // Cleanup
    HIP_CHK(hipFree(d_gate));HIP_CHK(hipFree(d_up));HIP_CHK(hipFree(d_down));HIP_CHK(hipFree(d_x));
    HIP_CHK(hipFree(d_gpA));HIP_CHK(hipFree(d_upA));HIP_CHK(hipFree(d_intA));HIP_CHK(hipFree(d_foA));
    HIP_CHK(hipFree(d_gpB));HIP_CHK(hipFree(d_upB));HIP_CHK(hipFree(d_intB));HIP_CHK(hipFree(d_foB));
    HIP_CHK(hipFree(d_tmp));
    RB_CHK(rocblas_destroy_handle(rh));
    HIP_CHK(hipStreamDestroy(st));

    printf("\nTest 6: %s\n",fail==0?"PASS":"FAIL");
    return fail;
}
