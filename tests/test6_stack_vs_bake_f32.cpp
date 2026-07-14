// test6_stack_vs_bake_f32.cpp — f32 GEMM for pure math validation
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

static void gemm(rocblas_handle h,rocblas_operation ta,rocblas_operation tb,
                 int m,int n,int k,const float*A,int lda,const float*B,int ldb,
                 float*C,int ldc,float alpha=1.f,float beta=0.f){
    RB_CHK(rocblas_gemm_ex(h,ta,tb,m,n,k,&alpha,
        A,rocblas_datatype_f32_r,lda,B,rocblas_datatype_f32_r,ldb,
        &beta,C,rocblas_datatype_f32_r,ldc,C,rocblas_datatype_f32_r,ldc,
        rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
}
static void expand_fwd(rocblas_handle h,int M,int K,int N,
                       const float*w,const float*in,int in_stride,float*out){
    gemm(h,rocblas_operation_transpose,rocblas_operation_none,N,M,K,w,K,in,in_stride,out,N);
}
static void contract_fwd(rocblas_handle h,int M,int S,int H,
                         const float*w,const float*in,int in_stride,float*out){
    gemm(h,rocblas_operation_none,rocblas_operation_none,H,M,S,w,H,in,in_stride,out,H);
}
static void h2d(const float*h,int n,float**d){
    HIP_CHK(hipMalloc(d,n*sizeof(float)));
    HIP_CHK(hipMemcpy(*d,h,n*sizeof(float),hipMemcpyHostToDevice));
}

int main(){
    const int S=3,H=2,D=4,F=6,M=2;
    int fail=0;
    hipStream_t st; HIP_CHK(hipStreamCreate(&st));
    rocblas_handle rh; RB_CHK(rocblas_create_handle(&rh));
    RB_CHK(rocblas_set_stream(rh,st));

    // host data
    std::vector<float> gate0(F*D),up0(F*D),down0(D*F);
    for(int i=0;i<F*D;i++){gate0[i]=(i+1)*0.1f; up0[i]=(i+1)*0.07f;}
    for(int i=0;i<D*F;i++)down0[i]=(i+1)*0.05f;
    float hx[M*D]={1,2,3,4,5,6,7,8};

    auto mk=[&](float b,float stp){
        std::vector<float> v(S*H);
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)v[i*H+j]=b+i*stp+j;
        return v;
    };
    auto g0=mk(0.5f,0.1f),g1=mk(0.3f,0.2f);
    auto u0=mk(0.9f,0.15f),u1=mk(0.7f,0.05f);
    auto d0=mk(1.2f,0.08f),d1=mk(1.0f,0.12f);
    std::vector<float> d0t(H*S),d1t(H*S);
    for(int i=0;i<H;i++)for(int j=0;j<S;j++)d0t[i*S+j]=d0[j*H+i];
    for(int i=0;i<H;i++)for(int j=0;j<S;j++)d1t[i*S+j]=d1[j*H+i];

    // device pointers
    float *d_gate,*d_up,*d_down,*d_x,*d_tmp;
    h2d(gate0.data(),F*D,&d_gate); h2d(up0.data(),F*D,&d_up);
    h2d(down0.data(),D*F,&d_down); h2d(hx,M*D,&d_x);
    HIP_CHK(hipMalloc(&d_tmp,M*S*sizeof(float)));

    // output buffers
    float *d_gpA,*d_upA,*d_intA,*d_foA,*d_gpB,*d_upB,*d_intB,*d_foB;
    HIP_CHK(hipMalloc(&d_gpA,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_upA,M*F*sizeof(float)));HIP_CHK(hipMalloc(&d_intA,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_foA,M*D*sizeof(float)));
    HIP_CHK(hipMalloc(&d_gpB,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_upB,M*F*sizeof(float)));HIP_CHK(hipMalloc(&d_intB,M*F*sizeof(float)));
    HIP_CHK(hipMalloc(&d_foB,M*D*sizeof(float)));

    // ====== PATH A: explicit injection ======
    HIP_CHK(hipMemset(d_gpA,0,M*F*sizeof(float)));
    gemm(rh,rocblas_operation_transpose,rocblas_operation_none,F,M,D,d_gate,D,d_x,D,d_gpA,F);

    {   // inject gate deltas
        float *dg0,*dg1; h2d(g0.data(),S*H,&dg0); h2d(g1.data(),S*H,&dg1);
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_fwd(rh,M,H,S,dg0,d_x,D,d_tmp);
        std::vector<float> hgp(M*F),ht(M*S);
        HIP_CHK(hipMemcpy(hgp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hgp[i*F+j]+=ht[i*S+j];
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_fwd(rh,M,H,S,dg1,d_x+H,D,d_tmp);
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hgp[i*F+S+j]+=ht[i*S+j];
        HIP_CHK(hipMemcpy(d_gpA,hgp.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(dg0));HIP_CHK(hipFree(dg1));
    }

    HIP_CHK(hipMemset(d_upA,0,M*F*sizeof(float)));
    gemm(rh,rocblas_operation_transpose,rocblas_operation_none,F,M,D,d_up,D,d_x,D,d_upA,F);

    {   // inject up deltas
        float *du0,*du1; h2d(u0.data(),S*H,&du0); h2d(u1.data(),S*H,&du1);
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_fwd(rh,M,H,S,du0,d_x,D,d_tmp);
        std::vector<float> hup(M*F),ht(M*S);
        HIP_CHK(hipMemcpy(hup.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hup[i*F+j]+=ht[i*S+j];
        HIP_CHK(hipMemset(d_tmp,0,M*S*sizeof(float)));
        expand_fwd(rh,M,H,S,du1,d_x+H,D,d_tmp);
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*S*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<S;j++)hup[i*F+S+j]+=ht[i*S+j];
        HIP_CHK(hipMemcpy(d_upA,hup.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(du0));HIP_CHK(hipFree(du1));
    }

    {   // intermediate = silu(gate_pre) * up_pre
        std::vector<float> hgp(M*F),hup(M*F),hint(M*F);
        HIP_CHK(hipMemcpy(hgp.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(hup.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M*F;i++){float gp=hgp[i],sg=1.f/(1.f+expf(-gp));hint[i]=gp*sg*hup[i];}
        HIP_CHK(hipMemcpy(d_intA,hint.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
    }

    {   // ffn_out = intermediate @ down^T
        std::vector<float> h_int(M*F);
        HIP_CHK(hipMemcpy(h_int.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        float *d_ibf; h2d(h_int.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_foA,0,M*D*sizeof(float)));
        gemm(rh,rocblas_operation_transpose,rocblas_operation_none,D,M,F,d_down,F,d_ibf,F,d_foA,D);
        HIP_CHK(hipFree(d_ibf));
    }

    {   // inject down deltas
        float *dd0,*dd1; h2d(d0.data(),S*H,&dd0); h2d(d1.data(),S*H,&dd1);
        std::vector<float> h_int(M*F);
        HIP_CHK(hipMemcpy(h_int.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
        float *d_ibf; h2d(h_int.data(),M*F,&d_ibf);
        HIP_CHK(hipMemset(d_tmp,0,M*H*sizeof(float)));
        contract_fwd(rh,M,S,H,dd0,d_ibf,F,d_tmp);
        std::vector<float> hfo(M*D),ht(M*H);
        HIP_CHK(hipMemcpy(hfo.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*H*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<H;j++)hfo[i*D+j]+=ht[i*H+j];
        HIP_CHK(hipMemset(d_tmp,0,M*H*sizeof(float)));
        contract_fwd(rh,M,S,H,dd1,d_ibf+S,F,d_tmp);
        HIP_CHK(hipMemcpy(ht.data(),d_tmp,M*H*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<M;i++)for(int j=0;j<H;j++)hfo[i*D+H+j]+=ht[i*H+j];
        HIP_CHK(hipMemcpy(d_foA,hfo.data(),M*D*sizeof(float),hipMemcpyHostToDevice));
        HIP_CHK(hipFree(dd0));HIP_CHK(hipFree(dd1));HIP_CHK(hipFree(d_ibf));
    }

    // ====== PATH B: bake + single forward ======
    {
        std::vector<float> gb(gate0),ub(up0),db(down0);
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)gb[i*D+j]+=g0[i*H+j];
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)gb[(S+i)*D+(H+j)]+=g1[i*H+j];
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)ub[i*D+j]+=u0[i*H+j];
        for(int i=0;i<S;i++)for(int j=0;j<H;j++)ub[(S+i)*D+(H+j)]+=u1[i*H+j];
        for(int i=0;i<H;i++)for(int j=0;j<S;j++)db[i*F+j]+=d0t[i*S+j];
        for(int i=0;i<H;i++)for(int j=0;j<S;j++)db[(H+i)*F+(S+j)]+=d1t[i*S+j];

        float *d_gb,*d_ub,*d_db;
        h2d(gb.data(),F*D,&d_gb); h2d(ub.data(),F*D,&d_ub); h2d(db.data(),D*F,&d_db);

        HIP_CHK(hipMemset(d_gpB,0,M*F*sizeof(float)));
        gemm(rh,rocblas_operation_transpose,rocblas_operation_none,F,M,D,d_gb,D,d_x,D,d_gpB,F);
        HIP_CHK(hipMemset(d_upB,0,M*F*sizeof(float)));
        gemm(rh,rocblas_operation_transpose,rocblas_operation_none,F,M,D,d_ub,D,d_x,D,d_upB,F);

        {
            std::vector<float> hgp(M*F),hup(M*F),hint(M*F);
            HIP_CHK(hipMemcpy(hgp.data(),d_gpB,M*F*sizeof(float),hipMemcpyDeviceToHost));
            HIP_CHK(hipMemcpy(hup.data(),d_upB,M*F*sizeof(float),hipMemcpyDeviceToHost));
            for(int i=0;i<M*F;i++){float gp=hgp[i],sg=1.f/(1.f+expf(-gp));hint[i]=gp*sg*hup[i];}
            HIP_CHK(hipMemcpy(d_intB,hint.data(),M*F*sizeof(float),hipMemcpyHostToDevice));
        }
        {
            std::vector<float> hint(M*F);
            HIP_CHK(hipMemcpy(hint.data(),d_intB,M*F*sizeof(float),hipMemcpyDeviceToHost));
            float *d_ibf; h2d(hint.data(),M*F,&d_ibf);
            HIP_CHK(hipMemset(d_foB,0,M*D*sizeof(float)));
            gemm(rh,rocblas_operation_transpose,rocblas_operation_none,D,M,F,d_db,F,d_ibf,F,d_foB,D);
            HIP_CHK(hipFree(d_ibf));
        }
        HIP_CHK(hipFree(d_gb));HIP_CHK(hipFree(d_ub));HIP_CHK(hipFree(d_db));
    }

    HIP_CHK(hipStreamSynchronize(st));

    // compare
    std::vector<float> vgpA(M*F),vgpB(M*F),vupA(M*F),vupB(M*F),vintA(M*F),vintB(M*F),vfoA(M*D),vfoB(M*D);
    HIP_CHK(hipMemcpy(vgpA.data(),d_gpA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vgpB.data(),d_gpB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vupA.data(),d_upA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vupB.data(),d_upB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vintA.data(),d_intA,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vintB.data(),d_intB,M*F*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vfoA.data(),d_foA,M*D*sizeof(float),hipMemcpyDeviceToHost));
    HIP_CHK(hipMemcpy(vfoB.data(),d_foB,M*D*sizeof(float),hipMemcpyDeviceToHost));

    auto cmp=[&](const char*nm,const float*a,const float*b,int n){
        float md=0,ma=0;int mi=0;
        for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]);float m=fmaxf(fabsf(a[i]),fabsf(b[i]));if(d>md){md=d;ma=m;mi=i;}}
        bool ok=md<1e-4f||(ma>0&&md/ma<1e-6f);
        printf("%-20s max|d|=%.2e rel=%.2e @%d  A=%.6f B=%.6f  %s\n",nm,(double)md,(double)(ma>0?md/ma:0),mi,(double)a[mi],(double)b[mi],ok?"OK":"FAIL");
        return ok;
    };

    printf("=== Path A (explicit) vs Path B (baked) — f32 GEMM ===\n");
    fail+=!cmp("gate_pre",vgpA.data(),vgpB.data(),M*F);
    fail+=!cmp("up_pre",vupA.data(),vupB.data(),M*F);
    fail+=!cmp("intermediate",vintA.data(),vintB.data(),M*F);
    fail+=!cmp("ffn_out",vfoA.data(),vfoB.data(),M*D);

    auto scmp=[&](const char*nm,int d,const float*a,const float*b,int off,int sz,int stride){
        float md=0,ma=0;
        for(int i=0;i<M;i++)for(int j=0;j<sz;j++){float dd=fabsf(a[i*stride+off+j]-b[i*stride+off+j]);float mv=fmaxf(fabsf(a[i*stride+off+j]),fabsf(b[i*stride+off+j]));if(dd>md){md=dd;ma=mv;}}
        bool ok=md<1e-4f||(ma>0&&md/ma<1e-6f);
        printf("  %s D%d slice max|d|=%.2e rel=%.2e %s\n",nm,d,(double)md,(double)(ma>0?md/ma:0),ok?"OK":"FAIL");
        if(!ok)fail++;
    };

    printf("\n=== ffn_out A ===\n");
    for(int i=0;i<M;i++){printf(" t%d:",i);for(int j=0;j<D;j++)printf(" %.2f",(double)vfoA[i*D+j]);printf("\n");}

    // cleanup
    HIP_CHK(hipFree(d_gate));HIP_CHK(hipFree(d_up));HIP_CHK(hipFree(d_down));HIP_CHK(hipFree(d_x));HIP_CHK(hipFree(d_tmp));
    HIP_CHK(hipFree(d_gpA));HIP_CHK(hipFree(d_upA));HIP_CHK(hipFree(d_intA));HIP_CHK(hipFree(d_foA));
    HIP_CHK(hipFree(d_gpB));HIP_CHK(hipFree(d_upB));HIP_CHK(hipFree(d_intB));HIP_CHK(hipFree(d_foB));
    RB_CHK(rocblas_destroy_handle(rh)); HIP_CHK(hipStreamDestroy(st));
    printf("\nTest 6: %s\n",fail==0?"PASS":"FAIL");
    return fail;
}
