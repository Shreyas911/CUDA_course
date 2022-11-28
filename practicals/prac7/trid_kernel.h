
__global__ void GPU_trid(int NX, int niter, float *u)
{
  __shared__  float a[128], c[128], d[128];

  float aa, bb, cc, dd, bbi, lambda=1.0;
  int   tid;

  for (int iter=0; iter<niter; iter++) {

    // set tridiagonal coefficients and r.h.s.

    tid = threadIdx.x;
    bbi = 1.0f / (2.0f + lambda);
    
    if (tid>0)
      aa = -bbi;
    else
      aa = 0.0f;

    if (tid<blockDim.x-1)
      cc = -bbi;
    else
      cc = 0.0f;

    if (iter==0) 
      dd = lambda*u[tid]*bbi;
    else
      dd = lambda*dd*bbi;

    a[tid] = aa;
    c[tid] = cc;
    d[tid] = dd;

    // forward pass

    for (int nt=1; nt<NX; nt=2*nt) {
      __syncthreads();  // finish writes before reads

      bb = 1.0f;

      if (tid-nt >= 0) {
        dd = dd - aa*d[tid-nt];
        bb = bb - aa*c[tid-nt];
        aa =    - aa*a[tid-nt];
      }

      if (tid+nt < NX) {
        dd = dd - cc*d[tid+nt];
        bb = bb - cc*a[tid+nt];
        cc =    - cc*c[tid+nt];
      }

      __syncthreads();  // finish reads before writes


      bbi = 1.0f / bb;
      aa  = aa*bbi;
      cc  = cc*bbi;
      dd  = dd*bbi;

      a[tid] = aa;
      c[tid] = cc;
      d[tid] = dd;
    }
  }

  u[tid] = dd;
}

