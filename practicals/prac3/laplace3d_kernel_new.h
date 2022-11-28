//
// Notes: one thread per node in the 3D block
//

// device code

__global__ void GPU_laplace3d(int NX, int NY, int NZ,
			const float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
			      //  float* d_u1,
			      //  float* d_u2)
{
  int   i, j, k, indg, IOFF, JOFF, KOFF;
  float u2, sixth=1.0f/6.0f;

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x + blockIdx.x*BLOCK_X;
  j    = threadIdx.y + blockIdx.y*BLOCK_Y;
  k    = threadIdx.z + blockIdx.z*BLOCK_Z;

  IOFF = 1;
  JOFF = NX;
  KOFF = NX*NY;

  indg = i + j*JOFF + k*KOFF;

  if (i>=0 && i<=NX-1 && j>=0 && j<=NY-1 && k>=0 && k<=NZ-1) {
    if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
      u2 = d_u1[indg];  // Dirichlet b.c.'s
    }
    else {
      u2 = ( d_u1[indg-IOFF] + d_u1[indg+IOFF]
           + d_u1[indg-JOFF] + d_u1[indg+JOFF]
           + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
    }
    d_u2[indg] = u2;
  }
}
