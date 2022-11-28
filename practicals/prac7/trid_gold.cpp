
void gold_trid(int NX, int niter, float* u, float* c)
{
  float lambda=1.0f, aa, bb, cc, dd;

  for (int iter=0; iter<niter; iter++) {

    //
    // forward pass
    //

    aa   = -1.0f;
    bb   =  2.0f + lambda;
    cc   = -1.0f;
    dd   = lambda*u[0];

    bb   = 1.0f / bb;
    cc   = bb*cc;
    dd   = bb*dd;
    c[0] = cc;
    u[0] = dd;

    for (int i=1; i<NX; i++) {
      aa   = -1.0f;
      bb   = 2.0f + lambda - aa*cc;
      dd   = lambda*u[i] - aa*dd;
      bb   = 1.0f/bb;
      cc   = -bb;
      dd   = bb*dd;
      c[i] = cc;
      u[i] = dd;
    }

    //
    // reverse pass
    //

    u[NX-1] = dd;

    for (int i=NX-2; i>=0; i--) {
      dd   = u[i] - c[i]*dd;
      u[i] = dd;
    }
  }
}


