#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "iris/fmm_swapxz.h"

#define IRIS_FMM_MAX_ORDER 20

void mget(float *M, int n, int m, float *out_re, float *out_im)  // n lower, m upper index
{
    int c = n * (n + 1);
    int r = c + m;
    int l = c - m;
    
    if(m == 0) {
	*out_re = M[c];
	*out_im = 0;
    }else if(m > 0) {
	if(m > n) {
	    *out_re = 0.0;
	    *out_im = 0.0;
	}else {
	    *out_re = M[r];
	    *out_im = M[l];
	}
    }else {
	if(-m > n) {
	    *out_re = 0.0;
	    *out_im = 0.0;
	}else if(m % 2 == 0) {
	    *out_re = M[l];
	    *out_im = -M[r];
	}else {
	    *out_re = -M[l];
	    *out_im = M[r];
	}
    }
}

void mrot(float *out, float *in, float alpha, int p)
{    
    for(int n=0;n<=p;n++) {
	int idx = n * (n + 1);

	// no need to rotate for m=0
	float re, im;
	mget(in, n, 0, &re, &im);
	out[idx] = re;
	
	for(int m=1;m<=n;m++) {
	    float re, im;
	    float cos_ma = cos(m*alpha);
	    float sin_ma = sin(m*alpha);
	    mget(in, n, m, &re, &im);
	    out[idx + m] = cos_ma * re - sin_ma * im;
	    out[idx - m] = sin_ma * re + cos_ma * im;
	}
    }
}

void swapT_xz(float *mult, int p)
{
    float tmp[(IRIS_FMM_MAX_ORDER+1)*(IRIS_FMM_MAX_ORDER+1)];
    
    for(int i=1;i<=p;i++) {
	float *src = mult + i*i;
	int cnt = 2*i + 1;
	swapT_fns[i-1](tmp, src);
	memcpy(src, tmp, cnt * sizeof(float));
    }
}

void swap_xz(iris_real *mult, int p)
{
    iris_real tmp[(IRIS_FMM_MAX_ORDER+1)*(IRIS_FMM_MAX_ORDER+1)];
    
    for(int i=1;i<=p;i++) {
	iris_real *src = mult + i*i;
	int cnt = 2*i + 1;
	swap_fns[i-1](tmp, src);
	memcpy(src, tmp, cnt * sizeof(iris_real));
    }
}

iris_real fact[] = {
		    1.0,
		    1.0,
		    2.0,
		    6.0,
		    24.0,
		    120.0,
		    720.0,
		    5040.0,
		    40320.0,
		    362880.0,
		    3628800.0
};

#define P 2
#define N ((P+1)*(P+1))

int main()
{
    float x = 2;
    float y = 3;
    float z = 4;
    float test[] = { 1.0, 7.5, 17.0, 6.5, 48.75, 127.5, 46.0, 110.5, -7.0 };
    float scratch[N];
    float f[N];
    float az = atan2f(x, y);
    float ax = - atan2f(sqrt(x*x + y*y), z);
    
    iris_real r = sqrt(x*x + y*y + z*z);
    printf("az = %f, ax = %f\n", az, ax);
    
    memset(f, 0, N*sizeof(float));
    mrot(scratch, test, az, P);
    swapT_xz(scratch, P);
    mrot(scratch, scratch, ax, P);
    swapT_xz(scratch, P);

    for(int i=0;i<N;i++) {
	printf("before shit[%d] = %f\n", i, scratch[i]);
    }
    
    for(int n=0;n<=P;n++) {
	int idx = n * (n + 1);
	iris_real s = 1.0;
	for(int m=0;m<=n;m++) {
	    for(int k=m;k<=P-n;k++) {
		iris_real re, im;
		mget(scratch, k, m, &re, &im);

		iris_real fct = s * fact[n+k] / pow_fn(r, n+k+1);
		printf("n = %d, m = %d, k = %d, M = (%f, %f), s = %f, fact = %f, denom = %f\n",
		       n, m, k, re, im, s, fact[n+k], pow_fn(r, n+k+1));
		re *= fct;
		im *= fct;
		
		f[idx + m] += re;
		f[idx - m] += im;
	    }
	    s *= -1.0;
	}
    }

    for(int i=0;i<N;i++) {
	printf("after shit[%d] = %f\n", i, f[i]);
    }
    
    swap_xz(f, P);
    mrot(f, f, -ax, P);
    swap_xz(f, P);
    mrot(test, f, -az, P);

    for(int i=0;i<N;i++) {
	printf("out[%d] = %f\n", i, test[i]);
    }
    
}
