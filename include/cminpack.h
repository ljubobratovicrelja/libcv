#ifndef __CMINPACK_H
#define __CMINPACK_H

#ifdef CV_REAL_TYPE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

#if	defined(__cplusplus)
extern "C" {
#endif


real_t enorm(int n, real_t x[]);
real_t rownorm(int m, int r, int c, real_t **x);
real_t colnorm(int m, int r, int c, real_t **x);

void fdjac2(void f(int,int, real_t*,real_t*,int*),int m,int n,real_t x[],real_t fvec[],real_t **fjac,
            int *iflag,real_t epsfcn,real_t wa[]);

void lmpar(int n,real_t **fjac,int ipvt[],real_t diag[],
           real_t qtf[],real_t delta,real_t *par,real_t wa1[],
           real_t wa2[],real_t wa3[],real_t wa4[]);

void qrfac(int m,int n,real_t **a,int pivot,int ipvt[],
           real_t rdiag[],real_t acnorm[],real_t wa[]);

void lmdif(void f(int,int, real_t*,real_t*,int*),int m,int n,real_t x[],int msk[],real_t fvec[],
           real_t ftol,real_t xtol,real_t gtol,int maxfev,real_t epsfcn,
           real_t diag[],int mode,real_t factor,int *info,int *nfev,
           real_t **fjac,int ipvt[],real_t qtf[],real_t wa1[],real_t wa2[],
           real_t wa3[],real_t wa4[]);

void qrsolv(int n,real_t **r,int ipvt[],real_t diag[],
            real_t qtb[],real_t x[],real_t sdiag[],real_t wa[]);

int lmdif0(void f(int,int, real_t*,real_t*,int*),int m,int n,real_t x[],int msk[],real_t fvec[],
           real_t tol,int *info,int *nfev);


#if defined(__cplusplus)
}
#endif


#endif  /* __CMINPACK_H */

/* CMINPACK.H */

