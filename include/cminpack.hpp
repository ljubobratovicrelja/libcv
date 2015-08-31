#ifndef __CMINPACK_H
#define __CMINPACK_H

#include "fwd.hpp"

real_t CV_EXPORT enorm(int n, real_t x[]);
real_t CV_EXPORT rownorm(int m, int r, int c, real_t **x);
real_t CV_EXPORT colnorm(int m, int r, int c, real_t **x);

void CV_EXPORT fdjac2(void f(int,int, real_t*,real_t*,int*),int m,int n,real_t x[],real_t fvec[],real_t **fjac,
            int *iflag,real_t epsfcn,real_t wa[]);

void CV_EXPORT lmpar(int n,real_t **fjac,int ipvt[],real_t diag[],
           real_t qtf[],real_t delta,real_t *par,real_t wa1[],
           real_t wa2[],real_t wa3[],real_t wa4[]);

void CV_EXPORT qrfac(int m,int n,real_t **a,int pivot,int ipvt[],
           real_t rdiag[],real_t acnorm[],real_t wa[]);

void CV_EXPORT lmdif(void f(int,int, real_t*,real_t*,int*),int m,int n,real_t x[],int msk[],real_t fvec[],
           real_t ftol,real_t xtol,real_t gtol,int maxfev,real_t epsfcn,
           real_t diag[],int mode,real_t factor,int *info,int *nfev,
           real_t **fjac,int ipvt[],real_t qtf[],real_t wa1[],real_t wa2[],
           real_t wa3[],real_t wa4[]);

void CV_EXPORT qrsolv(int n,real_t **r,int ipvt[],real_t diag[],
            real_t qtb[],real_t x[],real_t sdiag[],real_t wa[]);

int CV_EXPORT lmdif0(void f(int,int, real_t*,real_t*,int*),int m,int n,real_t x[],int msk[],real_t fvec[],
           real_t tol,int *info,int *nfev);


#endif  /* __CMINPACK_H */
