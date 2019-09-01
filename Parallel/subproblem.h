#ifndef SUBPROBLEM_H
#define SUBPROBLEM_H

#include "svm.h"
#include "kernels.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*      subproblem.h -- header file for subproblem.c
 *
 *      Author:     John Cormican
 *
 */

void Yinit_subprob(struct Projected *sp, struct Fullproblem *fp, struct yDenseData *ds, struct svm_args *params, int newRows);
void alloc_subprob(struct Projected *sp, int p);
void init_subprob(struct Projected *sp, struct Fullproblem *fp, struct denseData *ds, struct svm_args *params, int newRows);
void init_symmetric(struct Projected *sp, int p);
int cg(struct Projected *sp, struct Fullproblem *fp);
int checkConstraints(struct Projected* sp, struct Fullproblem *fp);
void init_error(struct Projected* sp);
void calc_Hrho(struct Projected *sp);
void linearOp2(double* vecOut, double* vecIn, double a, int p);
void linearOp(double* vecOut, double* vecIn, double a, int p);
void updateGamma(struct Projected *sp, double lambda);
void calcYTR(struct Projected *sp, struct Fullproblem *fp);
void copy_vector(double* a, double* b, int p);
void constraint_projection(double* vecOut, double* vecIn, int* y, int p);
double inner_prod(double *a, double *b, int p);
void   freeSubProblem( struct Projected* sp);

#endif
