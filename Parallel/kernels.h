#ifndef KERNELS_H
#define KERNELS_H

#include <stdlib.h>
#include <math.h>

#include <omp.h>

#include "svm.h"
#include "linked.h"

/*      kernels.h -- header file for kernels.c
 *
 *      Author:     John Cormican
 *
 */

int setH(struct Fullproblem *prob, struct denseData *ds, struct svm_args *params);
int updateSubH(struct Fullproblem *fp, struct Projected *sp, struct svm_args *params);
void YpartialHupdate(struct Fullproblem *fp, struct Projected *sp, struct yDenseData *ds, struct svm_args *params, int n, int worst);
void partialHupdate(struct Fullproblem *fp, struct Projected *sp, struct denseData *ds, struct svm_args *params, int n, int worst);
void appendUpdate(struct denseData *ds, double *line, int n);
void MPIappendUpdate(struct denseData *ds, struct Fullproblem *fp, double *line, int n);
void newAppendUpdate(struct denseData *ds, struct receiveData *rd, struct Fullproblem *oldfp, struct Fullproblem *newfp, double *line, int n);
void YappendUpdate(struct yDenseData *ds, double *line, int n);

#endif
