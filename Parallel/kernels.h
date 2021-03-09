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

int setH(struct Fullproblem *prob, struct denseData *fullDataset, struct svm_args *params);
int updateSubH(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct svm_args *params);
void YpartialHupdate(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct yDenseData *fullDataset, struct svm_args *params, int n, int worst);
void partialHupdate(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct denseData *fullDataset, struct svm_args *params, int n, int worst);
void appendUpdate(struct denseData *fullDataset, double *line, int n);
void MPIappendUpdate(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, double *line, int n);
void newAppendUpdate(struct denseData *fullDataset, struct receiveData *rd, struct Fullproblem *oldfp, struct Fullproblem *newfp, double *line, int n);
void YappendUpdate(struct yDenseData *fullDataset, double *line, int n);

#endif
