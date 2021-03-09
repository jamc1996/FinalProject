#ifndef KERNELS_H
#define KERNELS_H

#include <stdlib.h>
#include <math.h>

#include "svm.h"
#include "linked.h"

/*      kernels.h -- header file for kernels.c
 *
 *      Author:     John Cormican
 *
 */


int updateSubH(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct denseData *fullDataset, struct svm_args *params);
void partialHupdate(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct denseData *fullDataset, struct svm_args *params, int n, int worst);
void appendUpdate(struct denseData *fullDataset, double *line, int n);

#endif
