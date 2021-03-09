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

void allocProjectedProblem(struct Projected *projectedSubProblem, int p);
void initSubprob(struct Projected *projectedSubProblem, struct Fullproblem *alphOptProblem, struct denseData *fullDataset, struct svm_args *params, int newRows);
void init_symmetric(struct Projected *projectedSubProblem, int p);
int runConjGradient(struct Projected *projectedSubProblem, struct Fullproblem *alphOptProblem);
int checkConstraints(struct Projected* projectedSubProblem, struct Fullproblem *alphOptProblem);
void initError(struct Projected* projectedSubProblem);
void calcHrho(struct Projected *projectedSubProblem);
void multiplyVectorThenAddNewVector(double* vecOut, double* vecIn, double multiplier, int vecLength);
void vectorAdditionWithOperandMultiplication(double* vecOut, double* vecIn, double multiplier, int vecLength);
void updateGamma(struct Projected *projectedSubProblem, double lambda);
void calcYTR(struct Projected *projectedSubProblem, struct Fullproblem *alphOptProblem);
void copyVector(double* newCopy, double* templateVector, int vecLength);
void constraintProjection(double* vecOut, double* vecIn, double* yVec, int vecLength);
double innerProduct(double *a, double *b, int vecLength);
void   freeSubProblem( struct Projected* projectedSubProblem);

#endif
