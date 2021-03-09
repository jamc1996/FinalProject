#ifndef FULLPROBLEM_H
#define FULLPROBLEM_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#include "linked.h"
#include "svm.h"
#include "kernels.h"

/*      fullproblem.h -- header file for fullproblem.c
 *
 *      Author:     John Cormican
 *
 */

void changeP( struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int add);
int findWorstAdd(struct Fullproblem *alphOptProblem , int add, int* temp, int* temp2);
void shrinkSize( struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int k);
void allocProb(struct Fullproblem *prob, struct denseData *fullDataset, int p);
void initProb(struct Fullproblem *prob, struct denseData *fullDataset);
void updateAlphaR(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem);
void calculateBeta(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct denseData *fullDataset);
int swapMostNegative(struct Fullproblem *alphOptProblem);
int singleswap(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int n, struct svm_args *params);
void adjustGradF(struct Fullproblem *alphOptProblem, struct denseData *fullDataset, struct Projected *projectedSubProblem, int n, int worst, int signal, int target, int flag, struct svm_args *params, double diff);
int checkfpConstraints(struct Fullproblem *alphOptProblem);
void findWorst(int *worst, int* target, int* change, int *n, struct denseData *fullDataset, struct Fullproblem *alphOptProblem);
void spreadChange(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int target, double diff, int change, int n);
void reinitprob(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int add, int* temp, int* temp2);
void  freeFullproblem(struct Fullproblem *alphOptProblem);

#endif
