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

void changeP( struct Fullproblem *fp, struct Projected *sp, int add);
int findWorstAdd(struct Fullproblem *fp , int add, int* temp, int* temp2);
void shrinkSize( struct Fullproblem *fp, struct Projected *sp, int k);
void alloc_prob(struct Fullproblem *prob, struct denseData *ds, int p);
void init_prob(struct Fullproblem *prob, struct denseData *ds);
void updateAlphaR(struct Fullproblem *fp, struct Projected *sp);
void calculateBeta(struct Fullproblem *fp, struct Projected *sp, struct denseData *ds);
int swapMostNegative(struct Fullproblem *fp);
int singleswap(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp, int n, struct svm_args *params);
void adjustGradF(struct Fullproblem *fp, struct denseData *ds, struct Projected *sp, int n, int worst, int signal, int target, int flag, struct svm_args *params, double diff);
int checkfpConstraints(struct Fullproblem *fp);
void findWorst(int *worst, int* target, int* change, int *n, struct denseData *ds, struct Fullproblem *fp);
void spreadChange(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp, int target, double diff, int change, int n);
void reinitprob(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp, int add, int* temp, int* temp2);
void  freeFullproblem(struct Fullproblem *fp);

#endif
