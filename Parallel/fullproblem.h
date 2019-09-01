#ifndef FULLPROBLEM_H
#define FULLPROBLEM_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#include "linked.h"
#include "svm.h"
#include "kernels.h"

void YcalculateBeta(struct Fullproblem *fp, struct Projected *sp, struct yDenseData *ds);
void YfindWorst(int *worst, int* target, int* change, int *n, struct yDenseData *ds, struct Fullproblem *fp);
int Ysingleswap(struct yDenseData *ds, struct Fullproblem *fp, struct Projected *sp, int n, struct svm_args *params);
void Yreinitprob(struct yDenseData *ds, struct Fullproblem *fp, struct Projected *sp, int add, int* temp, int* temp2);
void YadjustGradF(struct Fullproblem *fp, struct yDenseData *ds, struct Projected *sp, int n, int worst, int signal, int target, int flag, struct svm_args *params, double diff);

void changeP( struct Fullproblem *fp, struct Projected *sp, int add);
int findWorstest(struct Fullproblem *fp , int add, int* temp, int* temp2);
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
void spreadChange( struct Fullproblem *fp, struct Projected *sp, int target, double diff, int change, int n);
void reinitprob(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp, int add, int* temp, int* temp2);
void  freeFullproblem(struct Fullproblem *fp);
void nds_alloc_prob(struct Fullproblem *prob, int p);
void nds_init_prob(struct Fullproblem *newfp, struct Fullproblem *oldfp, struct receiveData *rd, struct denseData *ds);
#endif
