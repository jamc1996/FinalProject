#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <stdio.h>
#include <math.h>

#include "svm.h"
#include "fullproblem.h"
#include "subproblem.h"
#include "kernels.h"
#include "linked.h"

/*      algorithm.h -- header file for algorithm.c
 *
 *      Author:     John Cormican
 *
 */

int runAlgorithm(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem);
void freeDenseData(struct denseData *fullDataset);

#endif
