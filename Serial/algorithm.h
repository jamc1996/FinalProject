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

int run_algorithm(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp);
void freeDenseData(struct denseData *ds);

#endif
