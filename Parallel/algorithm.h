#include "svm.h"
#include "io.h"
#include "fullproblem.h"
#include "subproblem.h"
#include "kernels.h"
#include "linked.h"

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

void run_Yserial_problem(struct yDenseData *ds, struct Fullproblem *fp, struct Projected *sp);
void run_serial_problem(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp);
int find_n_worst(int *temp, int n, struct Fullproblem *fp, int flag);
void rootCalcW(struct receiveData *rd, struct yDenseData *nds, struct Fullproblem *nfp);
void calcW(struct receiveData *rd);
void ReceiveCalcBeta(struct Fullproblem *fp, struct receiveData *rd, struct denseData *ds);
