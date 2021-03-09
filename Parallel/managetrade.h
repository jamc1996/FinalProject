#include "svm.h"
#include "io.h"
#include "fullproblem.h"
#include "subproblem.h"
#include "kernels.h"
#include "linked.h"
#include "algorithm.h"

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

/*    managetrade.h -- header file for managetrade.c
 *
 *    Author: John Cormican
 */

void tradeInfo(struct receiveData *rd, struct denseData *fullDataset, struct yDenseData *nds, struct Fullproblem *alphOptProblem, struct Fullproblem *newfp, int nprocs, int myid, MPI_Comm comm);
void freeRdata(struct receiveData *rd, int myid);
