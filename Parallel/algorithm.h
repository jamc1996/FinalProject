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

/*    algorithm.h -- header file for algorithm.c
 *
 *    Author: John Cormican
 */

void runParallelAlgorithm( struct receiveData *rd, struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct yDenseData *nds, struct Fullproblem *nfp, struct Projected *nsp, MPI_Win dataWin, MPI_Win yWin, MPI_Win alphaWin, MPI_Win ytrWin, MPI_Win gradWin, int myid, int nprocs, MPI_Comm Comm, int sending );
void runSerialProblem(struct yDenseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem);
void run_serial_problem(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem);
int find_n_worst(int *temp, int n, struct Fullproblem *alphOptProblem, int flag);
void rootCalcW(struct receiveData *rd, struct yDenseData *nds, struct Fullproblem *nfp);
void calcW(struct receiveData *rd);
void ReceiveCalcBeta(struct Fullproblem *alphOptProblem, struct receiveData *rd, struct denseData *fullDataset);
void updatePartialH(struct yDenseData *nds, struct Fullproblem *nfp, int global_n);
void freeYdata( struct yDenseData *nds);
void update_root_nfp(struct yDenseData *nds, struct Fullproblem *nfp, int nprocs, int global_n );
