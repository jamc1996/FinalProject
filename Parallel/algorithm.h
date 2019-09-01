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

void run_parallel_algorithm( struct receiveData *rd, struct denseData *ds, struct Fullproblem *fp, struct yDenseData *nds, struct Fullproblem *nfp, struct Projected *nsp, MPI_Win dataWin, MPI_Win yWin, MPI_Win alphaWin, MPI_Win ytrWin, MPI_Win gradWin, int myid, int nprocs, MPI_Comm Comm, int sending );
void run_Yserial_problem(struct yDenseData *ds, struct Fullproblem *fp, struct Projected *sp);
void run_serial_problem(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp);
int find_n_worst(int *temp, int n, struct Fullproblem *fp, int flag);
void rootCalcW(struct receiveData *rd, struct yDenseData *nds, struct Fullproblem *nfp);
void calcW(struct receiveData *rd);
void ReceiveCalcBeta(struct Fullproblem *fp, struct receiveData *rd, struct denseData *ds);
void updatePartialH(struct yDenseData *nds, struct Fullproblem *nfp, int global_n);
void update_root_nfp(struct yDenseData *nds, struct Fullproblem *nfp, int nprocs, int global_n );
