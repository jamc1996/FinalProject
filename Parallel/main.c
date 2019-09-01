#include "svm.h"
#include "io.h"
#include "fullproblem.h"
#include "subproblem.h"
#include "kernels.h"
#include "linked.h"
#include "algorithm.h"
#include "managetrade.h"

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>


int main(int argc, char *argv[])
/* main function to run the full program. */
{
  char* filename = NULL;
  struct denseData ds;
  struct Fullproblem fp;
  struct Projected sp;
	struct receiveData rd;
	MPI_Win dataWin, alphaWin, ytrWin, gradWin, yWin;

	int nprocs = 1, myid = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Input processed:
  parse_arguments(argc, argv, &filename, &parameters);
  read_file(filename, &ds, nprocs, myid);
  //preprocess(&ds);

  //cleanData(&ds);

  // Projected problem size chosen temporarily
  int p = 6;
  if(parameters.test){

	printf("testing not enabled in parallel. Please run in with serial test function.\n");
    return 0;
  }

  // Full problem allocated and filled in, all alpha = 0.0 all gradF = 1.0:
  alloc_prob(&fp, &ds, p);
  init_prob(&fp, &ds);

  // Subproblem allocated:
  alloc_subprob(&sp, p);



  // We loop until no negative entries in beta:
	if(myid == 0){
		printf("Running serial algorithm on each processor\n");
	}
	run_serial_problem(&ds, &fp, &sp);
	MPI_Barrier(MPI_COMM_WORLD);

 	struct Fullproblem nfp;
	struct yDenseData nds;
	struct Projected nsp;

	tradeInfo(&rd, &ds, &nds, &fp, &nfp, nprocs, myid, MPI_COMM_WORLD);

	if(myid == 0){
	  alloc_subprob(&nsp, nfp.p);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0){
		MPI_Win_create(nds.data1d, 20*nds.nInstances*nds.nFeatures*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &dataWin);
		MPI_Win_create(nfp.alpha, 20*nfp.n*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &alphaWin);
		MPI_Win_create(nfp.gradF, 20*nfp.n*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &gradWin);
		MPI_Win_create(nds.y, 20*nds.nInstances*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &yWin);
		MPI_Win_create(&(nsp.ytr), sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &ytrWin);
	}else{
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &dataWin);
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &alphaWin);
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &gradWin);
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &yWin);
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &ytrWin);
	}


	MPI_Win_fence(MPI_MODE_NOPRECEDE, dataWin);
	MPI_Win_fence(MPI_MODE_NOPRECEDE, yWin);
	MPI_Win_fence(MPI_MODE_NOPRECEDE, alphaWin);
	MPI_Win_fence(MPI_MODE_NOPRECEDE, ytrWin);
	MPI_Win_fence(MPI_MODE_NOPRECEDE, gradWin);

	int sending = 5;
	run_parallel_algorithm( &rd, &ds, &fp, &nds, &nfp, &nsp, dataWin, yWin, alphaWin, ytrWin, gradWin, myid, nprocs, MPI_COMM_WORLD, sending );
	if(myid == 0){
		printf("\nFinished Training!\n");
		freeYdata(&nds);
	}
	freeRdata(&rd, myid);



	freeDenseData(&ds);
	freeFullproblem(&fp);
	freeSubProblem(&sp);

	MPI_Win_fence(MPI_MODE_NOSUCCEED, dataWin);
	MPI_Win_fence(MPI_MODE_NOSUCCEED, alphaWin);
	MPI_Win_fence(MPI_MODE_NOSUCCEED, ytrWin);
	MPI_Win_fence(MPI_MODE_NOSUCCEED, gradWin);
	MPI_Win_fence(MPI_MODE_NOSUCCEED, yWin);

	MPI_Win_free(&yWin);
	MPI_Win_free(&gradWin);
	MPI_Win_free(&dataWin);
	MPI_Win_free(&alphaWin);
	MPI_Win_free(&ytrWin);

	MPI_Finalize();

	return 0;
}
