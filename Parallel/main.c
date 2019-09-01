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

void run_parallel_algorithm( struct receiveData *rd, struct denseData *ds, struct Fullproblem *fp, struct yDenseData *nds, struct Fullproblem *nfp, struct Projected *nsp, MPI_Win dataWin, MPI_Win yWin, MPI_Win alphaWin, MPI_Win ytrWin, MPI_Win gradWin, int myid, int nprocs, MPI_Comm Comm, int sending );



void update_root_nfp(struct yDenseData *nds, struct Fullproblem *nfp, int nprocs, int global_n ){
	nfp->inactive = realloc(nfp->inactive, sizeof(int)*(nfp->q+global_n)  );
	nfp->beta = realloc(nfp->beta,sizeof(double)*(nfp->q+global_n));
	for( int i=0; i<global_n; i++){
		nfp->alpha[i+nds->nInstances] = 0.0;
		nfp->inactive[i+nfp->q] = i + nds->nInstances;
	}
	nfp->n += global_n;
	nfp->q += global_n;
	Cell *temp = nfp->partialH.head;

	temp = nfp->partialH.head;
	while(temp != NULL){
		temp->line = realloc(temp->line, sizeof(double)*(nfp->n));
		temp = temp->next;
	}
}

void updatePartialH(struct yDenseData *nds, struct Fullproblem *nfp, int global_n){
	Cell* temp = nfp->partialH.head;
	while (temp!=NULL){
		for (int i=nds->nInstances; i<nfp->n; i++){
			temp->line[i] = 0.0;
			for(int j = 0; j<nds->nFeatures; j++){
				temp->line[i] += nds->data[temp->label][j]*nds->data[i][j];
			}
			temp->line[i] *= nds->y[temp->label]*nds->y[i];
		}
		temp = temp->next;
	}

	nds->nInstances += global_n;
}

void freeYdata( struct yDenseData *nds){
	free(nds->data1d);
	free(nds->data);
	free(nds->y);
}


int main(int argc, char *argv[]) {
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

void run_parallel_algorithm(struct receiveData *rd, struct denseData *ds, struct Fullproblem *fp, struct yDenseData *nds, struct Fullproblem *nfp, struct Projected *nsp, MPI_Win dataWin, MPI_Win yWin, MPI_Win alphaWin, MPI_Win ytrWin, MPI_Win gradWin, int myid, int nprocs, MPI_Comm Comm, int sending )
{
	int *temp = malloc(sizeof(int)*sending);
	int *local_n = malloc(sizeof(int)*nprocs);
	int itt = 0;
	while ( 1 ) 
	{
		itt++;
		if(myid != 0)
		{
			MPI_Get(rd->data1d, rd->total*ds->nFeatures, MPI_DOUBLE, 0, 0 , rd->total*ds->nFeatures, MPI_DOUBLE, dataWin);
			MPI_Get(rd->y, rd->total, MPI_INT, 0, 0, rd->total, MPI_INT, yWin);
		}
		if(myid == 0){
			printf("Parallel Iteration: %d\n",itt);
			run_Yserial_problem(nds, nfp, nsp);
		}
		MPI_Win_fence(0, dataWin);
		MPI_Win_fence(0, yWin);	
		MPI_Win_fence(0, alphaWin);
		MPI_Win_fence(0, ytrWin);

		if(myid !=0 ){
			MPI_Get(rd->alpha, rd->total, MPI_DOUBLE, 0, 0 ,rd->total, MPI_DOUBLE, alphaWin);
			MPI_Get(&(rd->ytr), 1, MPI_DOUBLE, 0, 0 ,1, MPI_DOUBLE, ytrWin);
		}
		MPI_Win_fence(0, alphaWin);
		MPI_Win_fence(0, ytrWin);

		if(myid != 0){
			calcW(rd);
		}else{
			rootCalcW(rd, nds, nfp);
			rd->ytr = nsp->ytr;
		}

		ReceiveCalcBeta(fp, rd, ds);

		local_n[myid] = find_n_worst(temp, sending, fp, nprocs);
		int global_n = 0;
		for(int i=0; i<nprocs; i++){
			MPI_Bcast(local_n + i, 1, MPI_INT, i, Comm);
			global_n += local_n[i];
		}
		int local_start = 0;
		for(int i=0; i<myid; i++){
			local_start += local_n[i];
		}
		if(global_n == 0){
			break;
		}
		MPI_Win_fence(0, yWin);
		
		for (int i=0; i < local_n[myid]; i++){
			MPI_Put(ds->data[temp[i]], ds->nFeatures, MPI_DOUBLE, 0, ds->nFeatures*(rd->total+local_start+i), ds->nFeatures, MPI_DOUBLE, dataWin);
			MPI_Put(&fp->gradF[temp[i]], 1, MPI_DOUBLE, 0, rd->total+local_start+i, 1, MPI_DOUBLE, gradWin);
			int y_send = -1;
			if(temp[i] < ds->procPos){
				y_send = 1;
			}
			MPI_Put(&y_send, 1, MPI_INT, 0, rd->total+(local_start)+i, 1, MPI_INT, yWin);
		}
		MPI_Win_fence(0, yWin);
		MPI_Win_fence(0, gradWin);
		MPI_Win_fence(0, dataWin);
		MPI_Barrier(Comm);	
		if(myid == 0){
			update_root_nfp(nds, nfp, nprocs,global_n);
		}
	
		rd->total += global_n;

		if(myid == 0){
			updatePartialH(nds, nfp, global_n);
		}

	}

	free(temp);
	free(local_n);
}























