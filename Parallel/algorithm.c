#include "algorithm.h"

/*    algorithm.h -- methods for handling the serial and parallel algorithms.
 *
 *    Author: John Cormican
 */

void run_Yserial_problem(struct yDenseData *ds, struct Fullproblem *fp, struct Projected *sp)
/*Function to run the serial program for data set with y vector. */
{
  int k = 1;
  int max_iters = 10000000;
  int itt = 0;
  int n = 0;

  while(k){

   	Yinit_subprob(sp, fp, ds, &parameters, 1);

   	//  congjugate gradient algorithm
   	//  if algorithm completes n == 0
   	//  if algorithm interrupt n != 0
   	n = cg(sp, fp);
   	updateAlphaR(fp, sp);
   	calcYTR(sp, fp);
   	YcalculateBeta(fp, sp, ds);
   	if (n==0) {

     	int add = 2;
	    int *temp = malloc(sizeof(int)*add);
	    int *temp2 = malloc(sizeof(int)*add);
	    add = findWorstest(fp,add,temp,temp2);

	    if (add == 0){
	      break;
	    }

	    changeP(fp, sp, add);
	    Yreinitprob(ds, fp, sp, add, temp, temp2);
	    free(temp);
	    free(temp2);
	  }

	  if (n) {
	    // BCs broken, fix one at a time for the moment
	    k = Ysingleswap(ds, fp, sp, n, &parameters);
	    if (k < 0) {
	      shrinkSize(fp, sp, k+fp->p);
	    }
	    else{
	      n = checkfpConstraints(fp);
	    }
	  }

	  itt++;
	  if(itt == max_iters){
	    printf("Reached max iters (%d)!!!!!\n\n\n",itt );
	    break;
	  }
	}
}

void run_serial_problem(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp)
/*Function to run the serial program for data set without y vector. */
{
  int k = 1;
  int max_iters = 10000000;
  int itt = 0;
  int n = 0;

  while(k){

   	// H matrix columns re-set and subproblem changed

   	init_subprob(sp, fp, ds, &parameters, 1);

   	//  congjugate gradient algorithm
   	//  if algorithm completes n == 0
   	//  if algorithm interrupt n != 0
   	n = cg(sp, fp);

   	updateAlphaR(fp, sp);
   	calcYTR(sp, fp);
   	calculateBeta(fp, sp, ds);

   	if (n==0) {
     	int add = 2;
	    int *temp = malloc(sizeof(int)*add);
	    int *temp2 = malloc(sizeof(int)*add);
	    add = findWorstest(fp,add,temp,temp2);

	    if (add == 0){
	      break;
	    }

	    changeP(fp, sp, add);
	    reinitprob(ds, fp, sp, add, temp, temp2);

	    free(temp);
	    free(temp2);
	  }

	  if (n) {
	    // BCs broken, fix one at a time for the moment
	    k = singleswap(ds, fp, sp, n, &parameters);
	    if (k < 0) {
	      shrinkSize(fp, sp, k+fp->p);
	    }
	    else{
	      n = checkfpConstraints(fp);
	    }
	  }

	  itt++;
	  if(itt == max_iters){
	    printf("Reached max iters (%d)!!!!!\n\n\n",itt );
	    break;
	  }
	}
}

void rootCalcW(struct receiveData *rd, struct yDenseData *nds, struct Fullproblem *nfp){
/* Function to calculate the w vector on the root node. */
	for(int j=0; j<nds->nFeatures; j++){
		rd->w[j] = 0.0;
	}
	int j;
	#pragma omp parallel private(j)
	for(int i =0; i<nds->nInstances; i++){
		if(nfp->alpha[i] > 0.0){
			for(j=0; j<nds->nFeatures; j++){
				rd->w[j] += (nds->y[i])*nfp->alpha[i]*nds->data[i][j];
			}
		}
	}
}


void calcW(struct receiveData *rd){
/* Funciton to calculate the w vector on non-root nodes. */
	for(int j=0; j<rd->nFeatures; j++){
		rd->w[j] = 0.0;
	}
	int j;
	#pragma omp parallel private(j)
	for(int i =0; i<rd->total; i++){
		if(rd->alpha[i] > 0.0){
			for( j=0; j<rd->nFeatures; j++){
				rd->w[j] += rd->alpha[i]*rd->data[i][j]*rd->y[i];
			}
		}
	}
}

void ReceiveCalcBeta(struct Fullproblem *fp, struct receiveData *rd, struct denseData *ds)
/* Function to calculate the beta from received data.*/
{
	int j;
	#pragma omp parallel private(j)
	for(int i=0; i<fp->q; i++){
		fp->beta[i] = 0.0;
		for( j=0; j<ds->nFeatures; j++){
			fp->beta[i] += rd->w[j]*ds->data[fp->inactive[i]][j];
		}
		if(fp->inactive[i] < ds->procPos){
			fp->gradF[fp->inactive[i]] = 1.0 - fp->beta[i];
		}else{
			fp->gradF[fp->inactive[i]] = 1.0 + fp->beta[i];
		}
		if(fp->inactive[i] < ds->procPos){
			fp->beta[i] = .05 + fp->beta[i] + rd->ytr;
		}else{
			fp->beta[i] = .05 - fp->beta[i] - rd->ytr;
		}
	}
}



void run_parallel_algorithm(struct receiveData *rd, struct denseData *ds, struct Fullproblem *fp, struct yDenseData *nds, struct Fullproblem *nfp, struct Projected *nsp, MPI_Win dataWin, MPI_Win yWin, MPI_Win alphaWin, MPI_Win ytrWin, MPI_Win gradWin, int myid, int nprocs, MPI_Comm Comm, int sending )
/* Function to run the parallel algorithm. */
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

void update_root_nfp(struct yDenseData *nds, struct Fullproblem *nfp, int nprocs, int global_n )
/* Function to update the new fullproblem on the root processor. */
{
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

int find_n_worst(int *temp, int n, struct Fullproblem *fp, int flag)
/* Function to find the n most negative beta values on a processor. */
{
  double *betaVal = malloc(sizeof(double)*n);
  for (int i = 0; i < n; i++) {
    betaVal[i] = 200000;
  }
  for (int i = 0; i < fp->q; i++)
  {
    for (int j = 0; j < n; j++)
    {
      if (fp->beta[i] < betaVal[j])
      {
        for (int k = n - 1; k > j ; k--)
        {
          temp[k] = temp[k-1];
          betaVal[k] = betaVal[k-1];
        }
        temp[j] = fp->inactive[i];
        betaVal[j] = fp->beta[i];
        break;
      }
    }
  }


  for (int i = 0; i < n; i++) {
    if (betaVal[i] > 0 || flag > 2) {
      free(betaVal);
      return i;
    }
  }
  free(betaVal);
  return n;
}

void updatePartialH(struct yDenseData *nds, struct Fullproblem *nfp, int global_n)
/* Function update partial H after data has been put on the root node. */
{
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

void freeYdata( struct yDenseData *nds)
/* Function to free a dense dataset containing a y vector.*/
{
	free(nds->data1d);
	free(nds->data);
	free(nds->y);
}
