#include "managetrade.h"
void freeRdata(struct receiveData *rd, int myid){
	if(myid != 0){
		free(rd->alpha);
		free(rd->y);
		free(rd->data1d);
		free(rd->data);
	}
	free(rd->w);
}


void tradeInfo(struct receiveData *rd, struct denseData *ds, struct yDenseData *nds, struct Fullproblem *fp, struct Fullproblem *nfp, int nprocs, int myid, MPI_Comm comm)
{
	int my_missed = 0;
	int *my_missedInds;	

	rd->nFeatures = ds->nFeatures;
	for(int i=0; i<fp->n; i++){
		if(fabs(fp->alpha[i]-fp->C) < 0.00005){
			my_missed++;
		}
	}
	my_missedInds = malloc(sizeof(int)*my_missed);

	if(my_missed > 0){
		int my_missedMark = ds->procInstances;
		int j = 0;
		for(int i=0; i<fp->n; i++){
			if(fabs(fp->alpha[i]-fp->C) < 0.00005){
				if(i >= ds->procPos && my_missedMark == ds->procInstances){
					my_missedMark = j;
				}
				my_missedInds[j] = i;
				j++;
			}
		}

	}
	int my_p = fp->p;
	double* alp;
	int tMissed = my_missed;
	int tP = my_p;
	MPI_Barrier(comm);
	MPI_Allreduce( &(my_missed), &tMissed, 1, MPI_INT, MPI_SUM, comm);
	MPI_Allreduce( &(my_p), &tP, 1, MPI_INT, MPI_SUM, comm);
	rd->total = tMissed+tP;

	int *otherP = malloc(sizeof(int)*nprocs);
	otherP[myid] = fp->p + my_missed;
	if(myid == 0){
		if(tMissed ==0){
			nfp->inactive = NULL;
			nfp->beta = NULL;
		}
		nds->nFeatures = ds->nFeatures;
		nds->nInstances = tMissed + tP;

		nds->data1d = malloc(sizeof(double)*20*nds->nFeatures*nds->nInstances);
		nds->data = malloc(sizeof(double*)*20*nds->nInstances);
		nds->y = malloc(sizeof(int)*20*nds->nInstances);
		for(int i=0; i< 20*nds->nInstances; i++){
			nds->data[i] = &nds->data1d[i*ds->nFeatures];
		}
		nfp->alpha = malloc(sizeof(double)*20*nds->nInstances);
		nfp->gradF = malloc(sizeof(double)*20*nds->nInstances);
		nfp->n = nds->nInstances;
		nfp->p = tP;
		nfp->q = tMissed;
		nfp->C = fp->C;
		nfp->active = malloc(sizeof(int)*nfp->p);
		nfp->inactive = malloc(sizeof(int)*nfp->q);
		nfp->beta = malloc(sizeof(double)*nfp->q);
	}
	else{
		rd->y = malloc(sizeof(int)*20*(fp->p+my_missed));
		alp = malloc(sizeof(double)*(fp->p+my_missed));
		for(int i=0; i<fp->p; i++){
			if(fp->active[i] < ds->procPos){
				alp[i] = fp->alpha[fp->active[i]];
			}
			else{
				alp[i] = -fp->alpha[fp->active[i]];
			}
		}
		for(int i=fp->p; i<fp->p+my_missed; i++){
			if(my_missedInds[i - fp->p] < ds->procPos){
				alp[i] = fp->C;
			}
			else{
				alp[i] = -fp->C;
			}
		}
	}


	int q = otherP[myid];
	for(int i = 1; i<nprocs; i++){
		if(i == myid){
			MPI_Send(&otherP[i], 1, MPI_INT, 0, i, comm);
			MPI_Send(alp, otherP[i], MPI_DOUBLE, 0, i, comm);
		}else if(myid == 0){
			MPI_Recv(&otherP[i] , 1 , MPI_INT, i, i, comm, MPI_STATUS_IGNORE);
			MPI_Recv(&(nfp->gradF[q]) , otherP[i], MPI_DOUBLE, i, i, comm, MPI_STATUS_IGNORE);
			q+=otherP[i];
		}
	}
	if(myid == 0){

		for(int i=0; i<fp->p; i++){
			nfp->gradF[i] = -fp->alpha[fp->active[i]];
			if(fp->active[i] < ds->procPos){
				nfp->gradF[i] = -nfp->gradF[i];
			}
		}
		for(int i=0; i<my_missed; i++){
			nfp->gradF[fp->p + i] = fp->C;
			if(my_missedInds[i] < ds->procPos){
				nfp->gradF[fp->p + i] = -nfp->gradF[fp->p + i];
			}
		}
		int pos = 0;
		for(int i = 0; i< nfp->n; i++){

			if(nfp->gradF[i] > 0){
				nds->y[i] = 1;
				nfp->alpha[i] = nfp->gradF[i];
				pos++;			
			}else{
				nds->y[i] = -1;
				nfp->alpha[i] = -nfp->gradF[i];
			}
			rd->nPos = pos;
		}
		int tot = 0;

		rd->nFeatures = ds->nFeatures;
		for(int id = 0; id< nprocs; id++){
			for(int j=0; j<otherP[id]; j++){
				if(id == 0){
					for(int k =0; k< ds->nFeatures; k++){
						nds->data[tot][k] = ds->data[fp->active[j]][k];
					}
					tot++;
				}else{
					MPI_Recv((nds->data[tot]), ds->nFeatures, MPI_DOUBLE, id, j, comm, MPI_STATUS_IGNORE);
					tot++;
				}
			}
		}
	}else{
		for(int j=0; j<fp->p; j++){
			MPI_Send(ds->data[fp->active[j]], ds->nFeatures, MPI_DOUBLE, 0, j, comm );
		}
		for(int j = fp->p ; j<otherP[myid] ; j++ ){
			MPI_Send(ds->data[my_missedInds[j - fp->p]], ds->nFeatures, MPI_DOUBLE, 0, j, comm);
		}
	}
		MPI_Bcast(&(rd->nPos), 1, MPI_INT, 0,  comm);

	if(myid == 0){
		int act=0, inact = 0;
		for(int i=0; i<nds->nInstances; i++){
			if(fabs(nfp->alpha[i] - nfp->C)< 0.0005){
				nfp->inactive[inact] = i;
				inact++;
			}
			else{
				nfp->active[act] = i;
				act++;
			}
		}
	}

	if(myid ==0){
		for(int i=0; i<nds->nInstances; i++){
			nfp->gradF[i] = 1.0;
		}
	}

	if(myid == 0){
		nfp->partialH = Init_Empty_List();
	  for (int i = 0; i < nfp->p; i++) {
  	  nfp->partialH = Yappend(nds,nfp->partialH,nfp->active[i]);
		}
		Cell* temp = nfp->partialH.head;
		while(temp != NULL){
			for(int i=0; i<nfp->n; i++){
				nfp->gradF[i] -= temp->line[i]*nfp->alpha[temp->label];
			}
			temp = temp->next;
		}
	}else{
		rd->data1d = malloc(sizeof(double)*ds->nFeatures*rd->total*20);
		rd->data = malloc(sizeof(double*)*rd->total*20);
		for(int i=0; i<rd->total*20; i++){
			rd->data[i] = &(rd->data1d[i*ds->nFeatures]);
		}

		rd->alpha = malloc(sizeof(double)*rd->total*20);
	}
		rd->w = malloc(sizeof(double)*rd->nFeatures);

	free(my_missedInds);	
}
