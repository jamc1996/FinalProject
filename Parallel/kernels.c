#include "kernels.h"

/*      kernels.c -- program with functions for calculating PH and hatH
 *                matrices for linear, polynomial, and exponential kernels.
 *
 *      Author:     John Cormican
 *
 *      Purpouse:   To perform the heavy calculation of the PH matrix.
 *
 *      Usage:      Functions called to update entries in fp.partialH, sp.H
 *
 */
 
int updateSubH(struct Fullproblem *fp, struct Projected *sp, struct svm_args *params)
{
  Cell* temp = fp->partialH.head;
  int i = 0;
  while ( temp != NULL ) {
    for (int j = i; j < fp->p ; j++) {
      sp->H[i][j] = temp->line[fp->active[j]];
    }
    i++;
    temp = temp->next;
  }

  return 0;
}

void MPIappendUpdate(struct denseData *ds, struct Fullproblem *fp, double *line, int n)
{
	int j;
  if (parameters.kernel == LINEAR) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < fp->p; i++) {
      line[i] = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        line[i] += ds->data[fp->active[i]][j]*ds->data[n][j];
      }
			if( (i<ds->procPos) ^  ( n < ds->procPos  ) ){
        	line[i] = -line[i];
			}
    }
  }
  if (parameters.kernel == POLYNOMIAL) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < fp->p; i++) {
      line[i] = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        line[i] += ds->data[fp->active[i]][j]*ds->data[n][j];
      }
      line[i] = pow(line[i]+parameters.Gamma, parameters.degree);
	if((i<ds->procPos) ^ (n < ds->procPos) ){
        	line[i] = -line[i];
	}
    }
  }
  else if (parameters.kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(j)
    for (int i = 0; i < ds->procInstances; i++) {
      y = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        x = ds->data[fp->active[i]][j] - ds->data[n][j];
        y -= x*x;
      }
      y *= (parameters.Gamma);
      line[i] = exp(y);
    	if( (i<ds->procPos) ^ (n < ds->procPos) ){
        	line[i] = -line[i];
	}
}
  }
}
/*
void newAppendUpdate(struct denseData *ds, struct receiveData *rd, struct Fullproblem *oldfp, struct Fullproblem *newfp, double *line, int n)
{
	int j;
  if (parameters.kernel == LINEAR) {
		printf("adding to line %d\n",n);
		if( n < rd->my_p){
			#pragma omp parallel for private(j)
  	  for (int i = 0; i < newfp->n; i++) {
				if(i<rd->my_p){
					line[i] = 0.0;
  	    	for ( j = 0; j < ds->nFeatures; j++) {
  	    	  line[i] += ds->data[rd->myIndex[i]][j]*ds->data[rd->myIndex[n]][j];
  	    	}
  				if( (rd->myIndex[i]<ds->procPos) ^ (rd->myIndex[n] < ds->procPos) ){
  	      	line[i] = -line[i];
					}
				}
				else{
					line[i] = 0.0;
  	    	for ( j = 0; j < ds->nFeatures; j++) {
  	    	  line[i] += ds->data[i - rd->my_p][j]*ds->data[rd->myIndex[n]][j];
  	    	}
  				if( (rd->yr[i-rd->my_p] == 1) ^ (rd->myIndex[n] < ds->procPos) ){
  	      	line[i] = -line[i];
					}
				}
			}
		}
		else{
			#pragma omp parallel for private(j)
  	  for (int i = 0; i < newfp->n; i++) {
				if(i<rd->my_p){
					line[i] = 0.0;
  	    	for ( j = 0; j < ds->nFeatures; j++) {
  	    	  line[i] += ds->data[rd->myIndex[i]][j]*ds->data[rd->myIndex[n]][j];
  	    	}
  				if( (rd->myIndex[i]<ds->procPos) ^ (rd->myIndex[n] < ds->procPos) ){
  	      	line[i] = -line[i];
					}
				}
				else{
					line[i] = 0.0;
  	    	for ( j = 0; j < ds->nFeatures; j++) {
  	    	  line[i] += ds->data[i - rd->my_p][j]*ds->data[rd->myIndex[n]][j];
  	    	}
  				if( (rd->yr[i-rd->my_p] == 1) ^ (rd->myIndex[n] < ds->procPos) ){
  	      	line[i] = -line[i];
					}
				}
			}

		}
}
}*/

void YappendUpdate(struct yDenseData *ds, double *line, int n)
{
	int j;
  if (parameters.kernel == LINEAR) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < ds->nInstances; i++) {
      line[i] = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        line[i] += ds->data[i][j]*ds->data[n][j];
      }
	line[i]*=ds->y[i]*ds->y[n];
    }
  }
  if (parameters.kernel == POLYNOMIAL) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < ds->nInstances; i++) {
      line[i] = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        line[i] += ds->data[i][j]*ds->data[n][j];
      }
      line[i] = pow(line[i]+parameters.Gamma, parameters.degree);
	line[i]*=ds->y[i]*ds->y[n];
    }
  }
  else if (parameters.kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(j)
    for (int i = 0; i < ds->nInstances; i++) {
      y = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        x = ds->data[i][j] - ds->data[n][j];
        y -= x*x;
      }
      y *= (parameters.Gamma);
      line[i] = exp(y);
	line[i]*=ds->y[i]*ds->y[n];
	}
  }
}

void appendUpdate(struct denseData *ds, double *line, int n)
{
	int j;
  if (parameters.kernel == LINEAR) {
    #pragma omp parallel for private(j)
    for (int i = 0; i < ds->procInstances; i++) {
      line[i] = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        line[i] += ds->data[i][j]*ds->data[n][j];
      }
  	if( (i<ds->procPos) ^ (n < ds->procPos) ){
        	line[i] = -line[i];
	}
    }
  }
  if (parameters.kernel == POLYNOMIAL) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < ds->procInstances; i++) {
      line[i] = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        line[i] += ds->data[i][j]*ds->data[n][j];
      }
      line[i] = pow(line[i]+parameters.Gamma, parameters.degree);
     	if( ( i<ds->procPos ) ^ (n < ds->procPos) ){
        	line[i] = -line[i];
	}
    }
  }
  else if (parameters.kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(j)
    for (int i = 0; i < ds->procInstances; i++) {
      y = 0.0;
      for ( j = 0; j < ds->nFeatures; j++) {
        x = ds->data[i][j] - ds->data[n][j];
        y -= x*x;
      }
      y *= (parameters.Gamma);
      line[i] = exp(y);
     	if( ( i<ds->procPos ) ^ ( n < ds->procPos) ){
        	line[i] = -line[i];
	}
}
  }
}


void YpartialHupdate(struct Fullproblem *fp, struct Projected *sp, struct yDenseData *ds, struct svm_args *params, int n, int worst)
{
	int k;
  double *nline = findListLineSetLabel(fp->partialH, fp->active[n],fp->inactive[worst]);
  if (params->kernel == LINEAR) {
		#pragma omp parallel for private(k)
    for (int j = 0; j < fp->n; j++) {
      nline[j] = 0.0;
      for ( k = 0; k < ds->nFeatures; k++) {
        nline[j]+=ds->data[fp->inactive[worst]][k]*ds->data[j][k];
      }
	nline[j] *= ds->y[j]*ds->y[fp->inactive[worst]];
    }
  }
  else if (params->kernel == POLYNOMIAL) {
		#pragma omp parallel for private(k)
    for (int j = 0; j < fp->n; j++) {
      nline[j] = 0.0;
      for ( k = 0; k < ds->nFeatures; k++) {
        nline[j]+=ds->data[fp->inactive[worst]][k]*ds->data[j][k];
      }
      nline[j] = pow(nline[j]+params->Gamma, params->degree);
     	nline[j] *= ds->y[j]*ds->y[fp->inactive[worst]];



    }
  }
  else if (params->kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(k)
    for (int j = 0; j < fp->n; j++) {
      y = 0.0;
      for ( k = 0; k < ds->nFeatures; k++) {
        x = ds->data[fp->inactive[worst]][k] - ds->data[j][k];
        y -= x*x;
      }
      y *= (params->Gamma);
      nline[j] = exp(y);
     	nline[j] *= ds->y[j]*ds->y[fp->inactive[worst]];	;



    }
  }

  for (int i = 0; i < n; i++) {
    sp->H[i][n] = nline[fp->active[i]];
  }
  sp->H[n][n] = nline[fp->inactive[worst]];

  for (int i = n+1; i < sp->p; i++) {
    sp->H[n][i] = nline[fp->active[i]];
  }
}

void partialHupdate(struct Fullproblem *fp, struct Projected *sp, struct denseData *ds, struct svm_args *params, int n, int worst)
{
	int k;
  double *nline = findListLineSetLabel(fp->partialH, fp->active[n],fp->inactive[worst]);
  if (params->kernel == LINEAR) {
		#pragma omp parallel for private(k)
    for (int j = 0; j < fp->n; j++) {
      nline[j] = 0.0;
      for ( k = 0; k < ds->nFeatures; k++) {
        nline[j]+=ds->data[fp->inactive[worst]][k]*ds->data[j][k];
      }
     	if( ( j<ds->procPos ) ^ ( fp->inactive[worst] < ds->procPos) ){
        	nline[j] = -nline[j];
	}
    }
  }
  else if (params->kernel == POLYNOMIAL) {
		#pragma omp parallel for private(k)
    for (int j = 0; j < fp->n; j++) {
      nline[j] = 0.0;
      for ( k = 0; k < ds->nFeatures; k++) {
        nline[j]+=ds->data[fp->inactive[worst]][k]*ds->data[j][k];
      }
      nline[j] = pow(nline[j]+params->Gamma, params->degree);
     	if( ( j<ds->procPos ) ^ ( fp->inactive[worst] < ds->procPos) ){
        	nline[j] = -nline[j];
	}
    }
  }
  else if (params->kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(k)
    for (int j = 0; j < fp->n; j++) {
      y = 0.0;
      for ( k = 0; k < ds->nFeatures; k++) {
        x = ds->data[fp->inactive[worst]][k] - ds->data[j][k];
        y -= x*x;
      }
      y *= (params->Gamma);
      nline[j] = exp(y);
     	if( ( j<ds->procPos ) ^ ( fp->inactive[worst] < ds->procPos) ){
        	nline[j] = -nline[j];
	}
    }
  }

  for (int i = 0; i < n; i++) {
    sp->H[i][n] = nline[fp->active[i]];
  }
  sp->H[n][n] = nline[fp->inactive[worst]];

  for (int i = n+1; i < sp->p; i++) {
    sp->H[n][i] = nline[fp->active[i]];
  }
}
