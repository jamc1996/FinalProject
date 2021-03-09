#include "kernels.h"

/*      kernels.c -- program with functions for calculating PH and hatH
 *                matrices for linear, polynomial, and exponential kernels.
 *
 *      Author:     John Cormican
 *
 *      Purpouse:   To perform the heavy calculation of the PH matrix.
 *
 *      Usage:      Functions called to update entries in alphOptProblem.partialH, projectedSubProblem.H
 *
 */
 
int updateSubH(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct svm_args *params)
{
  Cell* temp = alphOptProblem->partialH.head;
  int i = 0;
  while ( temp != NULL ) {
    for (int j = i; j < alphOptProblem->p ; j++) {
      projectedSubProblem->H[i][j] = temp->line[alphOptProblem->active[j]];
    }
    i++;
    temp = temp->next;
  }

  return 0;
}

void MPIappendUpdate(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, double *line, int n)
{
	int j;
  if (parameters.kernel == LINEAR) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < alphOptProblem->p; i++) {
      line[i] = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        line[i] += fullDataset->data[alphOptProblem->active[i]][j]*fullDataset->data[n][j];
      }
			if( (i<fullDataset->procPos) ^  ( n < fullDataset->procPos  ) ){
        	line[i] = -line[i];
			}
    }
  }
  if (parameters.kernel == POLYNOMIAL) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < alphOptProblem->p; i++) {
      line[i] = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        line[i] += fullDataset->data[alphOptProblem->active[i]][j]*fullDataset->data[n][j];
      }
      line[i] = pow(line[i]+parameters.Gamma, parameters.degree);
	if((i<fullDataset->procPos) ^ (n < fullDataset->procPos) ){
        	line[i] = -line[i];
	}
    }
  }
  else if (parameters.kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(j)
    for (int i = 0; i < fullDataset->procInstances; i++) {
      y = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        x = fullDataset->data[alphOptProblem->active[i]][j] - fullDataset->data[n][j];
        y -= x*x;
      }
      y *= (parameters.Gamma);
      line[i] = exp(y);
    	if( (i<fullDataset->procPos) ^ (n < fullDataset->procPos) ){
        	line[i] = -line[i];
	}
}
  }
}
/*
void newAppendUpdate(struct denseData *fullDataset, struct receiveData *rd, struct Fullproblem *oldfp, struct Fullproblem *newfp, double *line, int n)
{
	int j;
  if (parameters.kernel == LINEAR) {
		printf("adding to line %d\n",n);
		if( n < rd->my_p){
			#pragma omp parallel for private(j)
  	  for (int i = 0; i < newfp->n; i++) {
				if(i<rd->my_p){
					line[i] = 0.0;
  	    	for ( j = 0; j < fullDataset->nFeatures; j++) {
  	    	  line[i] += fullDataset->data[rd->myIndex[i]][j]*fullDataset->data[rd->myIndex[n]][j];
  	    	}
  				if( (rd->myIndex[i]<fullDataset->procPos) ^ (rd->myIndex[n] < fullDataset->procPos) ){
  	      	line[i] = -line[i];
					}
				}
				else{
					line[i] = 0.0;
  	    	for ( j = 0; j < fullDataset->nFeatures; j++) {
  	    	  line[i] += fullDataset->data[i - rd->my_p][j]*fullDataset->data[rd->myIndex[n]][j];
  	    	}
  				if( (rd->yr[i-rd->my_p] == 1) ^ (rd->myIndex[n] < fullDataset->procPos) ){
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
  	    	for ( j = 0; j < fullDataset->nFeatures; j++) {
  	    	  line[i] += fullDataset->data[rd->myIndex[i]][j]*fullDataset->data[rd->myIndex[n]][j];
  	    	}
  				if( (rd->myIndex[i]<fullDataset->procPos) ^ (rd->myIndex[n] < fullDataset->procPos) ){
  	      	line[i] = -line[i];
					}
				}
				else{
					line[i] = 0.0;
  	    	for ( j = 0; j < fullDataset->nFeatures; j++) {
  	    	  line[i] += fullDataset->data[i - rd->my_p][j]*fullDataset->data[rd->myIndex[n]][j];
  	    	}
  				if( (rd->yr[i-rd->my_p] == 1) ^ (rd->myIndex[n] < fullDataset->procPos) ){
  	      	line[i] = -line[i];
					}
				}
			}

		}
}
}*/

void YappendUpdate(struct yDenseData *fullDataset, double *line, int n)
{
	int j;
  if (parameters.kernel == LINEAR) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < fullDataset->nInstances; i++) {
      line[i] = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        line[i] += fullDataset->data[i][j]*fullDataset->data[n][j];
      }
	line[i]*=fullDataset->y[i]*fullDataset->y[n];
    }
  }
  if (parameters.kernel == POLYNOMIAL) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < fullDataset->nInstances; i++) {
      line[i] = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        line[i] += fullDataset->data[i][j]*fullDataset->data[n][j];
      }
      line[i] = pow(line[i]+parameters.Gamma, parameters.degree);
	line[i]*=fullDataset->y[i]*fullDataset->y[n];
    }
  }
  else if (parameters.kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(j)
    for (int i = 0; i < fullDataset->nInstances; i++) {
      y = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        x = fullDataset->data[i][j] - fullDataset->data[n][j];
        y -= x*x;
      }
      y *= (parameters.Gamma);
      line[i] = exp(y);
	line[i]*=fullDataset->y[i]*fullDataset->y[n];
	}
  }
}

void appendUpdate(struct denseData *fullDataset, double *line, int n)
{
	int j;
  if (parameters.kernel == LINEAR) {
    #pragma omp parallel for private(j)
    for (int i = 0; i < fullDataset->procInstances; i++) {
      line[i] = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        line[i] += fullDataset->data[i][j]*fullDataset->data[n][j];
      }
  	if( (i<fullDataset->procPos) ^ (n < fullDataset->procPos) ){
        	line[i] = -line[i];
	}
    }
  }
  if (parameters.kernel == POLYNOMIAL) {
		#pragma omp parallel for private(j)
    for (int i = 0; i < fullDataset->procInstances; i++) {
      line[i] = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        line[i] += fullDataset->data[i][j]*fullDataset->data[n][j];
      }
      line[i] = pow(line[i]+parameters.Gamma, parameters.degree);
     	if( ( i<fullDataset->procPos ) ^ (n < fullDataset->procPos) ){
        	line[i] = -line[i];
	}
    }
  }
  else if (parameters.kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(j)
    for (int i = 0; i < fullDataset->procInstances; i++) {
      y = 0.0;
      for ( j = 0; j < fullDataset->nFeatures; j++) {
        x = fullDataset->data[i][j] - fullDataset->data[n][j];
        y -= x*x;
      }
      y *= (parameters.Gamma);
      line[i] = exp(y);
     	if( ( i<fullDataset->procPos ) ^ ( n < fullDataset->procPos) ){
        	line[i] = -line[i];
	}
}
  }
}


void YpartialHupdate(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct yDenseData *fullDataset, struct svm_args *params, int n, int worst)
{
	int k;
  double *nline = findListLineSetLabel(alphOptProblem->partialH, alphOptProblem->active[n],alphOptProblem->inactive[worst]);
  if (params->kernel == LINEAR) {
		#pragma omp parallel for private(k)
    for (int j = 0; j < alphOptProblem->n; j++) {
      nline[j] = 0.0;
      for ( k = 0; k < fullDataset->nFeatures; k++) {
        nline[j]+=fullDataset->data[alphOptProblem->inactive[worst]][k]*fullDataset->data[j][k];
      }
	nline[j] *= fullDataset->y[j]*fullDataset->y[alphOptProblem->inactive[worst]];
    }
  }
  else if (params->kernel == POLYNOMIAL) {
		#pragma omp parallel for private(k)
    for (int j = 0; j < alphOptProblem->n; j++) {
      nline[j] = 0.0;
      for ( k = 0; k < fullDataset->nFeatures; k++) {
        nline[j]+=fullDataset->data[alphOptProblem->inactive[worst]][k]*fullDataset->data[j][k];
      }
      nline[j] = pow(nline[j]+params->Gamma, params->degree);
     	nline[j] *= fullDataset->y[j]*fullDataset->y[alphOptProblem->inactive[worst]];



    }
  }
  else if (params->kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(k)
    for (int j = 0; j < alphOptProblem->n; j++) {
      y = 0.0;
      for ( k = 0; k < fullDataset->nFeatures; k++) {
        x = fullDataset->data[alphOptProblem->inactive[worst]][k] - fullDataset->data[j][k];
        y -= x*x;
      }
      y *= (params->Gamma);
      nline[j] = exp(y);
     	nline[j] *= fullDataset->y[j]*fullDataset->y[alphOptProblem->inactive[worst]];	;



    }
  }

  for (int i = 0; i < n; i++) {
    projectedSubProblem->H[i][n] = nline[alphOptProblem->active[i]];
  }
  projectedSubProblem->H[n][n] = nline[alphOptProblem->inactive[worst]];

  for (int i = n+1; i < projectedSubProblem->p; i++) {
    projectedSubProblem->H[n][i] = nline[alphOptProblem->active[i]];
  }
}

void partialHupdate(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct denseData *fullDataset, struct svm_args *params, int n, int worst)
{
	int k;
  double *nline = findListLineSetLabel(alphOptProblem->partialH, alphOptProblem->active[n],alphOptProblem->inactive[worst]);
  if (params->kernel == LINEAR) {
		#pragma omp parallel for private(k)
    for (int j = 0; j < alphOptProblem->n; j++) {
      nline[j] = 0.0;
      for ( k = 0; k < fullDataset->nFeatures; k++) {
        nline[j]+=fullDataset->data[alphOptProblem->inactive[worst]][k]*fullDataset->data[j][k];
      }
     	if( ( j<fullDataset->procPos ) ^ ( alphOptProblem->inactive[worst] < fullDataset->procPos) ){
        	nline[j] = -nline[j];
	}
    }
  }
  else if (params->kernel == POLYNOMIAL) {
		#pragma omp parallel for private(k)
    for (int j = 0; j < alphOptProblem->n; j++) {
      nline[j] = 0.0;
      for ( k = 0; k < fullDataset->nFeatures; k++) {
        nline[j]+=fullDataset->data[alphOptProblem->inactive[worst]][k]*fullDataset->data[j][k];
      }
      nline[j] = pow(nline[j]+params->Gamma, params->degree);
     	if( ( j<fullDataset->procPos ) ^ ( alphOptProblem->inactive[worst] < fullDataset->procPos) ){
        	nline[j] = -nline[j];
	}
    }
  }
  else if (params->kernel == EXPONENTIAL) {
    double x,y;
		#pragma omp parallel for private(k)
    for (int j = 0; j < alphOptProblem->n; j++) {
      y = 0.0;
      for ( k = 0; k < fullDataset->nFeatures; k++) {
        x = fullDataset->data[alphOptProblem->inactive[worst]][k] - fullDataset->data[j][k];
        y -= x*x;
      }
      y *= (params->Gamma);
      nline[j] = exp(y);
     	if( ( j<fullDataset->procPos ) ^ ( alphOptProblem->inactive[worst] < fullDataset->procPos) ){
        	nline[j] = -nline[j];
	}
    }
  }

  for (int i = 0; i < n; i++) {
    projectedSubProblem->H[i][n] = nline[alphOptProblem->active[i]];
  }
  projectedSubProblem->H[n][n] = nline[alphOptProblem->inactive[worst]];

  for (int i = n+1; i < projectedSubProblem->p; i++) {
    projectedSubProblem->H[n][i] = nline[alphOptProblem->active[i]];
  }
}
