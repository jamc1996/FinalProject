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

void appendUpdate(struct denseData *fullDataset, double *line, int n)
/*  Function to calculate a column of the matrix PH and store it as an array
 *  in a list.
 */
{
  if (parameters.kernel == LINEAR) {
    for (int i = 0; i < fullDataset->nInstances; i++) {
      line[i] = 0.0;
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        line[i] += fullDataset->data[i][j]*fullDataset->data[n][j];
      }
			if(  (i < fullDataset->nPos  )   ^   (n < fullDataset->nPos)  ){
       	line[i] = -line[i];
      }
    }
  }
  if (parameters.kernel == POLYNOMIAL) {
    for (int i = 0; i < fullDataset->nInstances; i++) {
      line[i] = 0.0;
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        line[i] += fullDataset->data[i][j]*fullDataset->data[n][j];
      }
      line[i] = pow(line[i]+parameters.Gamma, parameters.degree);
      if(  (i < fullDataset->nPos  )   ^   (n < fullDataset->nPos)  ){
       	line[i] = -line[i];
      }
    }
  }
  else if (parameters.kernel == EXPONENTIAL) {
    double x,y;
    for (int i = 0; i < fullDataset->nInstances; i++) {
      y = 0.0;
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        x = fullDataset->data[i][j] - fullDataset->data[n][j];
        y -= x*x;
      }
      y *= (parameters.Gamma);
			line[i] = exp(y);
			if(  (i < fullDataset->nPos  )   ^   (n < fullDataset->nPos)  ){
       	line[i] = -line[i];
      }
    }
  }
}


void partialHupdate(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct denseData *fullDataset, struct svm_args *params, int n, int worst)
/*  Function to replace a line in the list of PH columns with a new line from
 *  column inactive[worst]
 */
{
  double *nline = findListLineSetLabel(alphOptProblem->partialH, alphOptProblem->active[n],alphOptProblem->inactive[worst]);
  if (params->kernel == LINEAR) {
    for (int j = 0; j < alphOptProblem->totalProblemSize; j++) {
      nline[j] = 0.0;
      for (int k = 0; k < fullDataset->nFeatures; k++) {
        nline[j]+=fullDataset->data[alphOptProblem->inactive[worst]][k]*fullDataset->data[j][k];
      }
			if(  (alphOptProblem->inactive[worst] < fullDataset->nPos  )   ^   (j < fullDataset->nPos)  ){
       	nline[j] = -nline[j];
      }
    }
  }
  else if (params->kernel == POLYNOMIAL) {
    for (int j = 0; j < alphOptProblem->totalProblemSize; j++) {
      nline[j] = 0.0;
      for (int k = 0; k < fullDataset->nFeatures; k++) {
        nline[j]+=fullDataset->data[alphOptProblem->inactive[worst]][k]*fullDataset->data[j][k];
      }
      nline[j] = pow(nline[j]+params->Gamma, params->degree);
      if(  (alphOptProblem->inactive[worst] < fullDataset->nPos  )   ^   (j < fullDataset->nPos)  ){
       	nline[j] = -nline[j];
      }
    }
  }
  else if (params->kernel == EXPONENTIAL) {
    double x,y;
    for (int j = 0; j < alphOptProblem->totalProblemSize; j++) {
      y = 0.0;
      for (int k = 0; k < fullDataset->nFeatures; k++) {
        x = fullDataset->data[alphOptProblem->inactive[worst]][k] - fullDataset->data[j][k];
        y -= x*x;
      }
      y *= (params->Gamma);
      nline[j] = exp(y);
			if(  (alphOptProblem->inactive[worst] < fullDataset->nPos  )   ^   (j < fullDataset->nPos)  ){
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

int updateSubH(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct denseData *fullDataset, struct svm_args *params)
/*   Function to caculate projectedSubProblem->H for solution using the conjugate gradient algorithm.
 */
{
  Cell* temp = alphOptProblem->partialH.head;
  int i = 0;
  while ( temp != NULL ) {
    for (int j = i; j < alphOptProblem->projectedProblemSize; j++) {
      projectedSubProblem->H[i][j] = temp->line[alphOptProblem->active[j]];
    }
    i++;
    temp = temp->next;
  }

  return 0;
}
