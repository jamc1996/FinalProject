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

void appendUpdate(struct denseData *ds, double *line, int n)
/*  Function to calculate a column of the matrix PH and store it as an array
 *  in a list.
 */
{
  if (parameters.kernel == LINEAR) {
    for (int i = 0; i < ds->nInstances; i++) {
      line[i] = 0.0;
      for (int j = 0; j < ds->nFeatures; j++) {
        line[i] += ds->data[i][j]*ds->data[n][j];
      }
			if(  (i < ds->nPos  )   ^   (n < ds->nPos)  ){
       	line[i] = -line[i];
      }
    }
  }
  if (parameters.kernel == POLYNOMIAL) {
    for (int i = 0; i < ds->nInstances; i++) {
      line[i] = 0.0;
      for (int j = 0; j < ds->nFeatures; j++) {
        line[i] += ds->data[i][j]*ds->data[n][j];
      }
      line[i] = pow(line[i]+parameters.Gamma, parameters.degree);
      if(  (i < ds->nPos  )   ^   (n < ds->nPos)  ){
       	line[i] = -line[i];
      }
    }
  }
  else if (parameters.kernel == EXPONENTIAL) {
    double x,y;
    for (int i = 0; i < ds->nInstances; i++) {
      y = 0.0;
      for (int j = 0; j < ds->nFeatures; j++) {
        x = ds->data[i][j] - ds->data[n][j];
        y -= x*x;
      }
      y *= (parameters.Gamma);
			line[i] = exp(y);
			if(  (i < ds->nPos  )   ^   (n < ds->nPos)  ){
       	line[i] = -line[i];
      }
    }
  }
}


void partialHupdate(struct Fullproblem *fp, struct Projected *sp, struct denseData *ds, struct svm_args *params, int n, int worst)
/*  Function to replace a line in the list of PH columns with a new line from
 *  column inactive[worst]
 */
{
  double *nline = findListLineSetLabel(fp->partialH, fp->active[n],fp->inactive[worst]);
  if (params->kernel == LINEAR) {
    for (int j = 0; j < fp->n; j++) {
      nline[j] = 0.0;
      for (int k = 0; k < ds->nFeatures; k++) {
        nline[j]+=ds->data[fp->inactive[worst]][k]*ds->data[j][k];
      }
			if(  (fp->inactive[worst] < ds->nPos  )   ^   (j < ds->nPos)  ){
       	nline[j] = -nline[j];
      }
    }
  }
  else if (params->kernel == POLYNOMIAL) {
    for (int j = 0; j < fp->n; j++) {
      nline[j] = 0.0;
      for (int k = 0; k < ds->nFeatures; k++) {
        nline[j]+=ds->data[fp->inactive[worst]][k]*ds->data[j][k];
      }
      nline[j] = pow(nline[j]+params->Gamma, params->degree);
      if(  (fp->inactive[worst] < ds->nPos  )   ^   (j < ds->nPos)  ){
       	nline[j] = -nline[j];
      }
    }
  }
  else if (params->kernel == EXPONENTIAL) {
    double x,y;
    for (int j = 0; j < fp->n; j++) {
      y = 0.0;
      for (int k = 0; k < ds->nFeatures; k++) {
        x = ds->data[fp->inactive[worst]][k] - ds->data[j][k];
        y -= x*x;
      }
      y *= (params->Gamma);
      nline[j] = exp(y);
			if(  (fp->inactive[worst] < ds->nPos  )   ^   (j < ds->nPos)  ){
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

int updateSubH(struct Fullproblem *fp, struct Projected *sp, struct denseData *ds, struct svm_args *params)
/*   Function to caculate sp->H for solution using the conjugate gradient algorithm.
 */
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
