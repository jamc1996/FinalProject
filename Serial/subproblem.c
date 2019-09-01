#include "subproblem.h"

/*      subproblem.c -- program with functions for solving the projected
 *                        sub problem using the conjugate gradient method.
 *
 *      Author:     John Cormican
 *
 *      Purpouse:   To manage the conjugate gradient algorithm on the subproblem.
 *
 *      Usage:      Various functions called from algorithm.c.
 *
 */


void alloc_subprob(struct Projected *sp, int p)
/*  Function to allocate space to solve the projected problem of size p.
 */
{
  sp->p = p;
  sp->C = 100.0;

  sp->alphaHat = malloc(sizeof(double)*p);
  sp->yHat = malloc(sizeof(double)*p);
  sp->rHat = malloc(sizeof(double)*p);
  sp->gamma = malloc(sizeof(double)*p);
  sp->rho = malloc(sizeof(double)*p);
  sp->Hrho = malloc(sizeof(double)*p);

  // H symmetric so can save space:
  sp->H = malloc(sizeof(double*)*p);
  sp->h = malloc(sizeof(double)*((p*(p+1))/2));
  int j = 0;
  for (int i = 0; i < p; i++) {
    sp->H[i] = &(sp->h[j]);
    j+=(p-i-1);
  }
}

void init_subprob(struct Projected *sp, struct Fullproblem *fp, struct denseData *ds, struct svm_args *params, int newRows)
/* Function to initialize the values of the subproblem struct, given information
 * in a dataset and fp struct with active/inactive vectors initialized.

 *  Required: Everything allocated, active and inactive correct.

 */
{
  for (int i = 0; i < sp->p; i++) {
		if(fp->active[i] < ds->nPos){
    	sp->yHat[i] = 1;
		}else{
			sp->yHat[i] = -1;
		}
    sp->alphaHat[i] = 0.0;    // This is the change from original
    sp->rHat[i] = fp->gradF[fp->active[i]];
  }
  if (newRows) {
    updateSubH(fp, sp, ds, params);
  }

}

int cg(struct Projected *sp, struct Fullproblem *fp)
/* Conjugate gradient method to solve projected subproblem. */
{
  double lambda, mu;
  init_error(sp);
  double rSq = inner_prod(sp->gamma,sp->gamma,sp->p);

  double newRSQ;
  int problem = 0;

  int its = 0;;
  while (rSq > 0.000000001) {
    its++;
    calc_Hrho(sp);

    if (fabs(inner_prod(sp->Hrho, sp->rho, sp->p)) < 0.00000000000000000000000000001) {
      exit(250);
    }
    lambda = rSq/inner_prod(sp->Hrho, sp->rho, sp->p);
    linearOp(sp->alphaHat, sp->rho, lambda, sp->p);

    problem = checkConstraints(sp, fp);

    if(problem){
      if (problem >= sp->p*2) {
        linearOp(sp->alphaHat, sp->rho, -lambda, sp->p);
        return problem;
      }
      else if( problem < -sp->p){
        linearOp(sp->alphaHat, sp->rho, -lambda, sp->p);
        return problem;
      }
      return problem;
    }

    updateGamma(sp, lambda);

    newRSQ = inner_prod(sp->gamma, sp->gamma, sp->p);

    mu = newRSQ/rSq;

    linearOp2(sp->rho, sp->gamma, mu, sp->p);

    rSq = newRSQ;

  }

  return 0;
}

void calcYTR(struct Projected *sp, struct Fullproblem *fp)
/* Function to calculate the average inner product yHat and rHat*/
{
  sp->ytr = 0.0;
  for (int i = 0; i < sp->p; i++) {
    sp->ytr += sp->yHat[i]*fp->gradF[fp->active[i]];
  }
  sp->ytr /= (double)(sp->p);
}

void linearOp2(double* vecOut, double* vecIn, double a, int p)
/* Function to perform vecOut = vecIn + a*vecOut */
{
  for (int i = 0; i < p; i++) {
    vecOut[i] *= a;
    vecOut[i] += vecIn[i];
  }
}

void updateGamma(struct Projected *sp, double lambda)
{
  for (int i = 0; i < sp->p; i++) {
    sp->Hrho[i]*=lambda;
  }
  for (int i = 0; i < sp->p; i++) {
    sp->gamma[i] -= sp->Hrho[i];
    for (int j = 0; j < sp->p; j++) {
      sp->gamma[i] += sp->yHat[i]*sp->yHat[j]*sp->Hrho[j]/((double)sp->p);
    }
  }
}


void linearOp(double* vecOut, double* vecIn, double a, int p)
/* Function to perform vecOut = vecOut + a*vecIn */
{
  for (int i = 0; i < p; i++) {
    vecOut[i] += a*vecIn[i];
  }
}

void calc_Hrho(struct Projected *sp)
/* Function to multiply rho by H */
{
  for (int i = 0; i < sp->p; i++) {
    sp->Hrho[i] = 0.0;
    for (int j = 0; j < i; j++) {
      sp->Hrho[i] += sp->H[j][i]*sp->rho[j];
    }
    for (int j = i; j < sp->p; j++) {
      sp->Hrho[i] += sp->H[i][j]*sp->rho[j];
    }
  }
}



int checkConstraints(struct Projected* sp, struct Fullproblem *fp)
/* Function to check if any constrainst have been violated by the cg process. */
{
  int flag = -1;
  double* temp = (double*)malloc(sizeof(double)*sp->p);
  constraint_projection(temp, sp->alphaHat, sp->yHat, sp->p);
  for (int i = 0; i < sp->p; i++) {
    temp[i] += fp->alpha[ fp->active[i] ];
  }
  for (int i = 0; i < sp->p; i++) {
    if(temp[i]>2*sp->C){
      flag = i;
      for (int j = 0; j < sp->p; j++) {
        if (temp[i] < temp[j]) {
          flag = j;
        }
      }
      free(temp);
      return flag+sp->p+sp->p;
    }
  }
  for (int i = 0; i < sp->p; i++) {
    if(temp[i] <= -sp->C){
      flag = i;
      for (int j = 0; j < sp->p; j++) {
        if (temp[i] > temp[j]) {
          flag = j;
        }
      }
      free(temp);
      return flag-(sp->p+sp->p);
    }
  }
  for (int i = 0; i < sp->p; i++) {
    if(temp[i]>sp->C){
      free(temp);
      return i+sp->p;
    }
    else if(temp[i]<0.0){
      free(temp);
      return i-sp->p;
    }
  }
  free(temp);
  return 0;
}

void init_error(struct Projected* sp)
/* Function to initialize the gamma and rho vectors for cg. */
{
  constraint_projection(sp->gamma, sp->rHat, sp->yHat, sp->p);
  copy_vector(sp->rho, sp->gamma, sp->p);
}

void copy_vector(double* a, double* b, int p)
/* Function to copy vector b in to vector a (both length p)*/
{
  for (int i = 0; i < p; i++) {
    a[i] = b[i];
  }
}

void constraint_projection(double* vecOut, double* vecIn, double* y, int p)
/* Function to perform the necessary constraint projection. */
{
  for (int i = 0; i < p; i++) {
    vecOut[i] = vecIn[i];
    for (int j = 0; j < p; j++) {
      vecOut[i] -= vecIn[j]*(y[i]*y[j]/((double)p));
    }
  }
}

double inner_prod(double *a, double *b, int p)
/* Function to find the inner product of two length p vectors. */
{
  double val = 0.0;
  for (int i = 0; i < p; i++) {
    val+=a[i]*b[i];
  }
  return val;
}



void   freeSubProblem( struct Projected* sp)
/* Function to free dynamically allocated memory in subproblem stuct. */
{
  free(sp->alphaHat);
  free(sp->yHat);
  free(sp->rHat);
  free(sp->H);

  free(sp->gamma);
  free(sp->rho);
  free(sp->Hrho);

  free(sp->h);
}
