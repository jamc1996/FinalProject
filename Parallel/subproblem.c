#include "subproblem.h"


void alloc_subprob(struct Projected *sp, int p)
/*  Function to allocate space to solve the projected problem of size p.
 */
{
  sp->p = p;
  sp->C = 100.0;

  sp->alphaHat = malloc(sizeof(double)*p);
  sp->yHat = malloc(sizeof(int)*p);
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

void Yinit_subprob(struct Projected *sp, struct Fullproblem *fp, struct yDenseData *ds, struct svm_args *params, int newRows)
/* Function to initialize the values of the subproblem struct, given information
 * in a dataset and fp struct with active/inactive vectors initialized.

 *  Required: Everything allocated, active and inactive correct.
 *

 */
{
  for (int i = 0; i < sp->p; i++) {
    sp->yHat[i] = ds->y[fp->active[i]];
    sp->alphaHat[i] = 0.0;    // This is the change from original
    sp->rHat[i] = fp->gradF[fp->active[i]];
  }

  if (newRows) {
    updateSubH(fp, sp, params);
  }
}


void init_subprob(struct Projected *sp, struct Fullproblem *fp, struct denseData *ds, struct svm_args *params, int newRows)
/* Function to initialize the values of the subproblem struct, given information
 * in a dataset and fp struct with active/inactive vectors initialized.

 *  Required: Everything allocated, active and inactive correct.
 *

 */
{
  for (int i = 0; i < sp->p; i++) {
    if(fp->active[i] < ds->procPos){
			sp->yHat[i] = 1;
    }else{
			sp->yHat[i] = -1;
		}
    sp->alphaHat[i] = 0.0;    // This is the change from original
    sp->rHat[i] = fp->gradF[fp->active[i]];
  }

  if (newRows) {
    updateSubH(fp, sp, params);
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
    if(its % 1000 == 0){
      printf("rSq is %lf\n",rSq );
    }
    calc_Hrho(sp);

    if (fabs(inner_prod(sp->Hrho, sp->rho, sp->p)) < 0.00000000000000000000000000001) {
      printf("%lf\n",inner_prod(sp->Hrho, sp->rho, sp->p) );
      for (int j = 0; j < fp->p; j++) {
        printf("act = %d\n",fp->active[j] );
      }
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
/* Function to calculate the innter product of the projected*/
{
  sp->ytr = 0.0;
	#pragma omp parallel for
  for (int i = 0; i < sp->p; i++) {
    sp->ytr += sp->yHat[i]*fp->gradF[fp->active[i]];
  }
  sp->ytr /= (double)(sp->p);
}

void linearOp2(double* vecOut, double* vecIn, double a, int p)
{
	#pragma omp parallel for
  for (int i = 0; i < p; i++) {
    vecOut[i] *= a;
    vecOut[i] += vecIn[i];
  }
}

void updateGamma(struct Projected *sp, double lambda)
{
	#pragma omp parallel for
  for (int i = 0; i < sp->p; i++) {
    sp->Hrho[i]*=lambda;
  }
	int j;
	#pragma omp parallel for private(j)
  for (int i = 0; i < sp->p; i++) {
    sp->gamma[i] -= sp->Hrho[i];
    for (j = 0; j < sp->p; j++) {
      sp->gamma[i] += sp->yHat[i]*sp->yHat[j]*sp->Hrho[j]/((double)sp->p);
    }
  }
}


void linearOp(double* vecOut, double* vecIn, double a, int p)
/* Function to add constant times */
{
  for (int i = 0; i < p; i++) {
    vecOut[i] += a*vecIn[i];
  }
}

void calc_Hrho(struct Projected *sp)
/* Function to multiply rho by H */
{
	int j;
	#pragma omp parallel for private(j)
  for (int i = 0; i < sp->p; i++) {
    sp->Hrho[i] = 0.0;
    for (j = 0; j < i; j++) {
      sp->Hrho[i] += sp->H[j][i]*sp->rho[j];
    }
    for (j = i; j < sp->p; j++) {
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
{
  for (int i = 0; i < p; i++) {
    a[i] = b[i];
  }
}

void constraint_projection(double* vecOut, double* vecIn, int* y, int p)
{
	int j;
	#pragma omp parallel for private(j)
  for (int i = 0; i < p; i++) {
    vecOut[i] = vecIn[i];
    for (j = 0; j < p; j++) {
vecOut[i] -= vecIn[j]*((y[i]*y[j])/((double)p));
    }
  }
}

double inner_prod(double *a, double *b, int p)
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
