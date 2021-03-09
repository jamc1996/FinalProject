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

void allocProjectedProblem(struct Projected *projectedSubProblem, int p)
/*  Function to allocate space to solve the projected problem of size p.
 */
{
  projectedSubProblem->p = p;
  projectedSubProblem->C = 100.0;

  projectedSubProblem->alphaHat = malloc(sizeof(double)*p);
  projectedSubProblem->yHat = malloc(sizeof(int)*p);
  projectedSubProblem->rHat = malloc(sizeof(double)*p);
  projectedSubProblem->gamma = malloc(sizeof(double)*p);
  projectedSubProblem->rho = malloc(sizeof(double)*p);
  projectedSubProblem->Hrho = malloc(sizeof(double)*p);

  // H symmetric so can save space:
  projectedSubProblem->H = malloc(sizeof(double*)*p);
  projectedSubProblem->h = malloc(sizeof(double)*((p*(p+1))/2));
  int j = 0;
  for (int i = 0; i < p; i++) {
    projectedSubProblem->H[i] = &(projectedSubProblem->h[j]);
    j+=(p-i-1);
  }
}

void Yinit_subprob(struct Projected *projectedSubProblem, struct Fullproblem *alphOptProblem, struct yDenseData *fullDataset, struct svm_args *params, int newRows)
/* Function to initialize the values of the subproblem struct, given information
 * in a dataset and alphOptProblem struct with active/inactive vectors initialized.

 *  Required: Everything allocated, active and inactive correct.
 *

 */
{
  for (int i = 0; i < projectedSubProblem->p; i++) {
    projectedSubProblem->yHat[i] = fullDataset->y[alphOptProblem->active[i]];
    projectedSubProblem->alphaHat[i] = 0.0;    // This is the change from original
    projectedSubProblem->rHat[i] = alphOptProblem->gradF[alphOptProblem->active[i]];
  }

  if (newRows) {
    updateSubH(alphOptProblem, projectedSubProblem, params);
  }
}


void init_subprob(struct Projected *projectedSubProblem, struct Fullproblem *alphOptProblem, struct denseData *fullDataset, struct svm_args *params, int newRows)
/* Function to initialize the values of the subproblem struct, given information
 * in a dataset and alphOptProblem struct with active/inactive vectors initialized.

 *  Required: Everything allocated, active and inactive correct.
 *

 */
{
  for (int i = 0; i < projectedSubProblem->p; i++) {
    if(alphOptProblem->active[i] < fullDataset->procPos){
			projectedSubProblem->yHat[i] = 1;
    }else{
			projectedSubProblem->yHat[i] = -1;
		}
    projectedSubProblem->alphaHat[i] = 0.0;    // This is the change from original
    projectedSubProblem->rHat[i] = alphOptProblem->gradF[alphOptProblem->active[i]];
  }

  if (newRows) {
    updateSubH(alphOptProblem, projectedSubProblem, params);
  }
}

int runConjGradient(struct Projected *projectedSubProblem, struct Fullproblem *alphOptProblem)
/* Conjugate gradient method to solve projected subproblem. */
{
  double lambda, mu;
  initError(projectedSubProblem);
  double rSq = innerProduct(projectedSubProblem->gamma,projectedSubProblem->gamma,projectedSubProblem->p);

  double newRSQ;
  int problem = 0;
  int its = 0;;


  while (rSq > 0.000000001) {
    its++;
    if(its % 1000 == 0){
      printf("rSq is %lf\n",rSq );
    }
    calcHrho(projectedSubProblem);

    if (fabs(innerProduct(projectedSubProblem->Hrho, projectedSubProblem->rho, projectedSubProblem->p)) < 0.00000000000000000000000000001) {
      printf("%lf\n",innerProduct(projectedSubProblem->Hrho, projectedSubProblem->rho, projectedSubProblem->p) );
      for (int j = 0; j < alphOptProblem->p; j++) {
        printf("act = %d\n",alphOptProblem->active[j] );
      }
      exit(250);
    }
    lambda = rSq/innerProduct(projectedSubProblem->Hrho, projectedSubProblem->rho, projectedSubProblem->p);
    vectorAdditionWithOperandMultiplication(projectedSubProblem->alphaHat, projectedSubProblem->rho, lambda, projectedSubProblem->p);

    problem = checkConstraints(projectedSubProblem, alphOptProblem);

    if(problem){
      if (problem >= projectedSubProblem->p*2) {
        vectorAdditionWithOperandMultiplication(projectedSubProblem->alphaHat, projectedSubProblem->rho, -lambda, projectedSubProblem->p);
        return problem;
      }
      else if( problem < -projectedSubProblem->p){
        vectorAdditionWithOperandMultiplication(projectedSubProblem->alphaHat, projectedSubProblem->rho, -lambda, projectedSubProblem->p);
        return problem;
      }
      return problem;
    }

    updateGamma(projectedSubProblem, lambda);
    newRSQ = innerProduct(projectedSubProblem->gamma, projectedSubProblem->gamma, projectedSubProblem->p);

    mu = newRSQ/rSq;

    multiplyVectorThenAddNewVector(projectedSubProblem->rho, projectedSubProblem->gamma, mu, projectedSubProblem->p);
    rSq = newRSQ;

  }

  return 0;
}

void calcYTR(struct Projected *projectedSubProblem, struct Fullproblem *alphOptProblem)
/* Function to calculate the innter product of the projected*/
{
  projectedSubProblem->ytr = 0.0;
	#pragma omp parallel for
  for (int i = 0; i < projectedSubProblem->p; i++) {
    projectedSubProblem->ytr += projectedSubProblem->yHat[i]*alphOptProblem->gradF[alphOptProblem->active[i]];
  }
  projectedSubProblem->ytr /= (double)(projectedSubProblem->p);
}

void multiplyVectorThenAddNewVector(double* vecOut, double* vecIn, double multiplier, int vecLength)
{
	#pragma omp parallel for
  for (int i = 0; i < vecLength; i++) {
    vecOut[i] *= multiplier;
    vecOut[i] += vecIn[i];
  }
}

void updateGamma(struct Projected *projectedSubProblem, double lambda)
{
	#pragma omp parallel for
  for (int i = 0; i < projectedSubProblem->p; i++) {
    projectedSubProblem->Hrho[i]*=lambda;
  }
	int j;
	#pragma omp parallel for private(j)
  for (int i = 0; i < projectedSubProblem->p; i++) {
    projectedSubProblem->gamma[i] -= projectedSubProblem->Hrho[i];
    for (j = 0; j < projectedSubProblem->p; j++) {
      projectedSubProblem->gamma[i] += projectedSubProblem->yHat[i]*projectedSubProblem->yHat[j]*projectedSubProblem->Hrho[j]/((double)projectedSubProblem->p);
    }
  }
}


void vectorAdditionWithOperandMultiplication(double* vecOut, double* vecIn, double multiplier, int vecLength)
/* Function to add constant times */
{
  for (int i = 0; i < vecLength; i++) {
    vecOut[i] += multiplier*vecIn[i];
  }
}

void calcHrho(struct Projected *projectedSubProblem)
/* Function to multiply rho by H */
{
	int j;
	#pragma omp parallel for private(j)
  for (int i = 0; i < projectedSubProblem->p; i++) {
    projectedSubProblem->Hrho[i] = 0.0;
    for (j = 0; j < i; j++) {
      projectedSubProblem->Hrho[i] += projectedSubProblem->H[j][i]*projectedSubProblem->rho[j];
    }
    for (j = i; j < projectedSubProblem->p; j++) {
      projectedSubProblem->Hrho[i] += projectedSubProblem->H[i][j]*projectedSubProblem->rho[j];
    }
  }
}


int checkConstraints(struct Projected* projectedSubProblem, struct Fullproblem *alphOptProblem)
/* Function to check if any constrainst have been violated by the runConjGradient process. */
{
  int flag = -1;
  double* temp = (double*)malloc(sizeof(double)*projectedSubProblem->p);
  constraintProjection(temp, projectedSubProblem->alphaHat, projectedSubProblem->yHat, projectedSubProblem->p);
  for (int i = 0; i < projectedSubProblem->p; i++) {
    temp[i] += alphOptProblem->alpha[ alphOptProblem->active[i] ];
  }
  for (int i = 0; i < projectedSubProblem->p; i++) {
    if(temp[i]>2*projectedSubProblem->C){
      flag = i;
      for (int j = 0; j < projectedSubProblem->p; j++) {
        if (temp[i] < temp[j]) {
          flag = j;
        }
      }
      free(temp);
      return flag+projectedSubProblem->p+projectedSubProblem->p;
    }
  }
  for (int i = 0; i < projectedSubProblem->p; i++) {
    if(temp[i] <= -projectedSubProblem->C){
      flag = i;
      for (int j = 0; j < projectedSubProblem->p; j++) {
        if (temp[i] > temp[j]) {
          flag = j;
        }
      }
      free(temp);
      return flag-(projectedSubProblem->p+projectedSubProblem->p);
    }
  }
  for (int i = 0; i < projectedSubProblem->p; i++) {
    if(temp[i]>projectedSubProblem->C){
      free(temp);
      return i+projectedSubProblem->p;
    }
    else if(temp[i]<0.0){
      free(temp);
      return i-projectedSubProblem->p;
    }
  }
  free(temp);
  return 0;
}

void initError(struct Projected* projectedSubProblem)
/* Function to initialize the gamma and rho vectors for runConjGradient. */
{
  constraintProjection(projectedSubProblem->gamma, projectedSubProblem->rHat, projectedSubProblem->yHat, projectedSubProblem->p);
  copyVector(projectedSubProblem->rho, projectedSubProblem->gamma, projectedSubProblem->p);
}

void copyVector(double* newCopy, double* templateVector, int vecLength)
{
  for (int i = 0; i < vecLength; i++) {
    newCopy[i] = templateVector[i];
  }
}

void constraintProjection(double* vecOut, double* vecIn, int* yVec, int vecLength)
{
	int j;
	#pragma omp parallel for private(j)
  for (int i = 0; i < vecLength; i++) {
    vecOut[i] = vecIn[i];
    for (j = 0; j < vecLength; j++) {
      vecOut[i] -= vecIn[j]*((yVec[i]*yVec[j])/((double)vecLength));
    }
  }
}

double innerProduct(double *a, double *b, int vectorLength)
{
  double val = 0.0;
  for (int i = 0; i < vectorLength; i++) {
    val+=a[i]*b[i];
  }
  return val;
}



void   freeSubProblem( struct Projected* projectedSubProblem)
/* Function to free dynamically allocated memory in subproblem stuct. */
{
  free(projectedSubProblem->alphaHat);
  free(projectedSubProblem->yHat);
  free(projectedSubProblem->rHat);
  free(projectedSubProblem->H);

  free(projectedSubProblem->gamma);
  free(projectedSubProblem->rho);
  free(projectedSubProblem->Hrho);

  free(projectedSubProblem->h);
}
