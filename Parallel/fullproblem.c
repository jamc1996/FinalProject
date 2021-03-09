
#include "fullproblem.h"

/*      fullproblem.c -- program with functions for solving the full problem
 *                        and monitoring the elements in need of optimisation.
 *
 *      Author:     John Cormican
 *
 *      Purpouse:   To store the solution vector alpha and manage selection of
 *                  elements for optimisation.
 *
 *      Usage:      Various functions called from algorithm.c.
 *
 */

void allocProb(struct Fullproblem *prob, struct denseData *fullDataset, int p)
/* Function to allocate space necessary for a full problem of size n,
 * that will be projected to size p.  */
{
  prob->n = fullDataset->procInstances;
  prob->p = p;
  prob->q = prob->n-prob->p;
  prob->C = 100.0;

  prob->alpha = (double*)calloc(prob->n, sizeof(double) );
  prob->gradF = (double*)malloc(sizeof(double)*prob->n);
  prob->active = (int*)malloc(sizeof(int)*prob->p);
  prob->inactive = (int*)malloc(sizeof(int)*prob->q);
  prob->beta = (double*)malloc(sizeof(double)*(prob->q));
  prob->partialH = Init_Empty_List();


}



void initProb(struct Fullproblem *prob, struct denseData *fullDataset)
/* Function to initialize values for full problem of size n,
 * that will be projected to size p.  */
{
  #pragma omp parallel for
  for (int i = 0; i < prob->n; i++) {
    prob->gradF[i] = 1.0;
 }
  for (int i = 0; i < prob->p/2; i++) {
    prob->active[i] = i;
  }
  for (int i = 0; i < prob->p/2; i++) {
    prob->active[i+prob->p/2] = fullDataset->procPos + i;
  }

  if (prob->p%2 != 0) {
    prob->active[prob->p-1] = fullDataset->procPos+ prob->p/2;
    for (int i = prob->p/2; i < fullDataset->procPos; i++) {
      prob->inactive[i-(prob->p/2)] = i;
    }
    for (int i = 1+(prob->p/2); i < fullDataset->procNeg; i++) {
      prob->inactive[fullDataset->procPos+i-(prob->p)] = fullDataset->procPos + i;
    }
  }
  else{
    for (int i = prob->p/2; i < fullDataset->procPos; i++) {
      prob->inactive[i-(prob->p/2)] = i;
    }
    for (int i = prob->p/2; i < fullDataset->procNeg; i++) {
      prob->inactive[fullDataset->procPos-(prob->p)+i] = fullDataset->procPos + i;
    }
  }

  for (int i = 0; i < prob->p; i++) {
    prob->partialH = append(fullDataset,prob->partialH,prob->active[i]);
	}
}


void updateAlphaR(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem)
/* Function to update the values of the alpha, r vectors after a runConjGradient sweep*/
{
  // Use rho as temporary place to store P*alpha:
  int j;
  #pragma omp parallel for private(j)
  for (int i = 0; i < projectedSubProblem->p; i++) {
    projectedSubProblem->rho[i] = projectedSubProblem->alphaHat[i];
    for (j = 0; j < projectedSubProblem->p; j++) {
      projectedSubProblem->rho[i] -= ((projectedSubProblem->yHat[i]*projectedSubProblem->yHat[j])/((double)(projectedSubProblem->p)))*projectedSubProblem->alphaHat[j];
    }
  }
  // Alpha of each active point is updated:
  for (int i = 0; i < projectedSubProblem->p; i++) {
    alphOptProblem->alpha[alphOptProblem->active[i]] += projectedSubProblem->rho[i];
  }

  // gradF of each inactive point is updated:
  alphOptProblem->partialH.head->prev = alphOptProblem->partialH.head;
  int i = 0;
  while (alphOptProblem->partialH.head->prev != NULL) {
    #pragma omp parallel for
    for ( j = 0; j < alphOptProblem->q; j++) {
      alphOptProblem->gradF[alphOptProblem->inactive[j]] -= alphOptProblem->partialH.head->prev->line[alphOptProblem->inactive[j]] * projectedSubProblem->rho[i];
    }
    i++;
    alphOptProblem->partialH.head->prev = alphOptProblem->partialH.head->prev->next;
  }

  // gradF of each active point is updated:
  for (int i = 0; i < alphOptProblem->p; i++) {
    for ( j = 0; j < i; j++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] -= projectedSubProblem->H[j][i]*projectedSubProblem->rho[j];
    }
    for ( j = i; j < alphOptProblem->p; j++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] -= projectedSubProblem->H[i][j]*projectedSubProblem->rho[j];
    }
  }
}

void YcalculateBeta(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct yDenseData *fullDataset)
/* Function to calculate the beta vector - tracks how suboptimal the values for*/
{
	#pragma omp parallel for
	for (int i = 0; i < alphOptProblem->q; i++) {
		if (alphOptProblem->alpha[alphOptProblem->inactive[i]] < 0.0000000001) {
			alphOptProblem->beta[i] =  - alphOptProblem->gradF[alphOptProblem->inactive[i]] +(fullDataset->y[alphOptProblem->inactive[i]])*(projectedSubProblem->ytr);
		}
		else if (alphOptProblem->alpha[alphOptProblem->inactive[i]] >= projectedSubProblem->C - 0.001) {
	      		alphOptProblem->beta[i] =  alphOptProblem->gradF[alphOptProblem->inactive[i]] - (fullDataset->y[alphOptProblem->inactive[i]])*(projectedSubProblem->ytr);
		}
	}

}


void calculateBeta(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, struct denseData *fullDataset)
/* Function to calculate the beta vector - tracks how suboptimal the values for*/
{
	#pragma omp parallel for
	for (int i = 0; i < alphOptProblem->q; i++) {
		if (alphOptProblem->alpha[alphOptProblem->inactive[i]] < 0.0000000001) {
			if(alphOptProblem->inactive[i] < fullDataset->procPos) {
				alphOptProblem->beta[i] =  - alphOptProblem->gradF[alphOptProblem->inactive[i]] + (projectedSubProblem->ytr);
 			}else{
				alphOptProblem->beta[i] =  - alphOptProblem->gradF[alphOptProblem->inactive[i]] - (projectedSubProblem->ytr);
			}
		}
		else if (alphOptProblem->alpha[alphOptProblem->inactive[i]] >= projectedSubProblem->C - 0.001) {
			if(alphOptProblem->inactive[i] < fullDataset->procPos){
	      			alphOptProblem->beta[i] =  alphOptProblem->gradF[alphOptProblem->inactive[i]] - (projectedSubProblem->ytr);
  	  		}else{
	      			alphOptProblem->beta[i] =  alphOptProblem->gradF[alphOptProblem->inactive[i]] + (projectedSubProblem->ytr);
			}
		}
	}

}

void YfindWorst(int *worst, int* target, int* change, int *n, struct yDenseData *fullDataset, struct Fullproblem *alphOptProblem)
/* Function to find the worst beta value of possible choices. */
{
  double tester = DBL_MAX;

  if (*n > 0) {
    (*change) = 1;
    *n -= alphOptProblem->p;
    if(*n >= alphOptProblem->p){
      *n -= alphOptProblem->p;
      (*change) = -1;
    }
  }
  else {
    (*change) = -1;
    *n+=alphOptProblem->p;
    if(*n < 0){
      *n += alphOptProblem->p;
      (*change) = 1;
    }
  }
  (*target) = (*change)*fullDataset->y[alphOptProblem->active[(*n)]];

	for (int i = 0; i < alphOptProblem->q; i++) {
		if( (fullDataset->y[alphOptProblem->inactive[i]] == *target)){
			if (alphOptProblem->beta[i] < tester) {
				if (alphOptProblem->alpha[alphOptProblem->inactive[i]] < alphOptProblem->C*0.8){
					*worst = i;
					tester = alphOptProblem->beta[i];
				}
			}
		}
		else{
			if (alphOptProblem->beta[i] < tester) {
        			if (alphOptProblem->alpha[alphOptProblem->inactive[i]] > 0.1){
					*worst = i;
					tester = alphOptProblem->beta[i];
				}
			}
		}
	}
	if (tester > 0.0) {
		*worst = -1;
	}
}


void findWorst(int *worst, int* target, int* change, int *n, struct denseData *fullDataset, struct Fullproblem *alphOptProblem)
/* Function to find the worst beta value of possible choices. */
{
  double tester = DBL_MAX;

  if (*n > 0) {
    (*change) = 1;
    *n -= alphOptProblem->p;
    if(*n >= alphOptProblem->p){
      *n -= alphOptProblem->p;
      (*change) = -1;
    }
  }
  else {
    (*change) = -1;
    *n+=alphOptProblem->p;
    if(*n < 0){
      *n += alphOptProblem->p;
      (*change) = 1;
    }
  }
	if(alphOptProblem->active[*n] < fullDataset->procPos){
		*target = (*change);
	}
	else{
		*target = -(*change);
	}

	for (int i = 0; i < alphOptProblem->q; i++) {
		if( (alphOptProblem->inactive[i] < fullDataset->procPos && *target == 1) || (fullDataset->procPos <= alphOptProblem->inactive[i] && *target == -1) ) {
			if (alphOptProblem->beta[i] < tester) {
				if (alphOptProblem->alpha[alphOptProblem->inactive[i]] < alphOptProblem->C*0.8){
					*worst = i;
					tester = alphOptProblem->beta[i];
				}
			}
		}
		else{
			if (alphOptProblem->beta[i] < tester) {
        			if (alphOptProblem->alpha[alphOptProblem->inactive[i]] > 0.1){
					*worst = i;
					tester = alphOptProblem->beta[i];
				}
			}
		}
	}
	if (tester > 0.0) {
		*worst = -1;
	}
}

void spreadChange(struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int target, double diff, int change, int n)
{
  //Change inactive gradF due to changes in alpha[active != n]
  Cell *temp = alphOptProblem->partialH.head;
  int i = 0;
  while (temp != NULL) {
    #pragma omp parallel for
    for (int j = 0; j < alphOptProblem->q; j++) {
      if (temp->label != alphOptProblem->active[n]) {
        if (projectedSubProblem->yHat[i] == target){
          alphOptProblem->gradF[alphOptProblem->inactive[j]] -= temp->line[alphOptProblem->inactive[j]]*diff/(double)(alphOptProblem->p-1);
        }
        else{
          alphOptProblem->gradF[alphOptProblem->inactive[j]] += temp->line[alphOptProblem->inactive[j]]*diff/(double)(alphOptProblem->p-1);
        }
      }
    }
  temp = temp->next;
	i++;
  }
  int j;
  // Change active gradF due to changes in alpha[active != n]
  for (int i = 0; i < alphOptProblem->p; i++) {
    for ( j = 0; j < alphOptProblem->p; j++) {
      if (j != n) {
        if (projectedSubProblem->yHat[j] == target){
          if (i<j) {
            alphOptProblem->gradF[alphOptProblem->active[i]] -= projectedSubProblem->H[i][j]*diff/(double)(alphOptProblem->p-1);
          }
          else{
            alphOptProblem->gradF[alphOptProblem->active[i]] -= projectedSubProblem->H[j][i]*diff/(double)(alphOptProblem->p-1);
          }
        }
        else{
          if (i<j) {
            alphOptProblem->gradF[alphOptProblem->active[i]] += projectedSubProblem->H[i][j]*diff/(double)(alphOptProblem->p-1);
          }
          else{
            alphOptProblem->gradF[alphOptProblem->active[i]] += projectedSubProblem->H[j][i]*diff/(double)(alphOptProblem->p-1);
          }
        }
      }
    }
  }
  double* nline = findListLine(alphOptProblem->partialH,alphOptProblem->active[n]);
  // Changes due to change in alpha[active == n]
  #pragma omp parallel for
  for (int i = 0; i < alphOptProblem->q; i++) {
    alphOptProblem->gradF[alphOptProblem->inactive[i]] += nline[alphOptProblem->inactive[i]]*diff*change;
  }
  for (int i = 0; i < alphOptProblem->p; i++) {
    if(i<n){
      alphOptProblem->gradF[alphOptProblem->active[i]] += projectedSubProblem->H[i][n]*diff*change;
    }
    else{
      alphOptProblem->gradF[alphOptProblem->active[i]] += projectedSubProblem->H[n][i]*diff*change;
    }
  }

  // Minor alpha changes
  for (int j = 0; j < alphOptProblem->p; j++) {
    if (j != n) {
      if (projectedSubProblem->yHat[j] == target){
        alphOptProblem->alpha[alphOptProblem->active[j]] += diff/(double)(alphOptProblem->p-1);
      }
      else{
        alphOptProblem->alpha[alphOptProblem->active[j]] -= diff/(double)(alphOptProblem->p-1);
      }
    }
  }
}


int Ysingleswap(struct yDenseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int n, struct svm_args *params)
/* Function to swap out a single element from the active set. */
{
  int flag = 0;
  if (n>0) {
    flag = 1;
  }

  int worst = -1;
  int target, change=1;


  YfindWorst(&worst,&target,&change,&n, fullDataset, alphOptProblem);

  double diff;
  if (flag == 1) {
    diff = change*(alphOptProblem->alpha[alphOptProblem->active[n]] - alphOptProblem->C)  ;
  }
  else{
    diff = change*alphOptProblem->alpha[alphOptProblem->active[n]];
  }

  if( worst < 0)
  {
    spreadChange(alphOptProblem, projectedSubProblem, target, diff, change, n);

    if (flag) {
      alphOptProblem->alpha[alphOptProblem->active[n]] = alphOptProblem->C;
    }
    else{
      alphOptProblem->alpha[alphOptProblem->active[n]] = 0;
    }

    return n - alphOptProblem->p;
  }

  int temp = alphOptProblem->active[n];
  if (flag)
  {
 if ( (fullDataset->y[alphOptProblem->inactive[worst]] == target)){
      YadjustGradF(alphOptProblem, fullDataset, projectedSubProblem, n, worst, change, 1, flag, params, diff);
      alphOptProblem->alpha[alphOptProblem->inactive[worst]] += diff;
      alphOptProblem->alpha[alphOptProblem->active[n]] = projectedSubProblem->C;
      alphOptProblem->active[n] = alphOptProblem->inactive[worst];
      alphOptProblem->inactive[worst] = temp;
      alphOptProblem->beta[worst] = DBL_MAX;
    }
    else{
      YadjustGradF(alphOptProblem, fullDataset, projectedSubProblem, n, worst, change, 0, flag, params, diff);
      alphOptProblem->alpha[alphOptProblem->inactive[worst]] -= diff;
      alphOptProblem->alpha[alphOptProblem->active[n]] = projectedSubProblem->C;
      alphOptProblem->active[n] = alphOptProblem->inactive[worst];
      alphOptProblem->inactive[worst] = temp;
      alphOptProblem->beta[worst] = DBL_MAX;
    }
  }
  else
  {
 if ( fullDataset->y[alphOptProblem->inactive[worst]] == target){
      YadjustGradF(alphOptProblem, fullDataset, projectedSubProblem, n, worst, change, 1, flag, params, diff);
      alphOptProblem->alpha[alphOptProblem->inactive[worst]] += diff;
      alphOptProblem->alpha[alphOptProblem->active[n]] = 0.0;//projectedSubProblem->C ;
      alphOptProblem->active[n] = alphOptProblem->inactive[worst];
      alphOptProblem->inactive[worst] = temp;
      alphOptProblem->beta[worst] = DBL_MAX-1.0;
    }
    else {
      YadjustGradF(alphOptProblem, fullDataset, projectedSubProblem, n, worst, change, 0, flag, params, diff);
      alphOptProblem->alpha[alphOptProblem->inactive[worst]] -= diff;
      alphOptProblem->alpha[alphOptProblem->active[n]] = 0.0;//projectedSubProblem->C ;
      alphOptProblem->active[n] = alphOptProblem->inactive[worst];
      alphOptProblem->inactive[worst] = temp;
      alphOptProblem->beta[worst] = DBL_MAX;
    }
  }


  return 1;
}

int singleswap(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int n, struct svm_args *params)
/* Function to swap out a single element from the active set. */
{
  int flag = 0;
  if (n>0) {
    flag = 1;
  }
  int worst = -1;
  int target, change=1;


  findWorst(&worst,&target,&change,&n, fullDataset, alphOptProblem);

  double diff;
  if (flag == 1) {
    diff = change*(alphOptProblem->alpha[alphOptProblem->active[n]] - alphOptProblem->C)  ;
  }
  else{
    diff = change*alphOptProblem->alpha[alphOptProblem->active[n]];
  }

  if( worst < 0)
  {
    spreadChange( alphOptProblem, projectedSubProblem, target, diff, change, n);

    if (flag) {
      alphOptProblem->alpha[alphOptProblem->active[n]] = alphOptProblem->C;
    }
    else{
      alphOptProblem->alpha[alphOptProblem->active[n]] = 0;
    }

    return n - alphOptProblem->p;
  }

  int temp = alphOptProblem->active[n];
  if (flag)
  {
 if ( (alphOptProblem->inactive[worst] < fullDataset->procPos && target > 0) || (alphOptProblem->inactive[worst] >= fullDataset->procPos && target < 0) ) {
      adjustGradF(alphOptProblem, fullDataset, projectedSubProblem, n, worst, change, 1, flag, params, diff);
      alphOptProblem->alpha[alphOptProblem->inactive[worst]] += diff;
      alphOptProblem->alpha[alphOptProblem->active[n]] = projectedSubProblem->C;
      alphOptProblem->active[n] = alphOptProblem->inactive[worst];
      alphOptProblem->inactive[worst] = temp;
      alphOptProblem->beta[worst] = DBL_MAX;
    }
    else{
      adjustGradF(alphOptProblem, fullDataset, projectedSubProblem, n, worst, change, 0, flag, params, diff);
      alphOptProblem->alpha[alphOptProblem->inactive[worst]] -= diff;
      alphOptProblem->alpha[alphOptProblem->active[n]] = projectedSubProblem->C;
      alphOptProblem->active[n] = alphOptProblem->inactive[worst];
      alphOptProblem->inactive[worst] = temp;
      alphOptProblem->beta[worst] = DBL_MAX;
    }
  }
  else
  {
 if ( (alphOptProblem->inactive[worst] < fullDataset->procPos && target > 0) || (alphOptProblem->inactive[worst] >= fullDataset->procPos && target < 0) ) {
      adjustGradF(alphOptProblem, fullDataset, projectedSubProblem, n, worst, change, 1, flag, params, diff);
      alphOptProblem->alpha[alphOptProblem->inactive[worst]] += diff;
      alphOptProblem->alpha[alphOptProblem->active[n]] = 0.0;//projectedSubProblem->C ;
      alphOptProblem->active[n] = alphOptProblem->inactive[worst];
      alphOptProblem->inactive[worst] = temp;
      alphOptProblem->beta[worst] = DBL_MAX-1.0;
    }
    else {
      adjustGradF(alphOptProblem, fullDataset, projectedSubProblem, n, worst, change, 0, flag, params, diff);
      alphOptProblem->alpha[alphOptProblem->inactive[worst]] -= diff;
      alphOptProblem->alpha[alphOptProblem->active[n]] = 0.0;//projectedSubProblem->C ;
      alphOptProblem->active[n] = alphOptProblem->inactive[worst];
      alphOptProblem->inactive[worst] = temp;
      alphOptProblem->beta[worst] = DBL_MAX;
    }
  }


  return 1;
}

int checkfpConstraints(struct Fullproblem *alphOptProblem)
{
  for (int i = 0; i < alphOptProblem->p; i++) {
    if(alphOptProblem->alpha[alphOptProblem->active[i]]>alphOptProblem->C){
      return i+alphOptProblem->p;
    }
    else if(alphOptProblem->alpha[alphOptProblem->active[i]] < 0.0){
      return i-alphOptProblem->p;
    }
  }
  return 0;
}

void YadjustGradF(struct Fullproblem *alphOptProblem, struct yDenseData *fullDataset, struct Projected *projectedSubProblem, int n, int worst, int signal, int target, int flag, struct svm_args *params, double diff)
{
  // updatee based on change of H matrix:
  double* nline = findListLine(alphOptProblem->partialH,alphOptProblem->active[n]);
  if (signal == -1) {
    #pragma omp parallel for
    for (int i = 0; i < alphOptProblem->q; i++) {
      alphOptProblem->gradF[ alphOptProblem->inactive[i] ] -= nline[alphOptProblem->inactive[i]]*diff ;
    }
    for (int i = 0; i < alphOptProblem->p; i++) {
      if(i<n){
        alphOptProblem->gradF[ alphOptProblem->active[i] ] -= projectedSubProblem->H[i][n]*diff;
      }
      else{
        alphOptProblem->gradF[ alphOptProblem->active[i] ] -= projectedSubProblem->H[n][i]*diff;
      }
    }
  }
  else
  {
    #pragma omp parallel for
    for (int i = 0; i < alphOptProblem->q; i++) {
      alphOptProblem->gradF[ alphOptProblem->inactive[i] ] += nline[alphOptProblem->inactive[i]]*diff ;
    }
    for (int i = 0; i < alphOptProblem->p; i++) {
      if(i<n){
        alphOptProblem->gradF[ alphOptProblem->active[i] ] += projectedSubProblem->H[i][n]*diff ;
      }
      else{
        alphOptProblem->gradF[ alphOptProblem->active[i] ] += projectedSubProblem->H[n][i]*diff ;
      }
    }
  }

  // Update based on change of

  YpartialHupdate(alphOptProblem, projectedSubProblem, fullDataset, params, n, worst);


  if (target) {
    for (int i = 0; i < alphOptProblem->q; i++) {
      if (i == worst) {
        alphOptProblem->gradF[alphOptProblem->active[n]] -= nline[alphOptProblem->active[n]]*diff;
        continue;
      }
      alphOptProblem->gradF[alphOptProblem->inactive[i]] -= nline[alphOptProblem->inactive[i]]*diff;
    }
    for (int i = 0; i < n; i++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] -= projectedSubProblem->H[i][n]*diff;
    }
    alphOptProblem->gradF[alphOptProblem->inactive[worst]] -= projectedSubProblem->H[n][n]*diff;
    for (int i = n+1; i < projectedSubProblem->p; i++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] -= projectedSubProblem->H[n][i]*diff;
    }
  }
  else{
    #pragma omp parallel for
    for (int i = 0; i < alphOptProblem->q; i++) {
      if (i == worst) {
        alphOptProblem->gradF[alphOptProblem->active[n]] += nline[alphOptProblem->active[n]]*diff;
        continue;
      }
      alphOptProblem->gradF[alphOptProblem->inactive[i]] += nline[alphOptProblem->inactive[i]]*diff;
    }
    for (int i = 0; i < n; i++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] += projectedSubProblem->H[i][n]*diff;
    }
    alphOptProblem->gradF[alphOptProblem->inactive[worst]] += projectedSubProblem->H[n][n]*diff;
    for (int i = n+1; i < projectedSubProblem->p; i++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] += projectedSubProblem->H[n][i]*diff;
    }
  }
}


void adjustGradF(struct Fullproblem *alphOptProblem, struct denseData *fullDataset, struct Projected *projectedSubProblem, int n, int worst, int signal, int target, int flag, struct svm_args *params, double diff)
{
  // updatee based on change of H matrix:
  double* nline = findListLine(alphOptProblem->partialH,alphOptProblem->active[n]);
  if (signal == -1) {
    for (int i = 0; i < alphOptProblem->q; i++) {
      alphOptProblem->gradF[ alphOptProblem->inactive[i] ] -= nline[alphOptProblem->inactive[i]]*diff ;
    }
    for (int i = 0; i < alphOptProblem->p; i++) {
      if(i<n){
        alphOptProblem->gradF[ alphOptProblem->active[i] ] -= projectedSubProblem->H[i][n]*diff;
      }
      else{
        alphOptProblem->gradF[ alphOptProblem->active[i] ] -= projectedSubProblem->H[n][i]*diff;
      }
    }
  }
  else
  {
    #pragma omp parallel for
    for (int i = 0; i < alphOptProblem->q; i++) {
      alphOptProblem->gradF[ alphOptProblem->inactive[i] ] += nline[alphOptProblem->inactive[i]]*diff ;
    }
    for (int i = 0; i < alphOptProblem->p; i++) {
      if(i<n){
        alphOptProblem->gradF[ alphOptProblem->active[i] ] += projectedSubProblem->H[i][n]*diff ;
      }
      else{
        alphOptProblem->gradF[ alphOptProblem->active[i] ] += projectedSubProblem->H[n][i]*diff ;
      }
    }
  }

  // Update based on change of

  partialHupdate(alphOptProblem, projectedSubProblem, fullDataset, params, n, worst);


  if (target) {
    for (int i = 0; i < alphOptProblem->q; i++) {
      if (i == worst) {
        alphOptProblem->gradF[alphOptProblem->active[n]] -= nline[alphOptProblem->active[n]]*diff;
        continue;
      }
      alphOptProblem->gradF[alphOptProblem->inactive[i]] -= nline[alphOptProblem->inactive[i]]*diff;
    }
    for (int i = 0; i < n; i++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] -= projectedSubProblem->H[i][n]*diff;
    }
    alphOptProblem->gradF[alphOptProblem->inactive[worst]] -= projectedSubProblem->H[n][n]*diff;
    for (int i = n+1; i < projectedSubProblem->p; i++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] -= projectedSubProblem->H[n][i]*diff;
    }
  }
  else{
    for (int i = 0; i < alphOptProblem->q; i++) {
      if (i == worst) {
        alphOptProblem->gradF[alphOptProblem->active[n]] += nline[alphOptProblem->active[n]]*diff;
        continue;
      }
      alphOptProblem->gradF[alphOptProblem->inactive[i]] += nline[alphOptProblem->inactive[i]]*diff;
    }
    for (int i = 0; i < n; i++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] += projectedSubProblem->H[i][n]*diff;
    }
    alphOptProblem->gradF[alphOptProblem->inactive[worst]] += projectedSubProblem->H[n][n]*diff;
    for (int i = n+1; i < projectedSubProblem->p; i++) {
      alphOptProblem->gradF[alphOptProblem->active[i]] += projectedSubProblem->H[n][i]*diff;
    }
  }
}

void Yreinitprob(struct yDenseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int add, int* temp, int* temp2)
{
  for (int i = 0; i < add; i++) {
    alphOptProblem->active[(alphOptProblem->p - add) + i] = temp[i];
    alphOptProblem->partialH = Yappend(fullDataset,alphOptProblem->partialH, alphOptProblem->active[(alphOptProblem->p - add) + i]);
  }

  int k = 0;
  for (int i = 0; i < alphOptProblem->q; i++) {
    for (int j = 0; j < add; j++) {
      if (alphOptProblem->inactive[i] == temp[j]) {
        alphOptProblem->inactive[i] = temp2[k];
        k++;
        if (k == add) {
          return;
        }
      }
    }
  }


}


void reinitprob(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int add, int* temp, int* temp2)
{
  for (int i = 0; i < add; i++) {
    alphOptProblem->active[(alphOptProblem->p - add) + i] = temp[i];
    alphOptProblem->partialH = append(fullDataset,alphOptProblem->partialH, alphOptProblem->active[(alphOptProblem->p - add) + i]);
  }
  int k = 0;
  for (int i = 0; i < alphOptProblem->q; i++) {
    for (int j = 0; j < add; j++) {
      if (alphOptProblem->inactive[i] == temp[j]) {
        alphOptProblem->inactive[i] = temp2[k];
        k++;
        if (k == add) {
          return;
        }
      }
    }
  }
}

void shrinkSize( struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int k)
{

  int temp = alphOptProblem->active[k];
  for (int i = k; i < alphOptProblem->p  - 1; i++) {
    alphOptProblem->active[i] = alphOptProblem->active[i+1];
  }
  alphOptProblem->partialH = delete(temp, alphOptProblem->partialH);


  alphOptProblem->p--;
  alphOptProblem->q++;

  alphOptProblem->active = realloc(alphOptProblem->active,sizeof(int)*alphOptProblem->p);
  alphOptProblem->inactive = realloc(alphOptProblem->inactive,sizeof(int)*alphOptProblem->q);
  alphOptProblem->inactive[alphOptProblem->q-1] = temp;
  alphOptProblem->beta = realloc(alphOptProblem->beta,sizeof(double)*alphOptProblem->q);



  // Change projected problem struct:

  projectedSubProblem->p--;

  projectedSubProblem->alphaHat = realloc(projectedSubProblem->alphaHat,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->yHat = realloc(projectedSubProblem->yHat,sizeof(int)*projectedSubProblem->p);
  projectedSubProblem->rHat = realloc(projectedSubProblem->rHat,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->gamma = realloc(projectedSubProblem->gamma,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->rho = realloc(projectedSubProblem->rho,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->Hrho = realloc(projectedSubProblem->Hrho,sizeof(double)*projectedSubProblem->p);

  projectedSubProblem->H = realloc(projectedSubProblem->H,sizeof(double*)*projectedSubProblem->p);
  projectedSubProblem->h = realloc(projectedSubProblem->h,sizeof(double)*((projectedSubProblem->p*(projectedSubProblem->p+1))/2));

  int j = 0;
  for (int i = 0; i < projectedSubProblem->p; i++) {
    projectedSubProblem->H[i] = &(projectedSubProblem->h[j]);
    j+=(projectedSubProblem->p-i-1);
  }
  alphOptProblem->partialH.head->prev = alphOptProblem->partialH.head;
  int i = 0;
  while (alphOptProblem->partialH.head->prev != NULL) {
    for (int j = i; j < projectedSubProblem->p; j++) {
      projectedSubProblem->H[i][j] = alphOptProblem->partialH.head->prev->line[alphOptProblem->active[j]];
    }
    i++;
    alphOptProblem->partialH.head->prev = alphOptProblem->partialH.head->prev->next;
  }
}

void changeP( struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem, int add)
{
  alphOptProblem->p += add;
  alphOptProblem->q -= add;
  projectedSubProblem->p += add;

  alphOptProblem->active = realloc(alphOptProblem->active,sizeof(int)*alphOptProblem->p);
  alphOptProblem->inactive = realloc(alphOptProblem->inactive,sizeof(int)*alphOptProblem->q);
  alphOptProblem->beta = realloc(alphOptProblem->beta,sizeof(double)*alphOptProblem->q);

  projectedSubProblem->alphaHat = realloc(projectedSubProblem->alphaHat,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->yHat = realloc(projectedSubProblem->yHat,sizeof(int)*projectedSubProblem->p);
  projectedSubProblem->rHat = realloc(projectedSubProblem->rHat,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->gamma = realloc(projectedSubProblem->gamma,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->rho = realloc(projectedSubProblem->rho,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->Hrho = realloc(projectedSubProblem->Hrho,sizeof(double)*projectedSubProblem->p);
  projectedSubProblem->H = realloc(projectedSubProblem->H,sizeof(double*)*projectedSubProblem->p);
  projectedSubProblem->h = realloc(projectedSubProblem->h,sizeof(double)*((projectedSubProblem->p*(projectedSubProblem->p+1))/2));

  int j = 0;
  for (int i = 0; i < projectedSubProblem->p; i++) {
    projectedSubProblem->H[i] = &(projectedSubProblem->h[j]);
    j+=(projectedSubProblem->p-i-1);
  }
}

int findWorstest(struct Fullproblem *alphOptProblem , int add, int* temp, int* temp2)
{
  double *betaVal = malloc(sizeof(double)*add);
  for (int i = 0; i < add; i++) {
    betaVal[i] = DBL_MAX;
  }
  for (int i = 0; i < alphOptProblem->q; i++)
  {
    for (int j = 0; j < add; j++)
    {
      if (alphOptProblem->beta[i] < betaVal[j])
      {
        for (int k = add - 1; k > j ; k--)
        {
          temp[k] = temp[k-1];
          betaVal[k] = betaVal[k-1];
        }
        temp[j] = alphOptProblem->inactive[i];
        betaVal[j] = alphOptProblem->beta[i];
        break;
      }
    }
  }


  for (int i = 0; i < add; i++) {
    if (betaVal[i] > 0) {
      add = i;
    }
  }


  int flag;
  int k = 0;
  for (int i = 0; i < add; i++) {
    flag = 0;
    for (int j = 0; j < add; j++) {
      if (alphOptProblem->inactive[alphOptProblem->q - add + i] == temp[i]) {
        flag = 1;
      }
    }
    if (flag == 0) {
      temp2[k] = alphOptProblem->inactive[alphOptProblem->q - add + i];
      k++;
    }
  }



  return add;
}

void  freeFullproblem(struct Fullproblem *alphOptProblem)
/* Function to free dynamically allocated memory in Fullproblem struct */
{
  free(alphOptProblem->alpha);
  free(alphOptProblem->beta);
  free(alphOptProblem->gradF);

  free(alphOptProblem->active);
  free(alphOptProblem->inactive);

  free_list(alphOptProblem->partialH);
}
