
#include "fullproblem.h"

void alloc_prob(struct Fullproblem *prob, struct denseData *ds, int p)
/* Function to allocate space necessary for a full problem of size n,
 * that will be projected to size p.  */
{
  prob->n = ds->procInstances;
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



void init_prob(struct Fullproblem *prob, struct denseData *ds)
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
    prob->active[i+prob->p/2] = ds->procPos + i;
  }

  if (prob->p%2 != 0) {
    prob->active[prob->p-1] = ds->procPos+ prob->p/2;
    for (int i = prob->p/2; i < ds->procPos; i++) {
      prob->inactive[i-(prob->p/2)] = i;
    }
    for (int i = 1+(prob->p/2); i < ds->procNeg; i++) {
      prob->inactive[ds->procPos+i-(prob->p)] = ds->procPos + i;
    }
  }
  else{
    for (int i = prob->p/2; i < ds->procPos; i++) {
      prob->inactive[i-(prob->p/2)] = i;
    }
    for (int i = prob->p/2; i < ds->procNeg; i++) {
      prob->inactive[ds->procPos-(prob->p)+i] = ds->procPos + i;
    }
  }

  for (int i = 0; i < prob->p; i++) {
    prob->partialH = append(ds,prob->partialH,prob->active[i]);
	}
}


void updateAlphaR(struct Fullproblem *fp, struct Projected *sp)
/* Function to update the values of the alpha, r vectors after a cg sweep*/
{
  // Use rho as temporary place to store P*alpha:
  int j;
  #pragma omp parallel for private(j)
  for (int i = 0; i < sp->p; i++) {
    sp->rho[i] = sp->alphaHat[i];
    for (j = 0; j < sp->p; j++) {
      sp->rho[i] -= ((sp->yHat[i]*sp->yHat[j])/((double)(sp->p)))*sp->alphaHat[j];
    }
  }
  // Alpha of each active point is updated:
  for (int i = 0; i < sp->p; i++) {
    fp->alpha[fp->active[i]] += sp->rho[i];
  }

  // gradF of each inactive point is updated:
  fp->partialH.head->prev = fp->partialH.head;
  int i = 0;
  while (fp->partialH.head->prev != NULL) {
    #pragma omp parallel for
    for ( j = 0; j < fp->q; j++) {
      fp->gradF[fp->inactive[j]] -= fp->partialH.head->prev->line[fp->inactive[j]] * sp->rho[i];
    }
    i++;
    fp->partialH.head->prev = fp->partialH.head->prev->next;
  }

  // gradF of each active point is updated:
  for (int i = 0; i < fp->p; i++) {
    for ( j = 0; j < i; j++) {
      fp->gradF[fp->active[i]] -= sp->H[j][i]*sp->rho[j];
    }
    for ( j = i; j < fp->p; j++) {
      fp->gradF[fp->active[i]] -= sp->H[i][j]*sp->rho[j];
    }
  }
}

void YcalculateBeta(struct Fullproblem *fp, struct Projected *sp, struct yDenseData *ds)
/* Function to calculate the beta vector - tracks how suboptimal the values for*/
{
	#pragma omp parallel for
	for (int i = 0; i < fp->q; i++) {
		if (fp->alpha[fp->inactive[i]] < 0.0000000001) {
			fp->beta[i] =  - fp->gradF[fp->inactive[i]] +(ds->y[fp->inactive[i]])*(sp->ytr);
		}
		else if (fp->alpha[fp->inactive[i]] >= sp->C - 0.001) {
	      		fp->beta[i] =  fp->gradF[fp->inactive[i]] - (ds->y[fp->inactive[i]])*(sp->ytr);
		}
	}

}


void calculateBeta(struct Fullproblem *fp, struct Projected *sp, struct denseData *ds)
/* Function to calculate the beta vector - tracks how suboptimal the values for*/
{
	#pragma omp parallel for
	for (int i = 0; i < fp->q; i++) {
		if (fp->alpha[fp->inactive[i]] < 0.0000000001) {
			if(fp->inactive[i] < ds->procPos) {
				fp->beta[i] =  - fp->gradF[fp->inactive[i]] + (sp->ytr);
 			}else{
				fp->beta[i] =  - fp->gradF[fp->inactive[i]] - (sp->ytr);
			}
		}
		else if (fp->alpha[fp->inactive[i]] >= sp->C - 0.001) {
			if(fp->inactive[i] < ds->procPos){
	      			fp->beta[i] =  fp->gradF[fp->inactive[i]] - (sp->ytr);
  	  		}else{
	      			fp->beta[i] =  fp->gradF[fp->inactive[i]] + (sp->ytr);
			}
		}
	}

}

void YfindWorst(int *worst, int* target, int* change, int *n, struct yDenseData *ds, struct Fullproblem *fp)
/* Function to find the worst beta value of possible choices. */
{
  double tester = DBL_MAX;

  if (*n > 0) {
    (*change) = 1;
    *n -= fp->p;
    if(*n >= fp->p){
      *n -= fp->p;
      (*change) = -1;
    }
  }
  else {
    (*change) = -1;
    *n+=fp->p;
    if(*n < 0){
      *n += fp->p;
      (*change) = 1;
    }
  }
  (*target) = (*change)*ds->y[fp->active[(*n)]];

	for (int i = 0; i < fp->q; i++) {
		if( (ds->y[fp->inactive[i]] == *target)){
			if (fp->beta[i] < tester) {
				if (fp->alpha[fp->inactive[i]] < fp->C*0.8){
					*worst = i;
					tester = fp->beta[i];
				}
			}
		}
		else{
			if (fp->beta[i] < tester) {
        			if (fp->alpha[fp->inactive[i]] > 0.1){
					*worst = i;
					tester = fp->beta[i];
				}
			}
		}
	}
	if (tester > 0.0) {
		*worst = -1;
	}
}


void findWorst(int *worst, int* target, int* change, int *n, struct denseData *ds, struct Fullproblem *fp)
/* Function to find the worst beta value of possible choices. */
{
  double tester = DBL_MAX;

  if (*n > 0) {
    (*change) = 1;
    *n -= fp->p;
    if(*n >= fp->p){
      *n -= fp->p;
      (*change) = -1;
    }
  }
  else {
    (*change) = -1;
    *n+=fp->p;
    if(*n < 0){
      *n += fp->p;
      (*change) = 1;
    }
  }
	if(fp->active[*n] < ds->procPos){
		*target = (*change);
	}
	else{
		*target = -(*change);
	}

	for (int i = 0; i < fp->q; i++) {
		if( (fp->inactive[i] < ds->procPos && *target == 1) || (ds->procPos <= fp->inactive[i] && *target == -1) ) {
			if (fp->beta[i] < tester) {
				if (fp->alpha[fp->inactive[i]] < fp->C*0.8){
					*worst = i;
					tester = fp->beta[i];
				}
			}
		}
		else{
			if (fp->beta[i] < tester) {
        			if (fp->alpha[fp->inactive[i]] > 0.1){
					*worst = i;
					tester = fp->beta[i];
				}
			}
		}
	}
	if (tester > 0.0) {
		*worst = -1;
	}
}

void spreadChange(struct Fullproblem *fp, struct Projected *sp, int target, double diff, int change, int n)
{
  //Change inactive gradF due to changes in alpha[active != n]
  Cell *temp = fp->partialH.head;
  int i = 0;
  while (temp != NULL) {
    #pragma omp parallel for
    for (int j = 0; j < fp->q; j++) {
      if (temp->label != fp->active[n]) {
        if (sp->yHat[i] == target){
          fp->gradF[fp->inactive[j]] -= temp->line[fp->inactive[j]]*diff/(double)(fp->p-1);
        }
        else{
          fp->gradF[fp->inactive[j]] += temp->line[fp->inactive[j]]*diff/(double)(fp->p-1);
        }
      }
    }
  temp = temp->next;
	i++;
  }
  int j;
  // Change active gradF due to changes in alpha[active != n]
  for (int i = 0; i < fp->p; i++) {
    for ( j = 0; j < fp->p; j++) {
      if (j != n) {
        if (sp->yHat[j] == target){
          if (i<j) {
            fp->gradF[fp->active[i]] -= sp->H[i][j]*diff/(double)(fp->p-1);
          }
          else{
            fp->gradF[fp->active[i]] -= sp->H[j][i]*diff/(double)(fp->p-1);
          }
        }
        else{
          if (i<j) {
            fp->gradF[fp->active[i]] += sp->H[i][j]*diff/(double)(fp->p-1);
          }
          else{
            fp->gradF[fp->active[i]] += sp->H[j][i]*diff/(double)(fp->p-1);
          }
        }
      }
    }
  }
  double* nline = findListLine(fp->partialH,fp->active[n]);
  // Changes due to change in alpha[active == n]
  #pragma omp parallel for
  for (int i = 0; i < fp->q; i++) {
    fp->gradF[fp->inactive[i]] += nline[fp->inactive[i]]*diff*change;
  }
  for (int i = 0; i < fp->p; i++) {
    if(i<n){
      fp->gradF[fp->active[i]] += sp->H[i][n]*diff*change;
    }
    else{
      fp->gradF[fp->active[i]] += sp->H[n][i]*diff*change;
    }
  }

  // Minor alpha changes
  for (int j = 0; j < fp->p; j++) {
    if (j != n) {
      if (sp->yHat[j] == target){
        fp->alpha[fp->active[j]] += diff/(double)(fp->p-1);
      }
      else{
        fp->alpha[fp->active[j]] -= diff/(double)(fp->p-1);
      }
    }
  }
}


int Ysingleswap(struct yDenseData *ds, struct Fullproblem *fp, struct Projected *sp, int n, struct svm_args *params)
/* Function to swap out a single element from the active set. */
{
  int flag = 0;
  if (n>0) {
    flag = 1;
  }

  int worst = -1;
  int target, change=1;


  YfindWorst(&worst,&target,&change,&n, ds, fp);

  double diff;
  if (flag == 1) {
    diff = change*(fp->alpha[fp->active[n]] - fp->C)  ;
  }
  else{
    diff = change*fp->alpha[fp->active[n]];
  }

  if( worst < 0)
  {
    spreadChange(fp, sp, target, diff, change, n);

    if (flag) {
      fp->alpha[fp->active[n]] = fp->C;
    }
    else{
      fp->alpha[fp->active[n]] = 0;
    }

    return n - fp->p;
  }

  int temp = fp->active[n];
  if (flag)
  {
 if ( (ds->y[fp->inactive[worst]] == target)){ 
      YadjustGradF(fp, ds, sp, n, worst, change, 1, flag, params, diff);
      fp->alpha[fp->inactive[worst]] += diff;
      fp->alpha[fp->active[n]] = sp->C;
      fp->active[n] = fp->inactive[worst];
      fp->inactive[worst] = temp;
      fp->beta[worst] = DBL_MAX;
    }
    else{
      YadjustGradF(fp, ds, sp, n, worst, change, 0, flag, params, diff);
      fp->alpha[fp->inactive[worst]] -= diff;
      fp->alpha[fp->active[n]] = sp->C;
      fp->active[n] = fp->inactive[worst];
      fp->inactive[worst] = temp;
      fp->beta[worst] = DBL_MAX;
    }
  }
  else
  {
 if ( ds->y[fp->inactive[worst]] == target){
      YadjustGradF(fp, ds, sp, n, worst, change, 1, flag, params, diff);
      fp->alpha[fp->inactive[worst]] += diff;
      fp->alpha[fp->active[n]] = 0.0;//sp->C ;
      fp->active[n] = fp->inactive[worst];
      fp->inactive[worst] = temp;
      fp->beta[worst] = DBL_MAX-1.0;
    }
    else {
      YadjustGradF(fp, ds, sp, n, worst, change, 0, flag, params, diff);
      fp->alpha[fp->inactive[worst]] -= diff;
      fp->alpha[fp->active[n]] = 0.0;//sp->C ;
      fp->active[n] = fp->inactive[worst];
      fp->inactive[worst] = temp;
      fp->beta[worst] = DBL_MAX;
    }
  }


  return 1;
}

int singleswap(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp, int n, struct svm_args *params)
/* Function to swap out a single element from the active set. */
{
  int flag = 0;
  if (n>0) {
    flag = 1;
  }
  int worst = -1;
  int target, change=1;


  findWorst(&worst,&target,&change,&n, ds, fp);

  double diff;
  if (flag == 1) {
    diff = change*(fp->alpha[fp->active[n]] - fp->C)  ;
  }
  else{
    diff = change*fp->alpha[fp->active[n]];
  }

  if( worst < 0)
  {
    spreadChange( fp, sp, target, diff, change, n);

    if (flag) {
      fp->alpha[fp->active[n]] = fp->C;
    }
    else{
      fp->alpha[fp->active[n]] = 0;
    }

    return n - fp->p;
  }

  int temp = fp->active[n];
  if (flag)
  {
 if ( (fp->inactive[worst] < ds->procPos && target > 0) || (fp->inactive[worst] >= ds->procPos && target < 0) ) {
      adjustGradF(fp, ds, sp, n, worst, change, 1, flag, params, diff);
      fp->alpha[fp->inactive[worst]] += diff;
      fp->alpha[fp->active[n]] = sp->C;
      fp->active[n] = fp->inactive[worst];
      fp->inactive[worst] = temp;
      fp->beta[worst] = DBL_MAX;
    }
    else{
      adjustGradF(fp, ds, sp, n, worst, change, 0, flag, params, diff);
      fp->alpha[fp->inactive[worst]] -= diff;
      fp->alpha[fp->active[n]] = sp->C;
      fp->active[n] = fp->inactive[worst];
      fp->inactive[worst] = temp;
      fp->beta[worst] = DBL_MAX;
    }
  }
  else
  {
 if ( (fp->inactive[worst] < ds->procPos && target > 0) || (fp->inactive[worst] >= ds->procPos && target < 0) ) {
      adjustGradF(fp, ds, sp, n, worst, change, 1, flag, params, diff);
      fp->alpha[fp->inactive[worst]] += diff;
      fp->alpha[fp->active[n]] = 0.0;//sp->C ;
      fp->active[n] = fp->inactive[worst];
      fp->inactive[worst] = temp;
      fp->beta[worst] = DBL_MAX-1.0;
    }
    else {
      adjustGradF(fp, ds, sp, n, worst, change, 0, flag, params, diff);
      fp->alpha[fp->inactive[worst]] -= diff;
      fp->alpha[fp->active[n]] = 0.0;//sp->C ;
      fp->active[n] = fp->inactive[worst];
      fp->inactive[worst] = temp;
      fp->beta[worst] = DBL_MAX;
    }
  }


  return 1;
}

int checkfpConstraints(struct Fullproblem *fp)
{
  for (int i = 0; i < fp->p; i++) {
    if(fp->alpha[fp->active[i]]>fp->C){
      return i+fp->p;
    }
    else if(fp->alpha[fp->active[i]] < 0.0){
      return i-fp->p;
    }
  }
  return 0;
}

void YadjustGradF(struct Fullproblem *fp, struct yDenseData *ds, struct Projected *sp, int n, int worst, int signal, int target, int flag, struct svm_args *params, double diff)
{
  // updatee based on change of H matrix:
  double* nline = findListLine(fp->partialH,fp->active[n]);
  if (signal == -1) {
    #pragma omp parallel for
    for (int i = 0; i < fp->q; i++) {
      fp->gradF[ fp->inactive[i] ] -= nline[fp->inactive[i]]*diff ;
    }
    for (int i = 0; i < fp->p; i++) {
      if(i<n){
        fp->gradF[ fp->active[i] ] -= sp->H[i][n]*diff;
      }
      else{
        fp->gradF[ fp->active[i] ] -= sp->H[n][i]*diff;
      }
    }
  }
  else
  {
    #pragma omp parallel for
    for (int i = 0; i < fp->q; i++) {
      fp->gradF[ fp->inactive[i] ] += nline[fp->inactive[i]]*diff ;
    }
    for (int i = 0; i < fp->p; i++) {
      if(i<n){
        fp->gradF[ fp->active[i] ] += sp->H[i][n]*diff ;
      }
      else{
        fp->gradF[ fp->active[i] ] += sp->H[n][i]*diff ;
      }
    }
  }

  // Update based on change of

  YpartialHupdate(fp, sp, ds, params, n, worst);


  if (target) {
    for (int i = 0; i < fp->q; i++) {
      if (i == worst) {
        fp->gradF[fp->active[n]] -= nline[fp->active[n]]*diff;
        continue;
      }
      fp->gradF[fp->inactive[i]] -= nline[fp->inactive[i]]*diff;
    }
    for (int i = 0; i < n; i++) {
      fp->gradF[fp->active[i]] -= sp->H[i][n]*diff;
    }
    fp->gradF[fp->inactive[worst]] -= sp->H[n][n]*diff;
    for (int i = n+1; i < sp->p; i++) {
      fp->gradF[fp->active[i]] -= sp->H[n][i]*diff;
    }
  }
  else{
    #pragma omp parallel for
    for (int i = 0; i < fp->q; i++) {
      if (i == worst) {
        fp->gradF[fp->active[n]] += nline[fp->active[n]]*diff;
        continue;
      }
      fp->gradF[fp->inactive[i]] += nline[fp->inactive[i]]*diff;
    }
    for (int i = 0; i < n; i++) {
      fp->gradF[fp->active[i]] += sp->H[i][n]*diff;
    }
    fp->gradF[fp->inactive[worst]] += sp->H[n][n]*diff;
    for (int i = n+1; i < sp->p; i++) {
      fp->gradF[fp->active[i]] += sp->H[n][i]*diff;
    }
  }
}


void adjustGradF(struct Fullproblem *fp, struct denseData *ds, struct Projected *sp, int n, int worst, int signal, int target, int flag, struct svm_args *params, double diff)
{
  // updatee based on change of H matrix:
  double* nline = findListLine(fp->partialH,fp->active[n]);
  if (signal == -1) {
    for (int i = 0; i < fp->q; i++) {
      fp->gradF[ fp->inactive[i] ] -= nline[fp->inactive[i]]*diff ;
    }
    for (int i = 0; i < fp->p; i++) {
      if(i<n){
        fp->gradF[ fp->active[i] ] -= sp->H[i][n]*diff;
      }
      else{
        fp->gradF[ fp->active[i] ] -= sp->H[n][i]*diff;
      }
    }
  }
  else
  {
    #pragma omp parallel for
    for (int i = 0; i < fp->q; i++) {
      fp->gradF[ fp->inactive[i] ] += nline[fp->inactive[i]]*diff ;
    }
    for (int i = 0; i < fp->p; i++) {
      if(i<n){
        fp->gradF[ fp->active[i] ] += sp->H[i][n]*diff ;
      }
      else{
        fp->gradF[ fp->active[i] ] += sp->H[n][i]*diff ;
      }
    }
  }

  // Update based on change of

  partialHupdate(fp, sp, ds, params, n, worst);


  if (target) {
    for (int i = 0; i < fp->q; i++) {
      if (i == worst) {
        fp->gradF[fp->active[n]] -= nline[fp->active[n]]*diff;
        continue;
      }
      fp->gradF[fp->inactive[i]] -= nline[fp->inactive[i]]*diff;
    }
    for (int i = 0; i < n; i++) {
      fp->gradF[fp->active[i]] -= sp->H[i][n]*diff;
    }
    fp->gradF[fp->inactive[worst]] -= sp->H[n][n]*diff;
    for (int i = n+1; i < sp->p; i++) {
      fp->gradF[fp->active[i]] -= sp->H[n][i]*diff;
    }
  }
  else{
    for (int i = 0; i < fp->q; i++) {
      if (i == worst) {
        fp->gradF[fp->active[n]] += nline[fp->active[n]]*diff;
        continue;
      }
      fp->gradF[fp->inactive[i]] += nline[fp->inactive[i]]*diff;
    }
    for (int i = 0; i < n; i++) {
      fp->gradF[fp->active[i]] += sp->H[i][n]*diff;
    }
    fp->gradF[fp->inactive[worst]] += sp->H[n][n]*diff;
    for (int i = n+1; i < sp->p; i++) {
      fp->gradF[fp->active[i]] += sp->H[n][i]*diff;
    }
  }
}

void Yreinitprob(struct yDenseData *ds, struct Fullproblem *fp, struct Projected *sp, int add, int* temp, int* temp2)
{
  for (int i = 0; i < add; i++) {
    fp->active[(fp->p - add) + i] = temp[i];
    fp->partialH = Yappend(ds,fp->partialH, fp->active[(fp->p - add) + i]);
  }

  int k = 0;
  for (int i = 0; i < fp->q; i++) {
    for (int j = 0; j < add; j++) {
      if (fp->inactive[i] == temp[j]) {
        fp->inactive[i] = temp2[k];
        k++;
        if (k == add) {
          return;
        }
      }
    }
  }


}


void reinitprob(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp, int add, int* temp, int* temp2)
{
  for (int i = 0; i < add; i++) {
    fp->active[(fp->p - add) + i] = temp[i];
    fp->partialH = append(ds,fp->partialH, fp->active[(fp->p - add) + i]);
  }
  int k = 0;
  for (int i = 0; i < fp->q; i++) {
    for (int j = 0; j < add; j++) {
      if (fp->inactive[i] == temp[j]) {
        fp->inactive[i] = temp2[k];
        k++;
        if (k == add) {
          return;
        }
      }
    }
  }
}

void shrinkSize( struct Fullproblem *fp, struct Projected *sp, int k)
{

  int temp = fp->active[k];
  for (int i = k; i < fp->p  - 1; i++) {
    fp->active[i] = fp->active[i+1];
  }
  fp->partialH = delete(temp, fp->partialH);


  fp->p--;
  fp->q++;

  fp->active = realloc(fp->active,sizeof(int)*fp->p);
  fp->inactive = realloc(fp->inactive,sizeof(int)*fp->q);
  fp->inactive[fp->q-1] = temp;
  fp->beta = realloc(fp->beta,sizeof(double)*fp->q);



  // Change projected problem struct:

  sp->p--;

  sp->alphaHat = realloc(sp->alphaHat,sizeof(double)*sp->p);
  sp->yHat = realloc(sp->yHat,sizeof(int)*sp->p);
  sp->rHat = realloc(sp->rHat,sizeof(double)*sp->p);
  sp->gamma = realloc(sp->gamma,sizeof(double)*sp->p);
  sp->rho = realloc(sp->rho,sizeof(double)*sp->p);
  sp->Hrho = realloc(sp->Hrho,sizeof(double)*sp->p);

  sp->H = realloc(sp->H,sizeof(double*)*sp->p);
  sp->h = realloc(sp->h,sizeof(double)*((sp->p*(sp->p+1))/2));

  int j = 0;
  for (int i = 0; i < sp->p; i++) {
    sp->H[i] = &(sp->h[j]);
    j+=(sp->p-i-1);
  }
  fp->partialH.head->prev = fp->partialH.head;
  int i = 0;
  while (fp->partialH.head->prev != NULL) {
    for (int j = i; j < sp->p; j++) {
      sp->H[i][j] = fp->partialH.head->prev->line[fp->active[j]];
    }
    i++;
    fp->partialH.head->prev = fp->partialH.head->prev->next;
  }
}

void changeP( struct Fullproblem *fp, struct Projected *sp, int add)
{
  fp->p += add;
  fp->q -= add;
  sp->p += add;

  fp->active = realloc(fp->active,sizeof(int)*fp->p);
  fp->inactive = realloc(fp->inactive,sizeof(int)*fp->q);
  fp->beta = realloc(fp->beta,sizeof(double)*fp->q);

  sp->alphaHat = realloc(sp->alphaHat,sizeof(double)*sp->p);
  sp->yHat = realloc(sp->yHat,sizeof(int)*sp->p);
  sp->rHat = realloc(sp->rHat,sizeof(double)*sp->p);
  sp->gamma = realloc(sp->gamma,sizeof(double)*sp->p);
  sp->rho = realloc(sp->rho,sizeof(double)*sp->p);
  sp->Hrho = realloc(sp->Hrho,sizeof(double)*sp->p);
  sp->H = realloc(sp->H,sizeof(double*)*sp->p);
  sp->h = realloc(sp->h,sizeof(double)*((sp->p*(sp->p+1))/2));

  int j = 0;
  for (int i = 0; i < sp->p; i++) {
    sp->H[i] = &(sp->h[j]);
    j+=(sp->p-i-1);
  }
}

int findWorstest(struct Fullproblem *fp , int add, int* temp, int* temp2)
{
  double *betaVal = malloc(sizeof(double)*add);
  for (int i = 0; i < add; i++) {
    betaVal[i] = DBL_MAX;
  }
  for (int i = 0; i < fp->q; i++)
  {
    for (int j = 0; j < add; j++)
    {
      if (fp->beta[i] < betaVal[j])
      {
        for (int k = add - 1; k > j ; k--)
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
      if (fp->inactive[fp->q - add + i] == temp[i]) {
        flag = 1;
      }
    }
    if (flag == 0) {
      temp2[k] = fp->inactive[fp->q - add + i];
      k++;
    }
  }



  return add;
}

void  freeFullproblem(struct Fullproblem *fp)
/* Function to free dynamically allocated memory in Fullproblem struct */
{
  free(fp->alpha);
  free(fp->beta);
  free(fp->gradF);

  free(fp->active);
  free(fp->inactive);

  free_list(fp->partialH);
}
