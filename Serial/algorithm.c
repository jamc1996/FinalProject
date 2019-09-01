#include "algorithm.h"

/*      algorithm.c -- program with functions for top level implementation of
 *                      the algorithm.
 *
 *      Author:     John Cormican
 *
 *      Purpouse:   To manage the top level running of the serial algorithm.
 *
 *      Usage:      Call run_algorithm() to completely train an SVM.
 *
 */


int run_algorithm(struct denseData *ds, struct Fullproblem *fp, struct Projected *sp)
/*  Function to run the full serial algorithm for svm training using the
 *  conjugate gradient method.
 */

{
  // Full problem allocated and filled in, all alpha = 0.0 all gradF = 1.0:
  int p = 6;
  alloc_prob(fp, ds, p);
  init_prob(fp, ds);

  // Subproblem allocated:
  alloc_subprob(sp, p);

  // We loop until no negative entries in beta:
  int k = 1;
  int max_iters = 10000000;
  int itt = 0;
  int n = 0;

 // setH(&fp, &ds, &parameters);

  while(k){
    // H matrix columns re-set and subproblem changed
    if(itt%10000 == 0){
      printf("itt = %d\n",itt );
    }

    init_subprob(sp, fp, ds, &parameters, 1);

    //  congjugate gradient algorithm
    //  if algorithm completes n == 0
    //  if algorithm interrupt n != 0
    n = cg(sp, fp);

    updateAlphaR(fp, sp);
    calcYTR(sp, fp);
    calculateBeta(fp, sp, ds);

    if (n==0) {
      // Successful completion of the cg algorithm:
      // If elements with beta<0 add them to problem.
      // Else algorithm has completed.
      int add = 2;
      int *temp = malloc(sizeof(int)*add);
      int *temp2 = malloc(sizeof(int)*add);
      add = findWorstAdd(fp,add,temp,temp2);

      if (add == 0){
        free(temp);
        free(temp2);
        break;
      }

      // Problem reallocated and full problem re-inited.
      changeP(fp, sp, add);
      reinitprob(ds, fp, sp, add, temp, temp2);

      free(temp);
      free(temp2);
    }

    if (n) {
      // BCs broken. If possible swap, otherwise shrink problem size.
      k = singleswap(ds, fp, sp, n, &parameters);
      if (k < 0) {
        shrinkSize(fp, sp, k+fp->p);
      }
      else{
        n = checkfpConstraints(fp);
      }
    }

    //If we reach max_iters without convergence, report the error.
    itt++;
    if(itt == max_iters){
      fprintf(stderr, "algorithm.c run_algorithm(): maximum iterations (%d) with no convergence.\n",itt );
      return 1;
    }

  }
  return 0;
}


void freeDenseData(struct denseData *ds)
/* Function to free dynamically allocated memory in dense data set struct. */
{
  free(ds->data);
  free(ds->data1d);
}
