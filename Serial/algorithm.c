#include "algorithm.h"

/*      algorithm.c -- program with functions for top level implementation of
 *                      the algorithm.
 *
 *      Author:     John Cormican
 *
 *      Purpouse:   To manage the top level running of the serial algorithm.
 *
 *      Usage:      Call runAlgorithm() to completely train an SVM.
 *
 */


int runAlgorithm(struct denseData *fullDataset, struct Fullproblem *alphOptProblem, struct Projected *projectedSubProblem)
/*  Function to run the full serial algorithm for svm training using the
 *  conjugate gradient method.
 */

{
  int p = 6;

  // Full problem allocated and 
  allocProb(alphOptProblem, fullDataset, p);

  // alpha = 0.0, gradF = 1.0:
  initProb(alphOptProblem, fullDataset);

  // Subproblem allocated:
  allocProjectedProblem(projectedSubProblem, p);

  // We loop until no negative entries in beta:
  int k = 1;
  int max_iters = 10000000;
  int itt = 0;
  int n = 0;

 // setH(&alphOptProblem, &fullDataset, &parameters);

  while(k){
    // H matrix columns re-set and subproblem changed
    if(itt%10000 == 0 && itt>0){
      printf("itt = %d\n",itt );
    }

    initSubprob(projectedSubProblem, alphOptProblem, fullDataset, &parameters, 1);

    //  congjugate gradient algorithm
    //  if algorithm completes n == 0
    //  if algorithm interrupt n != 0
    n = runConjGradient(projectedSubProblem, alphOptProblem);

    updateAlphaR(alphOptProblem, projectedSubProblem);
    calcYTR(projectedSubProblem, alphOptProblem);
    calculateBeta(alphOptProblem, projectedSubProblem, fullDataset);

    if (n==0) {
      // Successful completion of the runConjGradient algorithm:
      // If elements with beta<0 add them to problem.
      // Else algorithm has completed.
      int add = 2;
      int *temp = malloc(sizeof(int)*add);
      int *temp2 = malloc(sizeof(int)*add);
      add = findWorstAdd(alphOptProblem,add,temp,temp2);

      if (add == 0){
        free(temp);
        free(temp2);
        break;
      }

      // Problem reallocated and full problem re-inited.
      changeP(alphOptProblem, projectedSubProblem, add);
      reinitprob(fullDataset, alphOptProblem, projectedSubProblem, add, temp, temp2);

      free(temp);
      free(temp2);
    }

    if (n) {
      // BCs broken. If possible swap, otherwise shrink problem size.
      k = singleswap(fullDataset, alphOptProblem, projectedSubProblem, n, &parameters);
      if (k < 0) {
        shrinkSize(alphOptProblem, projectedSubProblem, k+alphOptProblem->projectedProblemSize);
      }
      else{
        n = checkfpConstraints(alphOptProblem);
      }
    }

    //If we reach max_iters without convergence, report the error.
    itt++;
    if(itt == max_iters){
      fprintf(stderr, "algorithm.c runAlgorithm(): maximum iterations (%d) with no convergence.\n",itt );
      return 1;
    }

  }
  return 0;
}


void freeDenseData(struct denseData *fullDataset)
/* Function to free dynamically allocated memory in dense data set struct. */
{
  free(fullDataset->data);
  free(fullDataset->data1d);
}
