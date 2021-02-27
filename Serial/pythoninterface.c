#include "svm.h"
#include "io.h"
#include "algorithm.h"

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

struct svmModel fit(
  double** trainData,
  int nFeatures,
  int nInstances,
  int nPos,
  int saveToFile,
  const char* fileName,
  double* decisionVectorToFill)
/* Function to train 

Returns time spent training in c.
*/
{
  struct denseData ds;
  struct Fullproblem fp;
  struct Projected sp;
  struct timeval trainStart, trainEnd;
  int kernel = LINEAR;
  change_params(&parameters);
  gettimeofday(&trainStart, 0);
  setUpDense(&ds, trainData, nFeatures, nInstances, nPos);
  run_algorithm(&ds, &fp, &sp);
  gettimeofday(&trainEnd, 0);

  long trainElapsedTime = (trainEnd.tv_sec-trainStart.tv_sec)*1000000 + trainEnd.tv_usec-trainStart.tv_usec;
  // printf("Total Time spent: %ld micro seconds.\n", totalelapsed);

  //Model saved to a txt file.
  if (saveToFile) {
    saveTrainedModel2(&fp, &ds, sp.ytr, fileName);
  }

  struct svmModel fittedModel = createFittedModel(
    decisionVectorToFill,
    kernel,
    trainElapsedTime,
    &ds,
    &fp,
    sp.ytr
  );

  // //Memory freed and program exits
  freeFullproblem(&fp);
  freeSubProblem(&sp);

  return fittedModel;
}

void transform(double** testData, int nInstances, struct svmModel fittedModel, double* output){
  for (int i = 0; i < nInstances; i++) {
    output[i] = fittedModel.biasTerm;
    for (int j = 0; j < fittedModel.nFeatures; j++) {
      output[i] += fittedModel.decisionVector[j]*testData[i][j];
    }
  }
}

int main(){
    return 0;
}