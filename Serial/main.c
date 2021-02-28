#include "svm.h"
#include "io.h"
#include "algorithm.h"

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) {
  char* filename = NULL;

  struct denseData ds;
  struct Fullproblem fp;
  struct Projected sp;

  struct timeval start, trainStart, trainEnd, end;

  gettimeofday(&start, 0);

  // Input processed:
  parse_arguments(argc, argv, &filename);
  read_file(filename, &ds);

  //  preprocess(&ds);
  if(parameters.test){
    testSavedModel(&ds, parameters.modelfile);
    return 0;
  }

  gettimeofday(&trainStart, 0);
  run_algorithm(&ds, &fp, &sp);
  gettimeofday(&trainEnd, 0);

  //Model saved to a txt file.
  if (parameters.save) {
    saveTrainedModel(&fp, &ds, sp.ytr);
  }

  gettimeofday(&end, 0);
  long totalelapsed = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec-start.tv_usec;
  long trainelapsed = (trainEnd.tv_sec-trainStart.tv_sec)*1000000 + trainEnd.tv_usec-trainStart.tv_usec;

  //Timing output
  printf("Training Complete\n" );
  if(totalelapsed>1000000){
    printf("Total Time spent: %lf seconds.\n",(double)totalelapsed/(double)1000000.0 );
    printf("Time Spent Training: %lf seconds.\n",(double)trainelapsed/(double)1000000.0 );
  }else{
    printf("Total Time spent: %ld micro seconds.\n",totalelapsed );
    printf("Time Spent Training: %ld micro seconds.\n",trainelapsed );
  }


  //Memory freed and program exits
  freeDenseData(&ds);
  freeFullproblem(&fp);
  freeSubProblem(&sp);
  return 0;
}
