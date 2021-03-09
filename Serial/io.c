#include "io.h"

/*      io.c -- program with functions for reading a .txt file aswell as
 *               storing/testing trained models
 *
 *      Author:     John Cormican
 *
 *      Purpouse:   To manage the input and output of the program.
 *
 *      Usage:      readFile() called from main(). save_trained_model or
 *                    test_trained_model may also be used.
 *
 */

void readFile(char* filename, struct denseData* fullDataset)
/* Function to read an appropriately formatted text file and store the
 * information in denseData struct.
 */
{
  // File opened
  FILE *alphOptProblem = fopen(filename, "r");
  if (alphOptProblem == NULL){
    fprintf(stderr, "gert: io.c: readFile() - file %s not found\n",filename );
    exit(1);
  }

  // Dimensions of the data found
  count_entries(alphOptProblem, fullDataset);

  //Space allocated
  fullDataset->data1d = malloc(sizeof(double)*fullDataset->nInstances*fullDataset->nFeatures);
  fullDataset->data = malloc(sizeof(double*)*fullDataset->nInstances);

  double* temp;
  char* line = NULL;
  char* endptr;

  //Iterate through the data
  for (int i = 0; i < fullDataset->nInstances; i++) {
    fullDataset->data[i] = &fullDataset->data1d[i*fullDataset->nFeatures];
  }
  int r = 0;
  int q = fullDataset->nPos;
  for (int i = 0; i < fullDataset->nInstances; i++) {
    readline(alphOptProblem,&line);
    char* p = strtok(line, " \t");
    if (p == NULL || *p == '\n') {
      fprintf(stderr, "main.cpp: readFile(): bad read at %d\n",i );
      exit(1);
    }
    int numP = atoi(p);
    if(numP == 1){
      temp = fullDataset->data[r];
      r++;
    }else if (numP == -1){
      temp = fullDataset->data[q];
      q++;
    }

    for (int j = 0; j < fullDataset->nFeatures; j++) {
      char* p = strtok(NULL, " \t");
      if (p == NULL || *p == '\n') {
        fprintf(stderr, "main.cpp: readFile(): bad read at line %d\n",i );
        exit(1);
      }
      temp[j] = strtod(p, &endptr);
    }
  }


  fclose(alphOptProblem);
}

void count_entries(FILE *input, struct denseData* fullDataset)
/*  Function to find the dimensions of the input data and store them
 *  useful information is fullDataset.
 */
{
  fullDataset->nInstances = 0;
  fullDataset->nFeatures = -1;
  fullDataset->nPos = 0;
  fullDataset->nNeg = 0;
  char* line = NULL;

  int counter = 0;
  // Find size of dataset:
  while (readline(input, &line)) {
    //Find number of features from first line
    if (fullDataset->nFeatures==-1) {
      fullDataset->nFeatures++;
      char *p = strtok(line," \t");
      while (1) {
        p  = strtok(NULL, " \t");
        if (p == NULL || *p == '\n') {
          break;
        }
        fullDataset->nFeatures++;
      }
      rewind(input);
      continue;
    }

    //Now find nInstances as well as positive/negative divide.
    char* p = strtok(line," \t");
    counter++;
    int num = atoi(p);
    if (num == 1) {
      fullDataset->nPos++;
    }else if(num == -1){
      fullDataset->nNeg++;
    }else{
      fprintf(stderr, "invalid classes (should be 1 or -1, %d found at line %d)\n",num,counter);
      exit(1);
    }
    fullDataset->nInstances++;
  }

  rewind(input);

}

void saveTrainedModel(struct Fullproblem *alphOptProblem, struct denseData *fullDataset, double ytr)
{
  FILE *file = fopen(parameters.savename, "w");

  fprintf(file, "%d\n",parameters.kernel );
  fprintf(file, "%d\n",fullDataset->nFeatures );
  int missed = 0;
  for (int i = 0; i < alphOptProblem->q; i++) {
    if (alphOptProblem->alpha[alphOptProblem->inactive[i]] > 0) {
      missed++;
    }
  }
  int *missedInds = malloc(sizeof(int)*missed);
  int j = 0;
  for (int i = 0; i < alphOptProblem->q; i++) {
    if (alphOptProblem->alpha[alphOptProblem->inactive[i]] > 0.0) {
      missedInds[j] = alphOptProblem->inactive[i];
      j++;
    }
  }

  if (parameters.kernel == LINEAR) {
    double *w = malloc(sizeof(double)*fullDataset->nFeatures);

    for (int i = 0; i < fullDataset->nFeatures; i++) {
      w[i] = 0.0;
      for (int j = 0; j < alphOptProblem->projectedProblemSize; j++) {
				if(alphOptProblem->active[j] < fullDataset->nPos){
	        w[i] += alphOptProblem->alpha[alphOptProblem->active[j]]*fullDataset->data[alphOptProblem->active[j]][i];
				}
				else{
	        w[i] -= alphOptProblem->alpha[alphOptProblem->active[j]]*fullDataset->data[alphOptProblem->active[j]][i];
				}
      }
      for (int j = 0; j < missed; j++) {
        if(missedInds[j] < fullDataset->nPos){
          w[i] += fullDataset->data[missedInds[j]][i]*alphOptProblem->C;
        }
        else{
          w[i] -= fullDataset->data[missedInds[j]][i]*alphOptProblem->C;
        }
      }
    }
    printf("%lf\n",ytr );
    fprintf(file, "%lf\n",ytr );
    for (int i = 0; i < fullDataset->nFeatures; i++) {
      fprintf(file, "%lf\n",w[i] );
    }
    free(w);

  }
  else if(parameters.kernel == POLYNOMIAL){
    fprintf(file, "%d\n", alphOptProblem->projectedProblemSize + missed );

	  fprintf(file, "%lf\n",ytr );

    for (int i = 0; i < alphOptProblem->projectedProblemSize; i++) {
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        fprintf(file, "%lf\n",fullDataset->data[alphOptProblem->active[i]][j] );
      }
    }
    for (int i = 0; i < missed; i++) {
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        fprintf(file, "%lf\n",fullDataset->data[missedInds[i]][j] );
      }
    }

    for (int i = 0; i < alphOptProblem->projectedProblemSize; i++) {
			if(alphOptProblem->active[i] < fullDataset->nPos){
	      fprintf(file, "%lf\n",alphOptProblem->alpha[alphOptProblem->active[i]] );
  	  }
			else{
	      fprintf(file, "%lf\n",-alphOptProblem->alpha[alphOptProblem->active[i]] );
			}
  	}
    for (int i = 0; i < missed; i++) {
      if(missedInds[i] < fullDataset->nPos){
        fprintf(file, "%lf\n",alphOptProblem->C );
      }
      else{
        fprintf(file, "%lf\n",-alphOptProblem->C );
      }
    }
	}
  else if(parameters.kernel == EXPONENTIAL)
  {
    fprintf(file, "%d\n", alphOptProblem->projectedProblemSize + missed );

    fprintf(file, "%lf\n",ytr );

    for (int i = 0; i < alphOptProblem->projectedProblemSize; i++) {
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        fprintf(file, "%lf\n",fullDataset->data[alphOptProblem->active[i]][j] );
      }
    }
    for (int i = 0; i < missed; i++) {
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        fprintf(file, "%lf\n",fullDataset->data[missedInds[i]][j] );
      }
    }

    for (int i = 0; i < alphOptProblem->projectedProblemSize; i++) {
			if(alphOptProblem->active[i] < fullDataset->nPos){
	      fprintf(file, "%lf\n",alphOptProblem->alpha[alphOptProblem->active[i]] );
  	  }
			else{
	      fprintf(file, "%lf\n",-alphOptProblem->alpha[alphOptProblem->active[i]] );
  	  }
  	}
    for (int i = 0; i < missed; i++) {
      if(missedInds[i] < fullDataset->nPos){
        fprintf(file, "%lf\n",alphOptProblem->C );
      }
      else{
        fprintf(file, "%lf\n",-alphOptProblem->C );
      }
    }
	}
  free(missedInds);
  fclose(file);
}

void testSavedModel(struct denseData *fullDataset, char* fn)
{
  FILE *alphOptProblem = fopen(fn, "r");
  int k;
  double b;
  int kernel;

  int res = fscanf(alphOptProblem, "%d",&kernel);
  res = fscanf(alphOptProblem, "%d",&k);

  if (k!= fullDataset->nFeatures) {
    fprintf(stderr, "io.c: \n" );
    exit(1);
  }
  int wrong = 0;

  if (kernel == LINEAR) {
    res = fscanf(alphOptProblem, "%lf", &b);

    double *w = malloc(sizeof(double)*k);
    for (size_t i = 0; i < k; i++) {
      res = fscanf(alphOptProblem, "%lf",&w[i]);
    }
    double value;
    for (int i = 0; i < fullDataset->nInstances; i++) {
      value = b;
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        value += w[j]*fullDataset->data[i][j];
      }
			if(i<fullDataset->nPos){
	      if (value < 0.0) {
  	      printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
  	      wrong++;
  	    }
  	    else{
  	      printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
  	    }
    	}
			else{
	      if (value > 0.0) {
  	      printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
  	      wrong++;
  	    }
  	    else{
  	      printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
  	    }
    	}
		}
    free(w);
  }
  else if(parameters.kernel == POLYNOMIAL)  {
    int count;
    res = fscanf(alphOptProblem, "%d", &count);
    res = fscanf(alphOptProblem, "%lf", &b);
    double value;
    double *alphaY = malloc(sizeof(double)*count);
    double *x = malloc(sizeof(double)*count*k);
    double **X = malloc(sizeof(double*)*count);
    for (int i = 0; i < count; i++) {
      X[i] = &x[i*k];
    }
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < k; j++) {
        res = fscanf(alphOptProblem, "%lf", &X[i][j]);
      }
    }
    for (int i = 0; i < count; i++) {
      res = fscanf(alphOptProblem, "%lf", &alphaY[i]);
    }
    double contrib = 0;
    for (int i = 0; i < fullDataset->nInstances; i++) {
      value = b;
      for (int j = 0; j < count; j++) {
        contrib = 0;
        for (int k = 0; k < fullDataset->nFeatures; k++) {
          contrib += X[j][k]*fullDataset->data[i][k];
        }
        contrib = pow(contrib + parameters.Gamma, parameters.degree);
        value += alphaY[j]*contrib;
      }
			if(i<fullDataset->nPos){
	      if (value < 0.0) {
  	      printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
  	      wrong++;
  	    }
  	    else{
  	      printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
  	    }
    	}
			else{
	      if (value > 0.0) {
  	      printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
  	      wrong++;
  	    }
  	    else{
  	      printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
  	    }
    	}
    }
  }
  else if(parameters.kernel == EXPONENTIAL)  {
    int count;
    res = fscanf(alphOptProblem, "%d", &count);
    res = fscanf(alphOptProblem, "%lf", &b);
    double value;
    double *alphaY = malloc(sizeof(double)*count);
    double *x = malloc(sizeof(double)*count*k);
    double **X = malloc(sizeof(double*)*count);
    for (int i = 0; i < count; i++) {
      X[i] = &x[i*k];
    }
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < k; j++) {
        res = fscanf(alphOptProblem, "%lf", &X[i][j]);
      }
    }
    for (int i = 0; i < count; i++) {
      res = fscanf(alphOptProblem, "%lf", &alphaY[i]);
    }
    if (res != 0){
		printf("res\n");
	}
    double contrib = 0;
    double y;
    for (int i = 0; i < fullDataset->nInstances; i++) {
      value = b;
      for (int j = 0; j < count; j++) {
        contrib = 0;
        for (int k = 0; k < fullDataset->nFeatures; k++) {
          y = X[j][k] - fullDataset->data[i][k];
          contrib -= y*y;
        }
        contrib *= parameters.Gamma;
        value += alphaY[j]*exp(contrib);
      }
			if(i<fullDataset->nPos){
	      if (value < 0.0) {
  	      printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
  	      wrong++;
  	    }
  	    else{
  	      printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
  	    }
    	}
			else{
	      if (value > 0.0) {
  	      printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
  	      wrong++;
  	    }
  	    else{
  	      printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
  	    }
    	}
    }
  }

  int right = fullDataset->nInstances - wrong;
  double pct = 100.0*((double)right/(double)fullDataset->nInstances);
  printf("%d correct classifications out of %d. %.2lf%% correct.\n",right,fullDataset->nInstances,pct );

  fclose(alphOptProblem);

}
int readline(FILE *input, char **line)
/* Function to read lines from file. Returns 1 upon successful reading
 *  and 0 if read is unsuccessful/file ends.
 */
{
  int len;
  int max_line_len = 1024;
  *line = (char*)realloc(*line,sizeof(char)*max_line_len);
  if(fgets(*line,max_line_len,input) == NULL)
  {
    return 0;
  }

  while(strrchr(*line,'\n') == NULL)
  {
    max_line_len *= 2;
    *line = (char *) realloc(*line, max_line_len);
    len = (int) strlen(*line);
    if (fgets(*line+len,max_line_len-len,input) == NULL) {
      break;
    }
  }
  return 1;
}

int parseArguments(int argc, char *argv[], char** filename)
/* Function to parse command line arguments with getopt */
{
  int c;

  // Default values set:
  parameters.kernel = 0;
  parameters.degree = 1;
  parameters.verbose = 0;
  parameters.C = 1;
  parameters.test = 0;
  parameters.modelfile = NULL;
  parameters.save = 0;
  parameters.savename = NULL;
  parameters.Gamma = 1;
  while ((c = getopt( argc, argv, "f:k:t:c:d:vhs:g:")) != -1){
    switch (c) {
      case 'f':
        *filename = optarg;
        break;
      case 'k':
        parameters.kernel = atoi(optarg);
        break;
      case 'c':
        parameters.C = atof(optarg);
        break;
      case 't':
        parameters.test = 1;
        parameters.modelfile = optarg;
        break;
      case 'd':
        parameters.degree = atoi(optarg);
        break;
      case 's':
        parameters.save = 1;
        parameters.savename = optarg;
        break;
      case 'v':
        parameters.verbose = 1;
        break;
      case 'g':
        parameters.Gamma = atof(optarg);
        break;
      case 'h':
        printf("I was supposed to put in a help message here.\n");
        break;
    }
  }

  if (*filename == NULL) {
    printf("io.c: parseArguments(): no input file selected.\n");
    exit(1);
  }
  return 0;
}

void preprocess(struct denseData *fullDataset)
/*  Function provides an option to normalise the input data.
 */
{
  double* means = (double*)calloc(fullDataset->nFeatures,sizeof(double));
  double* stdDev = (double*)calloc(fullDataset->nFeatures,sizeof(double));

  calcMeans(means, fullDataset);
  calcStdDev(stdDev,means,fullDataset);
  normalise(means,stdDev,fullDataset);
  free(means);
  free(stdDev);
}


void calcMeans(double *mean, struct denseData *fullDataset)
/*  Function to calculate the mean of each feature of the input data if
 *  normalisation required.
 */
{
  for (int i = 0; i < fullDataset->nInstances; i++) {
    for (int j = 0; j < fullDataset->nFeatures; j++) {
      mean[j] += fullDataset->data[i][j];
    }
  }
  for (int i = 0; i < fullDataset->nFeatures; i++) {
    mean[i]/=(double)(fullDataset->nInstances);
  }
}

void normalise(double* mean, double* stdDev, struct denseData* fullDataset)
/*  Function to normalise the data. */
{
  for (int i = 0; i < fullDataset->nInstances; i++) {
    for (int j = 0; j < fullDataset->nFeatures; j++) {
      fullDataset->data[i][j]-=mean[j];
      fullDataset->data[i][j]/=stdDev[j];
    }
  }
}

void calcStdDev(double* stdDev, double* mean, struct denseData *fullDataset)
/* Function to calculate the standard deviation of each feature in fullDataset. */
{
  for (int i = 0; i < fullDataset->nInstances; i++) {
    for (int j = 0; j < fullDataset->nFeatures; j++) {
      stdDev[j]+=(fullDataset->data[i][j]-mean[j])*(fullDataset->data[i][j]-mean[j]);
    }
  }
  for (int i = 0; i < fullDataset->nFeatures; i++) {
    stdDev[i] = sqrt(stdDev[i]/((double)(fullDataset->nInstances)-1.0));
  }
}

struct svmModel createFittedModel(double *w, int kernel, int trainElapsedTime, struct denseData *fullDataset, struct Fullproblem *alphOptProblem, double ytr){
  struct svmModel fittedModel;
  fittedModel.decisionVector = w;
  fittedModel.biasTerm = ytr;
  fittedModel.trainElapsedTime = trainElapsedTime;
  fittedModel.kernel = kernel;
  fittedModel.nFeatures = fullDataset->nFeatures;
  int missed = 0;
  for (int i = 0; i < alphOptProblem->q; i++) {
    if (alphOptProblem->alpha[alphOptProblem->inactive[i]] > 0) {
      missed++;
    }
  }
  int *missedInds = malloc(sizeof(int)*missed);
  int j = 0;
  for (int i = 0; i < alphOptProblem->q; i++) {
    if (alphOptProblem->alpha[alphOptProblem->inactive[i]] > 0.0) {
      missedInds[j] = alphOptProblem->inactive[i];
      j++;
    }
  }

  if (kernel == LINEAR) {
    for (int i = 0; i < fullDataset->nFeatures; i++) {
      fittedModel.decisionVector[i] = 0.0;
      for (int j = 0; j < alphOptProblem->projectedProblemSize; j++) {
				if(alphOptProblem->active[j] < fullDataset->nPos){
	        fittedModel.decisionVector[i] += alphOptProblem->alpha[alphOptProblem->active[j]]*fullDataset->data[alphOptProblem->active[j]][i];
				}
				else{
	        fittedModel.decisionVector[i] -= alphOptProblem->alpha[alphOptProblem->active[j]]*fullDataset->data[alphOptProblem->active[j]][i];
				}
      }
      for (int j = 0; j < missed; j++) {
        if(missedInds[j] < fullDataset->nPos){
          fittedModel.decisionVector[i] += fullDataset->data[missedInds[j]][i]*alphOptProblem->C;
        }
        else{
          fittedModel.decisionVector[i] -= fullDataset->data[missedInds[j]][i]*alphOptProblem->C;
        }
      }
    }
  }
  return fittedModel;
}


void saveTrainedModel2(
  struct Fullproblem *alphOptProblem,
  struct denseData *fullDataset,
  double ytr,
  const char* fileName
)
{
  FILE *file = fopen(fileName, "w");
  int kernel = LINEAR;
  fprintf(file, "%d\n", kernel );
  fprintf(file, "%d\n", fullDataset->nFeatures );
  int missed = 0;
  for (int i = 0; i < alphOptProblem->q; i++) {
    if (alphOptProblem->alpha[alphOptProblem->inactive[i]] > 0) {
      missed++;
    }
  }
  int *missedInds = malloc(sizeof(int)*missed);
  int j = 0;
  for (int i = 0; i < alphOptProblem->q; i++) {
    if (alphOptProblem->alpha[alphOptProblem->inactive[i]] > 0.0) {
      missedInds[j] = alphOptProblem->inactive[i];
      j++;
    }
  }

  if (kernel == LINEAR) {
    double *w = malloc(sizeof(double)*fullDataset->nFeatures);

    for (int i = 0; i < fullDataset->nFeatures; i++) {
      w[i] = 0.0;
      for (int j = 0; j < alphOptProblem->projectedProblemSize; j++) {
				if(alphOptProblem->active[j] < fullDataset->nPos){
	        w[i] += alphOptProblem->alpha[alphOptProblem->active[j]]*fullDataset->data[alphOptProblem->active[j]][i];
				}
				else{
	        w[i] -= alphOptProblem->alpha[alphOptProblem->active[j]]*fullDataset->data[alphOptProblem->active[j]][i];
				}
      }
      for (int j = 0; j < missed; j++) {
        if(missedInds[j] < fullDataset->nPos){
          w[i] += fullDataset->data[missedInds[j]][i]*alphOptProblem->C;
        }
        else{
          w[i] -= fullDataset->data[missedInds[j]][i]*alphOptProblem->C;
        }
      }
    }
    fprintf(file, "%lf\n",ytr );
    for (int i = 0; i < fullDataset->nFeatures; i++) {
      fprintf(file, "%lf\n",w[i] );
    }
    free(w);

  }
  else if(kernel == POLYNOMIAL){
    fprintf(file, "%d\n", alphOptProblem->projectedProblemSize + missed );

	  fprintf(file, "%lf\n",ytr );

    for (int i = 0; i < alphOptProblem->projectedProblemSize; i++) {
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        fprintf(file, "%lf\n",fullDataset->data[alphOptProblem->active[i]][j] );
      }
    }
    for (int i = 0; i < missed; i++) {
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        fprintf(file, "%lf\n",fullDataset->data[missedInds[i]][j] );
      }
    }

    for (int i = 0; i < alphOptProblem->projectedProblemSize; i++) {
			if(alphOptProblem->active[i] < fullDataset->nPos){
	      fprintf(file, "%lf\n",alphOptProblem->alpha[alphOptProblem->active[i]] );
  	  }
			else{
	      fprintf(file, "%lf\n",-alphOptProblem->alpha[alphOptProblem->active[i]] );
			}
  	}
    for (int i = 0; i < missed; i++) {
      if(missedInds[i] < fullDataset->nPos){
        fprintf(file, "%lf\n",alphOptProblem->C );
      }
      else{
        fprintf(file, "%lf\n",-alphOptProblem->C );
      }
    }
	}
  else if(kernel == EXPONENTIAL)
  {
    fprintf(file, "%d\n", alphOptProblem->projectedProblemSize + missed );

    fprintf(file, "%lf\n",ytr );

    for (int i = 0; i < alphOptProblem->projectedProblemSize; i++) {
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        fprintf(file, "%lf\n",fullDataset->data[alphOptProblem->active[i]][j] );
      }
    }
    for (int i = 0; i < missed; i++) {
      for (int j = 0; j < fullDataset->nFeatures; j++) {
        fprintf(file, "%lf\n",fullDataset->data[missedInds[i]][j] );
      }
    }

    for (int i = 0; i < alphOptProblem->projectedProblemSize; i++) {
			if(alphOptProblem->active[i] < fullDataset->nPos){
	      fprintf(file, "%lf\n",alphOptProblem->alpha[alphOptProblem->active[i]] );
  	  }
			else{
	      fprintf(file, "%lf\n",-alphOptProblem->alpha[alphOptProblem->active[i]] );
  	  }
  	}
    for (int i = 0; i < missed; i++) {
      if(missedInds[i] < fullDataset->nPos){
        fprintf(file, "%lf\n",alphOptProblem->C );
      }
      else{
        fprintf(file, "%lf\n",-alphOptProblem->C );
      }
    }
	}
  free(missedInds);
  fclose(file);
}

void change_params(struct svm_args *parameters)
/* Function to parse command line arguments with getopt */
{
  // Default values set:
  parameters->kernel = 0;
  parameters->degree = 1;
  parameters->verbose = 0;
  parameters->C = 1;
  parameters->test = 0;
  parameters->modelfile = NULL;
  parameters->save = 0;
  parameters->savename = NULL;
  parameters->Gamma = 1;
}

void setUpDense(struct denseData *fullDataset, double** trainData, int nFeatures, int nInstances, int nPos){
  fullDataset->nInstances = nInstances;
  fullDataset->nFeatures = nFeatures;
  fullDataset->nNeg = nInstances - nPos;
  fullDataset->nPos = nPos;
  fullDataset->data = trainData;
  fullDataset->data1d = fullDataset->data[0];
}

