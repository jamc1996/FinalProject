#include "io.h"

/*      io.c -- program with functions for reading a .txt file aswell as
 *               storing/testing trained models
 *
 *      Author:     John Cormican
 *
 *      Purpouse:   To manage the input and output of the program.
 *
 *      Usage:      read_file() called from main(). save_trained_model or
 *                    test_trained_model may also be used.
 *
 */

void read_file(char* filename, struct denseData* ds)
/* Function to read an appropriately formatted text file and store the
 * information in denseData struct.
 */
{
  // File opened
  FILE *fp = fopen(filename, "r");
  if (fp == NULL){
    fprintf(stderr, "gert: io.c: read_file() - file %s not found\n",filename );
    exit(1);
  }

  // Dimensions of the data found
  count_entries(fp, ds);

  //Space allocated
  ds->data1d = malloc(sizeof(double)*ds->nInstances*ds->nFeatures);
  ds->data = malloc(sizeof(double*)*ds->nInstances);

  double* temp;
  char* line = NULL;
  char* endptr;

  //Iterate through the data
  for (int i = 0; i < ds->nInstances; i++) {
    ds->data[i] = &ds->data1d[i*ds->nFeatures];
  }
  int r = 0;
  int q = ds->nPos;
  for (int i = 0; i < ds->nInstances; i++) {
    readline(fp,&line);
    char* p = strtok(line, " \t");
    if (p == NULL || *p == '\n') {
      fprintf(stderr, "main.cpp: read_file(): bad read at %d\n",i );
      exit(1);
    }
    int numP = atoi(p);
    if(numP == 1){
      temp = ds->data[r];
      r++;
    }else if (numP == -1){
      temp = ds->data[q];
      q++;
    }

    for (int j = 0; j < ds->nFeatures; j++) {
      char* p = strtok(NULL, " \t");
      if (p == NULL || *p == '\n') {
        fprintf(stderr, "main.cpp: read_file(): bad read at line %d\n",i );
        exit(1);
      }
      temp[j] = strtod(p, &endptr);
    }
  }


  fclose(fp);
}

void count_entries(FILE *input, struct denseData* ds)
/*  Function to find the dimensions of the input data and store them
 *  useful information is ds.
 */
{
  ds->nInstances = 0;
  ds->nFeatures = -1;
  ds->nPos = 0;
  ds->nNeg = 0;
  char* line = NULL;

  int counter = 0;
  // Find size of dataset:
  while (readline(input, &line)) {
    //Find number of features from first line
    if (ds->nFeatures==-1) {
      ds->nFeatures++;
      char *p = strtok(line," \t");
      while (1) {
        p  = strtok(NULL, " \t");
        if (p == NULL || *p == '\n') {
          break;
        }
        ds->nFeatures++;
      }
      rewind(input);
      continue;
    }

    //Now find nInstances as well as positive/negative divide.
    char* p = strtok(line," \t");
    counter++;
    int num = atoi(p);
    if (num == 1) {
      ds->nPos++;
    }else if(num == -1){
      ds->nNeg++;
    }else{
      fprintf(stderr, "invalid classes (should be 1 or -1, %d found at line %d)\n",num,counter);
      exit(1);
    }
    ds->nInstances++;
  }

  rewind(input);

}

void saveTrainedModel(struct Fullproblem *fp, struct denseData *ds, double ytr)
{
  FILE *file = fopen(parameters.savename, "w");

  fprintf(file, "%d\n",parameters.kernel );
  fprintf(file, "%d\n",ds->nFeatures );
  int missed = 0;
  for (int i = 0; i < fp->q; i++) {
    if (fp->alpha[fp->inactive[i]] > 0) {
      missed++;
    }
  }
  int *missedInds = malloc(sizeof(int)*missed);
  int j = 0;
  for (int i = 0; i < fp->q; i++) {
    if (fp->alpha[fp->inactive[i]] > 0.0) {
      missedInds[j] = fp->inactive[i];
      j++;
    }
  }

  if (parameters.kernel == LINEAR) {
    double *w = malloc(sizeof(double)*ds->nFeatures);

    for (int i = 0; i < ds->nFeatures; i++) {
      w[i] = 0.0;
      for (int j = 0; j < fp->p; j++) {
				if(fp->active[j] < ds->nPos){
	        w[i] += fp->alpha[fp->active[j]]*ds->data[fp->active[j]][i];
				}
				else{
	        w[i] -= fp->alpha[fp->active[j]]*ds->data[fp->active[j]][i];
				}
      }
      for (int j = 0; j < missed; j++) {
        if(missedInds[j] < ds->nPos){
          w[i] += ds->data[missedInds[j]][i]*fp->C;
        }
        else{
          w[i] -= ds->data[missedInds[j]][i]*fp->C;
        }
      }
    }
    printf("%lf\n",ytr );
    fprintf(file, "%lf\n",ytr );
    for (int i = 0; i < ds->nFeatures; i++) {
      fprintf(file, "%lf\n",w[i] );
    }
    free(w);

  }
  else if(parameters.kernel == POLYNOMIAL){
    fprintf(file, "%d\n", fp->p + missed );

	  fprintf(file, "%lf\n",ytr );

    for (int i = 0; i < fp->p; i++) {
      for (int j = 0; j < ds->nFeatures; j++) {
        fprintf(file, "%lf\n",ds->data[fp->active[i]][j] );
      }
    }
    for (int i = 0; i < missed; i++) {
      for (int j = 0; j < ds->nFeatures; j++) {
        fprintf(file, "%lf\n",ds->data[missedInds[i]][j] );
      }
    }

    for (int i = 0; i < fp->p; i++) {
			if(fp->active[i] < ds->nPos){
	      fprintf(file, "%lf\n",fp->alpha[fp->active[i]] );
  	  }
			else{
	      fprintf(file, "%lf\n",-fp->alpha[fp->active[i]] );
			}
  	}
    for (int i = 0; i < missed; i++) {
      if(missedInds[i] < ds->nPos){
        fprintf(file, "%lf\n",fp->C );
      }
      else{
        fprintf(file, "%lf\n",-fp->C );
      }
    }
	}
  else if(parameters.kernel == EXPONENTIAL)
  {
    fprintf(file, "%d\n", fp->p + missed );

    fprintf(file, "%lf\n",ytr );

    for (int i = 0; i < fp->p; i++) {
      for (int j = 0; j < ds->nFeatures; j++) {
        fprintf(file, "%lf\n",ds->data[fp->active[i]][j] );
      }
    }
    for (int i = 0; i < missed; i++) {
      for (int j = 0; j < ds->nFeatures; j++) {
        fprintf(file, "%lf\n",ds->data[missedInds[i]][j] );
      }
    }

    for (int i = 0; i < fp->p; i++) {
			if(fp->active[i] < ds->nPos){
	      fprintf(file, "%lf\n",fp->alpha[fp->active[i]] );
  	  }
			else{
	      fprintf(file, "%lf\n",-fp->alpha[fp->active[i]] );
  	  }
  	}
    for (int i = 0; i < missed; i++) {
      if(missedInds[i] < ds->nPos){
        fprintf(file, "%lf\n",fp->C );
      }
      else{
        fprintf(file, "%lf\n",-fp->C );
      }
    }
	}
  free(missedInds);
  fclose(file);
}

void testSavedModel(struct denseData *ds, char* fn)
{
  FILE *fp = fopen(fn, "r");
  int k;
  double b;
  int kernel;

  int res = fscanf(fp, "%d",&kernel);
  res = fscanf(fp, "%d",&k);

  if (k!= ds->nFeatures) {
    fprintf(stderr, "io.c: \n" );
    exit(1);
  }
  int wrong = 0;

  if (kernel == LINEAR) {
    res = fscanf(fp, "%lf", &b);

    double *w = malloc(sizeof(double)*k);
    for (size_t i = 0; i < k; i++) {
      res = fscanf(fp, "%lf",&w[i]);
    }
    double value;
    for (int i = 0; i < ds->nInstances; i++) {
      value = b;
      for (int j = 0; j < ds->nFeatures; j++) {
        value += w[j]*ds->data[i][j];
      }
			if(i<ds->nPos){
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
    res = fscanf(fp, "%d", &count);
    res = fscanf(fp, "%lf", &b);
    double value;
    double *alphaY = malloc(sizeof(double)*count);
    double *x = malloc(sizeof(double)*count*k);
    double **X = malloc(sizeof(double*)*count);
    for (int i = 0; i < count; i++) {
      X[i] = &x[i*k];
    }
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < k; j++) {
        res = fscanf(fp, "%lf", &X[i][j]);
      }
    }
    for (int i = 0; i < count; i++) {
      res = fscanf(fp, "%lf", &alphaY[i]);
    }
    double contrib = 0;
    for (int i = 0; i < ds->nInstances; i++) {
      value = b;
      for (int j = 0; j < count; j++) {
        contrib = 0;
        for (int k = 0; k < ds->nFeatures; k++) {
          contrib += X[j][k]*ds->data[i][k];
        }
        contrib = pow(contrib + parameters.Gamma, parameters.degree);
        value += alphaY[j]*contrib;
      }
			if(i<ds->nPos){
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
    res = fscanf(fp, "%d", &count);
    res = fscanf(fp, "%lf", &b);
    double value;
    double *alphaY = malloc(sizeof(double)*count);
    double *x = malloc(sizeof(double)*count*k);
    double **X = malloc(sizeof(double*)*count);
    for (int i = 0; i < count; i++) {
      X[i] = &x[i*k];
    }
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < k; j++) {
        res = fscanf(fp, "%lf", &X[i][j]);
      }
    }
    for (int i = 0; i < count; i++) {
      res = fscanf(fp, "%lf", &alphaY[i]);
    }
    if (res != 0){
		printf("res\n");
	}
    double contrib = 0;
    double y;
    for (int i = 0; i < ds->nInstances; i++) {
      value = b;
      for (int j = 0; j < count; j++) {
        contrib = 0;
        for (int k = 0; k < ds->nFeatures; k++) {
          y = X[j][k] - ds->data[i][k];
          contrib -= y*y;
        }
        contrib *= parameters.Gamma;
        value += alphaY[j]*exp(contrib);
      }
			if(i<ds->nPos){
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

  int right = ds->nInstances - wrong;
  double pct = 100.0*((double)right/(double)ds->nInstances);
  printf("%d correct classifications out of %d. %.2lf%% correct.\n",right,ds->nInstances,pct );

  fclose(fp);

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

int parse_arguments(int argc, char *argv[], char** filename)
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
    printf("io.c: parse_arguments(): no input file selected.\n");
    exit(1);
  }
  return 0;
}

void preprocess(struct denseData *ds)
/*  Function provides an option to normalise the input data.
 */
{
  double* means = (double*)calloc(ds->nFeatures,sizeof(double));
  double* stdDev = (double*)calloc(ds->nFeatures,sizeof(double));

  calcMeans(means, ds);
  calcStdDev(stdDev,means,ds);
  normalise(means,stdDev,ds);
  free(means);
  free(stdDev);
}


void calcMeans(double *mean, struct denseData *ds)
/*  Function to calculate the mean of each feature of the input data if
 *  normalisation required.
 */
{
  for (int i = 0; i < ds->nInstances; i++) {
    for (int j = 0; j < ds->nFeatures; j++) {
      mean[j] += ds->data[i][j];
    }
  }
  for (int i = 0; i < ds->nFeatures; i++) {
    mean[i]/=(double)(ds->nInstances);
  }
}

void normalise(double* mean, double* stdDev, struct denseData* ds)
/*  Function to normalise the data. */
{
  for (int i = 0; i < ds->nInstances; i++) {
    for (int j = 0; j < ds->nFeatures; j++) {
      ds->data[i][j]-=mean[j];
      ds->data[i][j]/=stdDev[j];
    }
  }
}

void calcStdDev(double* stdDev, double* mean, struct denseData *ds)
/* Function to calculate the standard deviation of each feature in ds. */
{
  for (int i = 0; i < ds->nInstances; i++) {
    for (int j = 0; j < ds->nFeatures; j++) {
      stdDev[j]+=(ds->data[i][j]-mean[j])*(ds->data[i][j]-mean[j]);
    }
  }
  for (int i = 0; i < ds->nFeatures; i++) {
    stdDev[i] = sqrt(stdDev[i]/((double)(ds->nInstances)-1.0));
  }
}

struct svmModel createFittedModel(double *w, int kernel, int trainElapsedTime, struct denseData *ds, struct Fullproblem *fp, double ytr){
  struct svmModel fittedModel;
  fittedModel.decisionVector = w;
  fittedModel.biasTerm = ytr;
  fittedModel.trainElapsedTime = trainElapsedTime;
  fittedModel.kernel = kernel;
  fittedModel.nFeatures = ds->nFeatures;
  int missed = 0;
  for (int i = 0; i < fp->q; i++) {
    if (fp->alpha[fp->inactive[i]] > 0) {
      missed++;
    }
  }
  int *missedInds = malloc(sizeof(int)*missed);
  int j = 0;
  for (int i = 0; i < fp->q; i++) {
    if (fp->alpha[fp->inactive[i]] > 0.0) {
      missedInds[j] = fp->inactive[i];
      j++;
    }
  }

  if (kernel == LINEAR) {
    for (int i = 0; i < ds->nFeatures; i++) {
      fittedModel.decisionVector[i] = 0.0;
      for (int j = 0; j < fp->p; j++) {
				if(fp->active[j] < ds->nPos){
	        fittedModel.decisionVector[i] += fp->alpha[fp->active[j]]*ds->data[fp->active[j]][i];
				}
				else{
	        fittedModel.decisionVector[i] -= fp->alpha[fp->active[j]]*ds->data[fp->active[j]][i];
				}
      }
      for (int j = 0; j < missed; j++) {
        if(missedInds[j] < ds->nPos){
          fittedModel.decisionVector[i] += ds->data[missedInds[j]][i]*fp->C;
        }
        else{
          fittedModel.decisionVector[i] -= ds->data[missedInds[j]][i]*fp->C;
        }
      }
    }
  }
  return fittedModel;
}


void saveTrainedModel2(
  struct Fullproblem *fp,
  struct denseData *ds,
  double ytr,
  const char* fileName
)
{
  FILE *file = fopen(fileName, "w");
  int kernel = LINEAR;
  fprintf(file, "%d\n", kernel );
  fprintf(file, "%d\n", ds->nFeatures );
  int missed = 0;
  for (int i = 0; i < fp->q; i++) {
    if (fp->alpha[fp->inactive[i]] > 0) {
      missed++;
    }
  }
  int *missedInds = malloc(sizeof(int)*missed);
  int j = 0;
  for (int i = 0; i < fp->q; i++) {
    if (fp->alpha[fp->inactive[i]] > 0.0) {
      missedInds[j] = fp->inactive[i];
      j++;
    }
  }

  if (kernel == LINEAR) {
    double *w = malloc(sizeof(double)*ds->nFeatures);

    for (int i = 0; i < ds->nFeatures; i++) {
      w[i] = 0.0;
      for (int j = 0; j < fp->p; j++) {
				if(fp->active[j] < ds->nPos){
	        w[i] += fp->alpha[fp->active[j]]*ds->data[fp->active[j]][i];
				}
				else{
	        w[i] -= fp->alpha[fp->active[j]]*ds->data[fp->active[j]][i];
				}
      }
      for (int j = 0; j < missed; j++) {
        if(missedInds[j] < ds->nPos){
          w[i] += ds->data[missedInds[j]][i]*fp->C;
        }
        else{
          w[i] -= ds->data[missedInds[j]][i]*fp->C;
        }
      }
    }
    fprintf(file, "%lf\n",ytr );
    for (int i = 0; i < ds->nFeatures; i++) {
      fprintf(file, "%lf\n",w[i] );
    }
    free(w);

  }
  else if(kernel == POLYNOMIAL){
    fprintf(file, "%d\n", fp->p + missed );

	  fprintf(file, "%lf\n",ytr );

    for (int i = 0; i < fp->p; i++) {
      for (int j = 0; j < ds->nFeatures; j++) {
        fprintf(file, "%lf\n",ds->data[fp->active[i]][j] );
      }
    }
    for (int i = 0; i < missed; i++) {
      for (int j = 0; j < ds->nFeatures; j++) {
        fprintf(file, "%lf\n",ds->data[missedInds[i]][j] );
      }
    }

    for (int i = 0; i < fp->p; i++) {
			if(fp->active[i] < ds->nPos){
	      fprintf(file, "%lf\n",fp->alpha[fp->active[i]] );
  	  }
			else{
	      fprintf(file, "%lf\n",-fp->alpha[fp->active[i]] );
			}
  	}
    for (int i = 0; i < missed; i++) {
      if(missedInds[i] < ds->nPos){
        fprintf(file, "%lf\n",fp->C );
      }
      else{
        fprintf(file, "%lf\n",-fp->C );
      }
    }
	}
  else if(kernel == EXPONENTIAL)
  {
    fprintf(file, "%d\n", fp->p + missed );

    fprintf(file, "%lf\n",ytr );

    for (int i = 0; i < fp->p; i++) {
      for (int j = 0; j < ds->nFeatures; j++) {
        fprintf(file, "%lf\n",ds->data[fp->active[i]][j] );
      }
    }
    for (int i = 0; i < missed; i++) {
      for (int j = 0; j < ds->nFeatures; j++) {
        fprintf(file, "%lf\n",ds->data[missedInds[i]][j] );
      }
    }

    for (int i = 0; i < fp->p; i++) {
			if(fp->active[i] < ds->nPos){
	      fprintf(file, "%lf\n",fp->alpha[fp->active[i]] );
  	  }
			else{
	      fprintf(file, "%lf\n",-fp->alpha[fp->active[i]] );
  	  }
  	}
    for (int i = 0; i < missed; i++) {
      if(missedInds[i] < ds->nPos){
        fprintf(file, "%lf\n",fp->C );
      }
      else{
        fprintf(file, "%lf\n",-fp->C );
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

void setUpDense(struct denseData *ds, double** trainData, int nFeatures, int nInstances, int nPos){
  ds->nInstances = nInstances;
  ds->nFeatures = nFeatures;
  ds->nNeg = nInstances - nPos;
  ds->nPos = nPos;
  ds->data = trainData;
  ds->data1d = ds->data[0];
}

