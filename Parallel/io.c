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

void divideSet(struct denseData *fullDataset, int nprocs, int myid)
/*Function to divide the dataset across multiple processors. */
{
	int allSz = fullDataset->nInstances/nprocs;
	int allMod = fullDataset->nInstances%nprocs;
	int posSz = fullDataset->nPos/nprocs;
	int posMod = fullDataset->nPos%nprocs;

	if(myid < allMod){
		fullDataset->procInstances = allSz + 1;
	}
	else{
		fullDataset->procInstances = allSz;
	}

	if(myid < posMod){
		fullDataset->procPos = posSz + 1;
		fullDataset->posStart = fullDataset->procPos*myid;
	}
	else{
		fullDataset->procPos = posSz;
		fullDataset->posStart = fullDataset->procPos*myid + posMod;
	}

	fullDataset->procNeg = fullDataset->procInstances - fullDataset->procPos;
	fullDataset->negStart = 0;

	for(int i = 0; i< myid; i++)
	{
		fullDataset->negStart += allSz - posSz;
		if( i < allMod )
		{
			fullDataset->negStart ++;
		}
		if (i < posMod)
		{
			fullDataset->negStart --;
		}
	}
}

void readFile(char* filename, struct denseData* fullDataset, int nprocs, int myid)
/* Function to read the data file.*/
{


  FILE *alphOptProblem = fopen(filename, "r");
  if (alphOptProblem == NULL){
    fprintf(stderr, "gert: io.c: readFile() - file %s not found\n",filename );
    exit(1);
  }
  count_entries(alphOptProblem, fullDataset);
  divideSet(fullDataset, nprocs, myid);

  fullDataset->data1d = malloc(sizeof(double)*fullDataset->procInstances*fullDataset->nFeatures);
  fullDataset->data = malloc(sizeof(double*)*fullDataset->procInstances);
  double* temp;

  char* line = NULL;
  char* endptr;
  for (int i = 0; i < fullDataset->procInstances; i++) {
    fullDataset->data[i] = &fullDataset->data1d[i*fullDataset->nFeatures];
  }
  int r = 0;
  int procR = 0;
  int q = 0;
	int procQ = 0;
  for (int i = 0; i < fullDataset->nInstances; i++) {
    readline(alphOptProblem,&line);
    char *p = strtok(line, " \t");
    int num = atoi(p);
//    if (fullDataset->instanceLabels[i] == NULL || *(fullDataset->instanceLabels[i]) == '\n') {
//      fprintf(stderr, "main.cpp: readFile(): bad read at %d\n",i );
//      exit(1);
//    }
    if(num == 1){
	if(r<fullDataset->posStart){
		r++;
		continue;
	}
	else if(procR == fullDataset->procPos){
		continue;
	}
	else {
		temp = fullDataset->data[procR];
		for (int j = 0; j < fullDataset->nFeatures; j++) {
      			char* p = strtok(NULL, " \t");
				if (p == NULL || *p == '\n') {
					fprintf(stderr, "Oh dear\n" );
					exit(1);
				}
				temp[j] = strtod(p, &endptr);
		}
		r++;
		procR++;
	}
    }
    else if (num == -1){
      	if(q<fullDataset->negStart){
		q++;
		continue;
	}
	else if(procQ == fullDataset->procNeg){
		continue;
	}
	else {
		temp = fullDataset->data[procQ + fullDataset->procPos];
		for (int j = 0; j < fullDataset->nFeatures; j++) {
      			char* p = strtok(NULL, " \t");
				if (p == NULL || *p == '\n') {
					fprintf(stderr, "Oh dear\n" );
					exit(1);
				}
				temp[j] = strtod(p, &endptr);
		}
		procQ++;
		q++;
	}
  }
}
}
void count_entries(FILE *input, struct denseData* fullDataset)
/* FUnction to count the entries of a data file.*/
{
  fullDataset->nInstances = 0;
  fullDataset->nFeatures = -1;
  fullDataset->nPos = 0;
  fullDataset->nNeg = 0;
  char* line = NULL;

  int counter = 0;
  // Find size of dataset:
  while (readline(input, &line)) {
    if (fullDataset->nFeatures==-1) {
	char* p = strtok(line, "\t");
	fullDataset->nFeatures++;
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
    char* p = strtok(line," \t");
    counter++;
    int num = atoi(p);
    if (num == 1) {
      fullDataset->nPos++;
    }else if(num == -1){
      fullDataset->nNeg++;
    }else{
      fprintf(stderr, "invalid classes (should be 1 or -1, %d found %d)\n",num,counter);
      exit(1);
    }
    fullDataset->nInstances++;
  }

  rewind(input);
}

void saveTrainedModel(struct Fullproblem *alphOptProblem, struct yDenseData *fullDataset, char *filename, struct svm_args *params)
/* Function to save the entries of a trained model. */
{
  FILE *file = fopen(filename, "w");
  fprintf(file, "%d\n",params->kernel );

  fprintf(file, "%d\n",fullDataset->nFeatures );

  int count = 0;
  for (int i = 0; i < alphOptProblem->n; i++) {
    if (alphOptProblem->alpha[i] > 0.0) {
      count++;
    }
  }
  int *active = malloc(sizeof(int)*count);
  int j = 0;
  for (int i = 0; i < alphOptProblem->n; i++) {
    if (alphOptProblem->alpha[i] > 0.0) {
      active[j] = i;
      j++;
    }
  }
  double *h = malloc(sizeof(double)*alphOptProblem->n*count);
  double **H = malloc(sizeof(double*)*alphOptProblem->n);
  int r = -1;

  for (int i = 0; i < alphOptProblem->n; i++) {
    H[i] = &h[i*count];
  }

  if (params->kernel == LINEAR) {
    for (int i = 0; i < alphOptProblem->n; i++) {
      for (int j = 0; j < count; j++) {
        H[i][j] = 0.0;
        for (int k = 0; k < fullDataset->nFeatures; k++) {
          H[i][j] += fullDataset->data[i][k]*fullDataset->data[active[j]][k];
        }
       	if( fullDataset->y[active[j]]*fullDataset->y[i]<0){

        	H[i][j] = -H[i][j];
      }
      }
    }
    for (int i = 0; i < count; i++) {
      if (alphOptProblem->alpha[active[i]] < alphOptProblem->C*0.99) {
        r = active[i];
        break;
      }
    }
    if (r<0) {
      exit(77);
    }
    double b = 1.0;
    for (int i = 0; i < count; i++) {
      b -= H[r][i]*alphOptProblem->alpha[active[i]];
    }
    if(fullDataset->y[r] < 0){
    b = -b;
	}

    double *w = malloc(sizeof(double)*fullDataset->nFeatures);

    for (int i = 0; i < fullDataset->nFeatures; i++) {
      w[i] = 0.0;
      for (int j = 0; j < count; j++) {
	if(fullDataset->y[active[j]] < 0){
        	w[i] += alphOptProblem->alpha[active[j]]*fullDataset->data[active[j]][i];

	}else{
	w[i] -= alphOptProblem->alpha[active[j]]*fullDataset->data[active[j]][i];
	}
      }
    }



    fprintf(file, "%lf\n",b );
    for (int i = 0; i < fullDataset->nFeatures; i++) {
      fprintf(file, "%lf\n",w[i] );
    }
    free(w);

  }
  free(active);
  free(h);
  free(H);
  fclose(file);

}

void testSavedModel(struct denseData *fullDataset, char* fn, struct svm_args *params)
{
/*  FILE *alphOptProblem = fopen(fn, "r");
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
      if (fullDataset->y[i]*value < 0.0) {
        printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
        wrong++;
      }
      else{
        printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
      }
    }
    free(w);
  }
  else if(params->kernel == POLYNOMIAL)  {
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
        contrib = pow(contrib + params->Gamma, params->degree);
        value += alphaY[j]*contrib;
      }
      if (fullDataset->y[i]*value < 0.0) {
        printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
        wrong++;
      }
      else{
        printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
      }
    }
  }
  else if(params->kernel == EXPONENTIAL)  {
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
    double y;
    for (int i = 0; i < fullDataset->nInstances; i++) {
      value = b;
      for (int j = 0; j < count; j++) {
        contrib = 0;
        for (int k = 0; k < fullDataset->nFeatures; k++) {
          y = X[j][k] - fullDataset->data[i][k];
          contrib -= y*y;
        }
        contrib *= params->Gamma;
        value += alphaY[j]*exp(contrib);
      }
      if (fullDataset->y[i]*value < 0.0) {
        printf("%sres[%d] = %.3lf%s\n",RED,i,value,RESET );
        wrong++;
      }
      else{
        printf("%sres[%d] = %.3lf%s\n",GRN,i,value,RESET );
      }
    }
  }

  int right = fullDataset->nInstances - wrong;
  double pct = 100.0*((double)right/(double)fullDataset->nInstances);
  printf("%d correct classifications out of %d. %.2lf%% correct.\n",right,fullDataset->nInstances,pct );

  fclose(alphOptProblem);
*/
}

int readline(FILE *input, char **line)
/* Function to read lines from file */
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

int parseArguments(int argc, char *argv[], char** filename, struct svm_args *parameters)
/* Function to parse command line arguments with getopt */
{
  int c;

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
  while ((c = getopt( argc, argv, "f:k:t:c:d:vhs:g:")) != -1){
    switch (c) {
      case 'f':
        *filename = optarg;
        break;
      case 'k':
        parameters->kernel = atoi(optarg);
	if (parameters->kernel != 0){
		printf("Non-linear kernels not enable for parallel implementation.\n");
	}
        break;
      case 'c':
        parameters->C = atof(optarg);
        break;
      case 't':
        parameters->test = 1;
        parameters->modelfile = optarg;
        break;
      case 'd':
        parameters->degree = atoi(optarg);
        break;
      case 's':
        parameters->save = 1;
        parameters->savename = optarg;
        break;
      case 'v':
        parameters->verbose = 1;
        break;
      case 'g':
        parameters->Gamma = atof(optarg);
        break;
      case 'h':
        printf("I was supposed to put in a help message here.\n");
        break;
    }
  }

  if (*filename == NULL) {
    printf("io.cpp: parseArguments: no input file selected.\n");
    exit(1);
  }
  return 0;
}

void preprocess(struct denseData *fullDataset)
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
{
  for (int i = 0; i < fullDataset->nInstances; i++) {
    for (int j = 0; j < fullDataset->nFeatures; j++) {
      fullDataset->data[i][j]-=mean[j];
      fullDataset->data[i][j]/=stdDev[j];
    }
  }
}

void calcStdDev(double* stdDev, double* mean, struct denseData *fullDataset)
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


void cleanData( struct denseData *fullDataset){
  for (int i = 0; i < fullDataset->nInstances - 1; i++) {
    printf("%d\n",i );
    for (int j = i; j < fullDataset->nInstances; j++) {
      int flag = 1;
      double check;
      double previous = fullDataset->data[i][0]/fullDataset->data[j][0];
      for (int k = 1; k < fullDataset->nFeatures; k++) {
        check = fullDataset->data[i][k]/fullDataset->data[j][k];
        printf("%lf and %lf\n",previous,check );
        if (fabs(check - previous) > 0.0001 ) {
          flag = 0;
          break;
        }
        previous = check;
      }
      if (flag == 1) {
        printf("%d is broken\n",j );
      }
    }
  }
}


void freeDenseData(struct denseData *fullDataset)
/* Function to free dynamically allocated memory in dense data set struct. */
{
  free(fullDataset->data);
  free(fullDataset->data1d);
}
