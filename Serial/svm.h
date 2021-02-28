#ifndef SVM_H
#define SVM_H

#define LINEAR				0
#define POLYNOMIAL 		1
#define EXPONENTIAL		2

/*      svm.h -- header file containing structures needed for gertSVM algorithm.
 *
 *      Author:     John Cormican
 *
 */


//Cell structure for linked list:
typedef struct Cell_struct
{
  struct Cell_struct* next;
  struct Cell_struct* prev;
  int label;
  double*  line;
} Cell;

//List structure used in linked.c
typedef struct
{
  struct Cell_struct* head;
  struct Cell_struct* tail;
} List;

//Command Line Arguments struct for args passed through the command line.
typedef struct svm_args
{
	int kernel;
	int degree;
	double C;
	double Gamma;
	int verbose;
	int test;
	char* modelfile;
	int save;
	char* savename;
} clargs;

// denseData structure for storing dataset
typedef struct denseData{
	int nInstances;
	int nFeatures;
	int nPos;
	int nNeg;
	double** data;
	double* data1d;
} dataSet;

// Projected struct for the projected subproblem
typedef struct Projected{
	// Re-inited each time:
	double* alphaHat;
	double* yHat;
	double* rHat;
	double** H;

	// Will be set during CG iterations
	double* gamma;
	double* rho;
	double* Hrho;

	// Changed independent (p size, ytr calc) or unchanging
	int p;
	double C;
	double* h;
	double ytr;
}Subproblem;

// Full problem struct for everything to be solved by the algorithm.
typedef struct Fullproblem{
	int n;
	int p;
	int q;

	double C;

	double* alpha;
	double* beta;
	double* gradF;

	int* active;
	int* inactive;

	List partialH;
} Bigprob;

typedef struct svmModel{
  int kernel;
  int trainElapsedTime;
  int nFeatures;
  double* decisionVector;
  double biasTerm;
} fitSvmModel;

struct svm_args parameters;


#endif
