#ifndef SVM_H
#define SVM_H

#define LINEAR				0
#define POLYNOMIAL 		1
#define EXPONENTIAL		2

#define nThreads 4

typedef struct Cell_struct
{
  struct Cell_struct* next;
  struct Cell_struct* prev;
  int label;
  double*  line;
}
Cell;

typedef struct
{
  struct Cell_struct* head;
  struct Cell_struct* tail;
}
List;

struct svm_args
{
	int kernel;
	int degree;
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	double Gamma;
	int verbose;
	int test;
	char* modelfile;
	int save;
	char* savename;
} clargs;

struct yDenseData{
	int nInstances;
	int nFeatures;

	int *y;
	double **data;
	double *data1d;
};

struct denseData{
  int nInstances;
  int nFeatures;
	int nPos;
	int nNeg;

	int procInstances;
	int procPos;
	int procNeg;

	int posStart;
	int negStart;

  double** data;
  double* data1d;
};

struct CGSLSProj{
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
};

struct Projected{
	// Re-inited each time:
	double* alphaHat;
	int* yHat;
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

struct Fullproblem{
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
} fullprob;

struct receiveData{
	int total;
	int nFeatures;
	int nPos;
	double ytr;
	double *alpha;
	int *y;
	double *data1d;
	double **data;

	double *w;
};

struct svm_args parameters;
#endif
