#ifndef IO_H
#define IO_H

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>

#include "svm.h"

/*      io.h -- header file for io.c
 *
 *      Author:     John Cormican
 *
 */

#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

void testSavedModel(struct denseData *fullDataset, char* fn);
void saveTrainedModel(struct Fullproblem *alphOptProblem, struct denseData *fullDataset, double ytr);
void readFile(char* const filename, struct denseData* fullDataset);
int readline(FILE *input, char **line);
void count_entries(FILE *input, struct denseData* fullDataset);
int parseArguments(int argc, char *argv[], char** filename);
void preprocess(struct denseData *fullDataset);
void calcMeans(double *mean, struct denseData *fullDataset);
void normalise(double* mean, double* stdDev, struct denseData* fullDataset);
void calcStdDev(double* stdDev, double* mean, struct denseData *fullDataset);
struct svmModel createFittedModel(double *w, int kernel, int trainElapsedTime, struct denseData *fullDataset, struct Fullproblem *alphOptProblem, double ytr);
void setUpDense(struct denseData *fullDataset, double** trainData, int nFeatures, int nInstances, int nPos);
void change_params(struct svm_args *parameters);
void saveTrainedModel2(struct Fullproblem *alphOptProblem, struct denseData *fullDataset, double ytr, const char* fileName);

#endif
