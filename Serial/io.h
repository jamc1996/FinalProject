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

void testSavedModel(struct denseData *ds, char* fn);
void saveTrainedModel(struct Fullproblem *fp, struct denseData *ds, double ytr);
void read_file(char* const filename, struct denseData* ds);
int readline(FILE *input, char **line);
void count_entries(FILE *input, struct denseData* ds);
int parse_arguments(int argc, char *argv[], char** filename);
void preprocess(struct denseData *ds);
void calcMeans(double *mean, struct denseData *ds);
void normalise(double* mean, double* stdDev, struct denseData* ds);
void calcStdDev(double* stdDev, double* mean, struct denseData *ds);

#endif
