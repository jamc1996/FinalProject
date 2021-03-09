#ifndef LINKED_H
#define LINKED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "svm.h"
#include "kernels.h"

/*      linked.h -- header file for linked.c
 *
 *      Author:     John Cormican
 *
 */

List Yappend(struct yDenseData *fullDataset, List l, int n);

double* findListLine(List l, int n);
List Init_Empty_List();
List append(struct denseData *fullDataset, List l, int n);
void print_list(List l);
void free_list(List l);
List delete(int find, List l);
double* findListLineSetLabel(List l, int n , int newLabel);
List newAppend(struct denseData *fullDataset, struct receiveData *rd, List l, int n, struct Fullproblem *newfp, struct Fullproblem *oldfp);

#endif
