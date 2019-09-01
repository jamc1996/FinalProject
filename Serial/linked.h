#ifndef LINKED_H
#define LINKED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "svm.h"
#include "kernels.h"

/*      fullproblem.h -- header file for fullproblem.c
 *
 *      Author:     John Cormican
 *
 */


double* findListLine(List l, int n);
List Init_Empty_List();
List append(struct denseData *ds, List l, int n);
void print_list(List l);
void free_list(List l);
List delete(int find, List l);
double* findListLineSetLabel(List l, int n , int newLabel);

#endif
