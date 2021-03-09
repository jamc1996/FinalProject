#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
	double k, ok, okay;
	FILE *alphOptProblem;
	srand48(77);
	int i,j;
	alphOptProblem = fopen("flawedbigData.txt","w");
	for (i = 0; i< 20000; i++){
		fprintf(alphOptProblem,"1\t");
		if(i%2000 == 0){
			for(j=0; j<500; j++){
				k = (double)j - (drand48()*5);
				fprintf(alphOptProblem,"%lf\t",k);
			}
			fprintf(alphOptProblem,"\n");
		}else{
			for(j=0; j<500; j++){
				k = (drand48()*5)+(double)j;
				fprintf(alphOptProblem,"%lf\t",k);
			}
			fprintf(alphOptProblem,"\n");
		}
	}
	for (i = 0; i< 20000; i++){
		fprintf(alphOptProblem,"-1\t");
		if(i%2000 == 0){
			for(j=0; j<500; j++){
				k = (double)j + (drand48()*5);
				fprintf(alphOptProblem,"%lf\t",k);
			}
			fprintf(alphOptProblem,"\n");
		}else{
			for(j=0; j<500; j++){
				k = (double)j - (drand48()*5);
				fprintf(alphOptProblem,"%lf\t",k);
			}
			fprintf(alphOptProblem,"\n");

		}
	}
	fclose(alphOptProblem);
	return 0;
}

