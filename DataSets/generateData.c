#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
	double k, ok, okay;
	FILE *fp;
	srand48(77);
	int i,j;
	fp = fopen("flawedbigData.txt","w");
	for (i = 0; i< 10000; i++){
		fprintf(fp,"1\t");
		if(i%3000 == 0){
			for(j=0; j<800; j++){
				k = (double)j - (drand48()*5);
				fprintf(fp,"%lf\t",k);
			}
			fprintf(fp,"\n");
		}else{
			for(j=0; j<800; j++){
				k = (drand48()*5)+(double)j;
				fprintf(fp,"%lf\t",k);
			}
			fprintf(fp,"\n");
		}
	}
	for (i = 0; i< 10000; i++){
		fprintf(fp,"-1\t");
		if(i%3000 == 0){
			for(j=0; j<800; j++){
				k = (double)j + (drand48()*5);
				fprintf(fp,"%lf\t",k);
			}
			fprintf(fp,"\n");
		}else{
			for(j=0; j<800; j++){
				k = (double)j - (drand48()*5);
				fprintf(fp,"%lf\t",k);
			}
			fprintf(fp,"\n");

		}
	}
	fclose(fp);
	return 0;
}

