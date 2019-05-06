// An interface to BigQUIC with raw data as input.
//
// See the README file for more information.

//#include <metis.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include "bigquic.h"

using namespace std;

void exit_with_help()
{
	printf(
	"Usage: ./QUIC [options] data_filename output_filename\n"
	"options:\n"
	"    -l lambda : set the regularization parameter lambda (default 0.5)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -t max_iter: set the number of iterations (default 5)\n"    
	"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"     
	"    -k num_blocks: number of blocks (default value depends on m)\n"    
	"    -m memory_usage: memory capacity in MB (default 8000)\n"  
	"                     This is just an estimate for the memory usage, so we recommend\n"
	"                     to use -k for exactly controling number of blocks\n"
	"    -q verbose: show information or not (default 1, can be 0, 1, 2)\n"
	"    -r correlation: whether to use correlation matrix (default 1, can be 0, 1)\n"
	);
	exit(1);
}


int main(int argc, char **argv)
{
	char input_file_name[1024], output_file_name[1024];
	char *linestr = (char *)malloc(sizeof(char)*2000);
	double lambda=0.5;
	int maxit = 5;
	int numthreads = 4;
	double epsilon = 1e-3; 
	int k = 0;
	int isnormalize = 1;
	int verbose = 0;

	double memory_size = 8000000000;
	int p, n;
	int i;

	// parse options
	for( i=1 ; i<argc ; i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'l':
				lambda = atof(argv[i]);
				break;

			case 'n':
				numthreads = atoi(argv[i]);
				break;

			case 't':
				maxit = atoi(argv[i]);
				break;

			case 'e':
				epsilon = atof(argv[i]);
				break;

			case 'k':
				k = atoi(argv[i]);
				break;

			case 'm':
				memory_size = atof(argv[i])*1000000;
				break;

			case 'q':
				verbose = atoi(argv[i]);
				break;

			case 'r':
				isnormalize = atoi(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if ( i>= argc-1)
		exit_with_help();

	strcpy(input_file_name, argv[i]);
	strcpy(output_file_name, argv[i+1]);
	FILE *fp = fopen(input_file_name, "r");
	FILE *fout = fopen(output_file_name, "w");

	// Set k if the user does not specify

	fscanf(fp, "%d", &p);
	fscanf(fp, "%d", &n);

	double kk = 32.0*p*p/memory_size; 
	k = int(kk)+1;

	printf("p:%d n:%d k:%d\n", p, n, k);
	fflush(stdout);
	double *samples = (double *)malloc(sizeof(double)*p*n);
	for ( int i=0 ; i<n ; i++ )
		for ( int j=0 ; j<p ; j++)
		{
			fscanf(fp, "%s", linestr);
			samples[j*n+i] = atof(linestr);
		}

	vector<int> mapping(p);
	double *samples_new;
	int subp;

	if ( isnormalize == 0 ) {
		samples_new = samples;
		subp = p;
		for ( long i=0 ; i<p*n ; i++)
			samples_new[i] /= sqrt(n-1);
		for ( long i=0 ; i<p ; i++ )
			mapping[i] = i;
	}
	else
	{
		if ( verbose >= 1)
		{
			printf("begin normalizing data\n");
			fflush(stdout);
		}
		
		samples_new = (double *)malloc(sizeof(double)*p*n);
		NormalizeData(p, n, samples, samples_new, mapping);
		subp = mapping.size();
		if ( verbose >= 1)
		{
			printf("Number of nodes with std>1e-10: %d\n", subp);
			fflush(stdout);
		}
	}

/*
	for ( long i=0 ; i<subp ; i++ )
	{
		for ( long j=0 ; j<n ; j++ )
			printf("%lf ", samples_new[i*n+j]);
		printf("\n");
	}
*/
	printf("lambda: %lf, p: %ld, n: %ld realp: %ld\n", lambda, p, n, subp);
	fflush(stdout);
	
	p = subp; // will not use original p anymore

	// Print out hostname; 
	char hostname[40];
	if ( gethostname(hostname, 40) == -1)
		printf("error getting hsotname\n");
	else
		printf("hostname: %s\n", hostname);
	fflush(stdout);

	vector<double> objlist;
	vector<double> timelist;

	// Initialize by a sparse identity matrix
	smat_t X(subp);
	X.identity(subp);

	// Set the idmap in the smat_t object
	// By setting this, X.print() can have the original node id. 
	X.id_map.resize(subp);
	for ( long i=0 ; i<subp ; i++ )
		X.id_map[i] = mapping[i];

	QUIC(p, n, samples_new, lambda, epsilon, verbose, maxit, k, numthreads, X, objlist, timelist, fout);

	// Output a symmetric solution
	X.print(fout);
	fclose(fout);
	
	return 0;
}

