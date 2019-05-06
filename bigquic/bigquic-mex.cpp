// This is the MEX wrapper for BigQUIC.  The algorithm is in QUIC.C.
// 
// Invocation form within Matlab:
// [X objlist timelist] = bigquic(data, [,'options']) 
// 
// input arguments: 
// 		data: the n by p matrix, 
// 			  where n is number of samples, p is number of random variables
//		options: 
//			-l lambda : set the regularization parameter lambda (default 0.5) 
//			-n threads : set the number of threads (default 4)
//			-t max_iter: set the number of iterations (default 5)
//			-e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)
//			-k num_blocks: number of blocks (default value depends on m)
//			-m memory_usage: memory capacity in MB (default 8000)
//			-q verbose: show information or not (default 1, can be -1, 0, 1, 2)
//			-r correlation: whether to use correlation matrix (default 1, can be 0, 1)
//
// output arguments: 
// 		X: the p by p sparse inverse covariance matrix. 
//		objlist: list of objective functions at each iteration. 
//		timelist: list of cpu time. 

#include "mex.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "bigquic.h"

void exit_with_help()
{
	mexPrintf(
	"Usage: [X objlist timelist] = bigquic(S, [, 'pmf_options'])\n"
	"options:\n"
	"   -l lambda : set the regularization parameter lambda (default 0.5)\n" 
	"   -n threads : set the number of threads (default 4)\n"    
    "	-t max_iter: set the number of iterations (default 5)\n"    
	"   -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n" 
	"   -k num_blocks: number of blocks (default value depends on m)\n"    
	"   -m memory_usage: memory capacity in MB (default 8000)\n"  
	"   -q verbose: show information or not (default 1, can be 0, 1, 2)\n"
	"   -r correlation: whether to use correlation matrix (default 1, can be 0, 1)\n"
	);
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double lambda=0.5;
	int maxit = 5;
	int numthreads = 4;
	double epsilon = 1e-3; 
	int k = 0;
	int isnormalize = 1;
	int verbose = 0;
	double memory_size = 8000000000;

    if (nrhs < 1 || nrhs >=3 ) {
		exit_with_help();
		fake_answer(plhs);
    } 
	else
	{
		// Get input data
		double* samples = mxGetPr(prhs[0]);
	    int p = (int) mxGetN(prhs[0]);
		int n = (int) mxGetM(prhs[0]);
		char cmd[2000];
		char *argv[100];
		int argc = 1;

		if ( nrhs == 2 )
		{
			mxGetString(prhs[1], cmd,  mxGetN(prhs[1]) + 1);
			if((argv[argc] = strtok(cmd, " ")) != NULL)
				while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
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
		}
		double kk = 32.0*p*p/memory_size; 
		k = int(kk)+1;

		vector<int> mapping(p);
		double *samples_new=(double *)malloc(sizeof(double)*p*n);
		int subp;

		if ( isnormalize == 0 ) {
			subp = p;
			for ( long i=0 ; i<p*n ; i++)
				samples_new[i] = samples[i]/sqrt(n-1);
			for ( long i=0 ; i<p ; i++ )
				mapping[i] = i;
		}
		else
		{
			NormalizeData(p, n, samples, samples_new, mapping);
			subp = mapping.size();
		}
	
		
		p = subp; // will not use original p anymore
	
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
	
		QUIC(p, n, samples_new, lambda, epsilon, verbose, maxit, k, numthreads, X, objlist, timelist);

		X.form_originalgraph();

		free(samples_new);
		plhs[0] = mxCreateSparse(X.p,X.p,X.nnz,mxREAL);
		mwIndex *ir = mxGetIr(plhs[0]);
		mwIndex *jc = mxGetJc(plhs[0]);
		double *pr = (double *)mxGetPr(plhs[0]);
		for ( long i=0 ; i<X.nnz ; i++)
			ir[i] = X.col_idx[i];
		for ( long i=0 ; i<=X.p ; i++ )
			jc[i] = X.row_ptr[i];
		for ( long i=0 ;i<X.nnz ; i++)
			pr[i] = X.values[i];

		int niter = objlist.size();
		plhs[1] = mxCreateDoubleMatrix(1, niter, mxREAL);
		plhs[2] = mxCreateDoubleMatrix(1, niter, mxREAL);
		double *val1 = mxGetPr(plhs[1]);
		double *val2 = mxGetPr(plhs[2]);
		for ( int i=0 ; i<niter ; i++ )
		{
			val1[i] = objlist[i];
			val2[i] = timelist[i];
		}
	}
}
