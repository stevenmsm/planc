#ifndef OPENMP_R2_HPP_
#define OPENMP_R2_HPP_

#include <cmath>
//#include <stdio.h>
//#include <omp.h>
#include "nmf.hpp"
#include "utils.hpp"
//#ifdef MKL_FOUND
//#include <mkl.h>
//#else
//#include <lapacke.h>
//#endif

//#define ONE_THREAD_MATRIX_SIZE 2000

template <class T>
class R2NMF: public NMF<T> {
  private:
    // NOTE: in PLANC, H is nxk matrix and it is nx2 in R2NMF
    T At;
    MAT WtW;
    MAT HtH;
    MAT WtA;
    MAT HtAt;
    // initiate Ht: kxn and Wt: kxm. Easier to compute later.
    MAT Ht;
    MAT Wt;
    /*
     * Collected statistics are
     * iteration Htime Wtime totaltime normH normW densityH densityW relError
     */
    void allocateMatrices() {
        WtW = arma::zeros<MAT >(this->k, this->k);
        HtH = arma::zeros<MAT >(this->k, this->k);
        WtA = arma::zeros<MAT >(this->k, this->n);
        HtAt = arma::zeros<MAT >(this->k, this->m);
	// initiate Ht and Wt from H and W
	Ht = this->H.t();
	//cout<<"Ht size"<<PRINTMATINFO(Ht)<<endl;
	Wt = this->W.t();
	//cout<<"Wt size"<<PRINTMATINFO(Wt)<<endl;
    }
    void freeMatrices() {
        this->At.clear();
        WtW.clear();
        HtH.clear();
        WtA.clear();
        HtAt.clear();
	Ht.clear();
	Wt.clear();
    }
    void R2Solve(MAT& X, const MAT& C, const MAT& R){
	    //Recall NNLS problem (solve fore X: CX=R): 1.WtWHt=WtA; 2.HtHWt=HtAt
	    //X is what we are trying to solve in this function: either Ht or Wt; C is the grand matrix: WtW or HtH;
	    //R is either WtA or HtAt; Other is the matrix that we don't solve (Other is W if X is Ht, H if X is Wt)
		
	    //initiate some local variables
    	    arma::rowvec u; // store row 1 projection, normalized
    	    arma::rowvec v; //store row 2 projection, normalized
    	    arma::umat notbothpos; //store indices of columns that have at least one negative entry
	    //UINT numThreads =  OPENBLAS_NUM_THREADS; // initiate number of threads

	    // X=C\R; solve()=matlab backslash=choleskey factorization
	    X = arma::solve(C,R);

            // check if any entries in H are negative. Fix by pythagorean theorem and orth projection
	    u = R.row(0)/C(0,0); //orth proj A/At down to Other_1 and normalize (row vec)
	    v = R.row(1)/C(1,1); //orth proj A/At down to Other_2 and normalize (row vec)
            notbothpos = arma::any(X<0); //store cols of Ht (kxn) that have at least one negative entry
	    double beta1 = std::sqrt(C(0,0));
	    double beta2 = std::sqrt(C(1,1));
	    
	    // parallel for loop
            #pragma omp parallel for 
	    for (int i=0; i<X.n_cols;i++) {//loop over cols of Ht, find notbothpos cols and fix them	
		if (notbothpos(i)!=0) {
		   if (beta1*u(i) > beta2*v(i)){//by pythagorean thm, choose the longer projection
			X(0,i)=u(i);
			X(1,i)=0;
		   }
		   else{
			X(0,i)=0;
			X(1,i)=v(i);
		   }
		}		
	    }
	
    }

    void computeNMF_R2() { //Recall Problem: ||A-WHt||, this function 
	int currentIteration = 0;
        double t1, t2;
        this->At = this->A.t();
        INFO << "computed transpose At=" << PRINTMATINFO(this->At) << std::endl;
        while (currentIteration < this->num_iterations()) {
            tic();
            // given W solve Ht (WtWHt=WtA)
            tic();
            WtA = Wt * this->A; 
            WtW = Wt * this->W;
            INFO << "starting H Prereq for " << " took=" << toc();
            INFO << PRINTMATINFO(WtW) << PRINTMATINFO(WtA) << std::endl;
	    // solve the nnls problem
	    tic();
            R2Solve(Ht, WtW, WtA); //private solve function for nnls rank 2
            INFO << "Completed H ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << std::endl;


            // given Ht solve W (HtHWt=HtAt)
            tic();
	    this->H = Ht.t(); //computed Ht kxn, update H as nxk
            HtAt = Ht * this->At;
            HtH = Ht * this->H;
            INFO << "starting W Prereq for " << " took=" << toc()
                 << PRINTMATINFO(HtH) << PRINTMATINFO(HtAt) << std::endl;
	    // solve the nnls problem: Wt = HtH \ HtAt
	    tic();
	    R2Solve(Wt, HtH, HtAt);
	    this->W = Wt.t(); //computed Wt kxm, update W mxk (prevent not up-to-date this->W)
            INFO << "Completed W ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << std::endl;
            INFO << "Completed It ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << std::endl;

	    //Compute Objective Error (call function in nmf.hpp, the more efficient one)
	    WtW=Wt*this->W; //updated Wt and this->W, but did not update WtW
            this->computeObjectiveError(At, WtW, HtH); //All inputs must be uptodate before calling error function
            INFO << "Completed it = " << currentIteration << " R2ERR="
                 << sqrt(this->objective_err)/this->normA << std::endl;

	    //iteration count
            currentIteration++;
        }
        this->normalize_by_W();
    }

  public:
    R2NMF(const T &A, int lowrank): NMF<T>(A, lowrank) {
        allocateMatrices();
    }
    R2NMF(const T &A, const MAT &llf, const MAT &rlf) : NMF<T>(A, llf, rlf) {
        allocateMatrices();
    }
    void computeNMF() { 
	computeNMF_R2();
    }
    ~R2NMF() {
        freeMatrices();
    }
};
#endif  // OPENMP_R2_HPP_
