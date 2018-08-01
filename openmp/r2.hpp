#ifndef OPENMP_R2_HPP_
#define OPENMP_R2_HPP_

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "nmf.hpp"
#include "utils.hpp"
using namespace std;


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
	    //int threadCount = 2; 

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
                //cout<<"number of threads: "<<omp_get_num_threads()<<endl;
                //printf("Thread %d working.\n", omp_get_thread_num());
	    }
	
    }

    //Recall Problem: ||A-WHt||. This function solved the rank2 nnls problem at each node of Hierarchical tree
    void computeNMF_R2(MAT& W, MAT& H, const MAT& A) {  
	// solve W (mx2) and H (nx2), given A. Function takes in an initialization of W and H, and local data matrix A
	// since W and H are passed by reference, and change of W and H in the function will "pass back" to W and H when 		// they are called

	    int currentIteration = 0;
        double t1, t2;
        T At_temp = A.t();
        Wt = W.t();
	    Ht = H.t();
        INFO << "computed transpose At=" << PRINTMATINFO(At_temp) << std::endl;
        while (currentIteration < this->num_iterations()) {
            tic();
            // given W solve Ht (WtWHt=WtA)
            tic();
            WtA = Wt * A; 
            WtW = Wt * W;
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
	        H = Ht.t(); //computed Ht kxn, update H as nxk
            HtAt = Ht * At_temp;
            HtH = Ht * H;
            INFO << "starting W Prereq for " << " took=" << toc()
                 << PRINTMATINFO(HtH) << PRINTMATINFO(HtAt) << std::endl;
            // solve the nnls problem
            tic();
            R2Solve(Wt, HtH, HtAt);
            W = Wt.t(); //computed Wt kxm, update W mxk (prevent not up-to-date this->W)
            INFO << "Completed W ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << std::endl;
            INFO << "Completed It ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << std::endl;

	        //Compute Objective Error (call function in nmf.hpp, the more efficient one)
	        WtW = Wt*W; //updated Wt and this->W, but did not update WtW
	        double normA_temp = arma::norm(A, "fro"); //compute A's F norm
            this->computeObjectiveError(At_temp, WtW, HtH, W, H); //All inputs must be updated before calling error function
            INFO << "Completed it = " << currentIteration << " R2ERR="
                 << sqrt(this->objective_err)/normA_temp << std::endl;

	        //iteration count
            currentIteration++;
        }
        //this->normalize_by_W();
        //  normalize W and H by norm of W (copied from nmf.hpp, since in different tree nodes, W and H are different but
        //  normalize_by_W() always uses this->W, this->H
        MAT W_square_temp = arma::pow(W, 2);
        ROWVEC norm2_temp = arma::sqrt(arma::sum(W_square_temp, 0));
        for (int i = 0; i < 2; i++) {
            if (norm2_temp(i) > 0) {
                W.col(i) = W.col(i) / norm2_temp(i);
                H.col(i) = H.col(i) * norm2_temp(i);
            }
        }
    }

  public:
    R2NMF(const T &A, int lowrank): NMF<T>(A, lowrank) {
        allocateMatrices();
    }
    R2NMF(const T &A, const MAT &llf, const MAT &rlf) : NMF<T>(A, llf, rlf) {
        allocateMatrices();
    }

    // this function outputs a hierarchy of the input data matrix, which each nodes is a rank2 NNLS problem
    // output:
    // tree: A 2-by-2*(k-1) matrix that encodes the tree structure. The two entries in the i-th column are the numberings
    //       of the two children of the node with numbering i.
    //       The root node has numbering 0, with its two children always having numbering 1 and numbering 2.
    //       Thus the root node is NOT included in the 'tree' variable.
    // splits: An array of length k-1. It keeps track of the numberings of the nodes being split
    //         from the 1st split to the (k-1)-th split. (The first entry is always 0.)
    // is_leaf: An array of length 2*(k-1). A "1" at index i means that the node with numbering i is a leaf node
    //          in the final tree generated, and "0" indicates non-leaf nodes in the final tree.
    // leaf_membership: An array of length 2*(k-1). The i-th element contains the subset of items
    //                  at the node with numbering i.
    // timings: An array of length k-1.
    //          Its i-th element is the wall-time for performing i splits (with i+1 leaf nodes).
    // Ws: A cell array of length 2*(k-1).
    //     Its i-th element is the topic vector of the cluster at the node with numbering i.
    // priorities: An array of length 2*(k-1).
    //             Its i-th element is the modified NDCG scores at the node with numbering i (see the reference paper).
    void computeNMF() {
        MAT tree(2 , 2*(this->k-1));
        tree.zeros();
        arma::rowvec splits = arma::zeros<arma::rowvec>(this->k-1);
        arma::rowvec is_leaf = arma::zeros<arma::rowvec>(2*(this->k-1));
        arma::rowvec leaf_membership = arma::zeros<arma::rowvec>(2*(this->k-1));

        MAT W_temp;
        MAT H_temp;
        W_temp = arma::randu<MAT>(this->A.n_rows,2);
        H_temp = arma::randu<MAT>(this->A.n_cols,2);
        
        computeNMF_R2(W_temp, H_temp, this->A);
    }

    ~R2NMF() {
        freeMatrices();
    }
};
#endif  // OPENMP_R2_HPP_
