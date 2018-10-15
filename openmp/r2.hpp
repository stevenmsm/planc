#ifndef OPENMP_R2_HPP_
#define OPENMP_R2_HPP_

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <time.h>
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

        this->H=H;
        this->W=W;
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
    // clusters: A cell array of length 2*(k-1). The i-th element contains the subset of items
    //           at the node with numbering i.
    // timings: An array of length k-1.
    //          Its i-th element is the wall-time for performing i splits (with i+1 leaf nodes), in seconds.
    // Ws: A cell array of length 2*(k-1).
    //     Its i-th element is the topic vector of the cluster at the node with numbering i.
    // priorities: An array of length 2*(k-1).
    //             Its i-th element is the modified NDCG scores at the node with numbering i (see the reference paper).
    void computeNMF() {
        //initialize all inputs for Hier-R2

        // initialize all outputs from Hier-R2
        MAT tree(2 , 2*(this->k-1));
        tree.zeros();
        arma::rowvec splits = arma::ones<arma::rowvec>(this->k-1);
        splits = splits * (-1); //no nodes are being splitted yet
        arma::rowvec is_leaf = arma::ones<arma::rowvec>(2*(this->k-1));
        is_leaf = is_leaf * (-1); //no one is a leaf node nor a parent node
        arma::field<VEC> clusters (2*(this->k-1)); //cell array of vectors
        arma::rowvec timings = arma::zeros<arma::rowvec>(this->k-1);
        MAT Ws(this->A.n_rows,2*(this->k-1)); //store all node's W vector, each is mx1
        Ws.zeros();
        arma::rowvec priorities = arma::zeros<arma::rowvec>(2*(this->k-1));


        // Split node 0, if not all rows of the data matrix sum to non-zero values, adjust as follow
        // initiate and randomize the first split's W, H
        arma::umat term_subset = arma::find(sum(this->A,1));
        MAT W = arma::randu<MAT>(term_subset.n_elem,2);
        MAT H = arma::randu<MAT>(this->A.n_cols,2);
        if (term_subset.n_elem == this->A.n_rows) { //if all rows sum to nonzero, split node 0
            computeNMF_R2(W, H, this->A);
        }
        else{ //if not, split node 0 for those nonzero summed rows and set zero summed rows to 0 for W
            computeNMF_R2(W, H, this->A.rows(term_subset));
            MAT W_complete(this->A.n_rows,2);
            W_complete.zeros();
            W_complete.rows(term_subset)=W;
            W = W_complete;
            W_complete.clear();
        }

        // keep splitting
        // initiate some variables used in the loop
        clock_t t0 = clock();
        int result_used=0; //keep track of the latest child number
        arma::uword split_node; //keep track of which node the loop is current splitting
        int new_node_1; // after the current split, record the 2 children's number
        int new_node_2;
        double min_priority; //store minimum priority among current leaf nodes
        double max_priority; //max priority among current leaf nodes
        arma::umat leaves; // store the indices of all current leaf nodes
        arma::vec temp_priority; // store the priorities (from the priorities array) of current leaf nodes
        arma::field<MAT> W_buffer(2*(this->k-1)); //stores all W's from all nodes, similar to cell array
        arma::field<MAT> H_buffer(2*(this->k-1)); //stores all H's from all nodes, similar to cell array
        VEC split_subset(this->A.n_cols);
        arma::colvec max_val; //max_val is a row vector that contains the max value of each column of H
        arma::ucolvec cluster_subset; //cluster_subset is a row vector that contains the index of max value of each column of H

        //initiate more variables
        int trial_allowance=3; //default trial_allowance
        double unbalanced=0.1; //default unbalanced


        //NOTE: C++ indexing start at 0, be careful transforming from matlab!!! Might have ERRORS!!!
        // k-1 is the number of splits for reaching k leaf nodes. Use a for loop around the number of splits to run Hier-R2
        for (int i=0; i<this->k-1; i++){ //from the first (i=0) split to the k-1 split (Note, C++ indexing start at 0)
            timings(i) = clock() - t0; //keep track of wall time per iteration in seconds

            //if this is the first split from node 0, initiate
            if (i==0){
                split_node = 0;// currently at node 0
                new_node_1 = 0; //splitted into node 1 and 2 (but indexing start at 0, so minus 1)
                new_node_2 = 1;
                min_priority = 1e308;
                for (int temp=0; temp<this->A.n_cols; temp++) {
                    split_subset(temp) = temp;
                }
                //split_subset.print();
            }
            //else, initiate
            else{
                leaves = arma::find(is_leaf==1); //find current all leaf nodes
                temp_priority = priorities(leaves); //store all current leaf nodes priorities. NOTE: temp_priority IS A COLVECTOR
                arma::umat temp_priority_G0 = arma::find(temp_priority>0);
                arma::vec temp_priority_temp = temp_priority(temp_priority_G0);
                min_priority = arma::min(temp_priority_temp); //store the min priority among current leaf nodes
                max_priority = max(temp_priority); //split the node with the largest "score" next
                split_node = index_max(temp_priority); //index of the node with the largest "score"
                if (max_priority <0){
                    cout<<"Cannot generate all leaf clusters."<<endl;
                    break;
                }
                //start with the current leaf node that has the highest "score"
                split_node = leaves (split_node);
                is_leaf(split_node) = 0; //we are now splitting this node, so it will no longer be a leaf node
                W = W_buffer(split_node);
                H = H_buffer(split_node); //NOTE: H IS (xxx) BY 2 IN THE CODE!!
                split_subset = clusters(split_node);
                new_node_1 = result_used; //c++ indexing start at 0, so minus 1 from matlab version
                new_node_2 = result_used+1;
                tree(0, split_node) = new_node_1;
                tree(1, split_node) = new_node_2;
            }
            result_used = result_used + 2;
            max_val = arma::max(H,1); //max_val is a row vector that contains the max value of each column of H
            cluster_subset = index_max(H,1); //NOTE: H IS (xxx) BY 2 IN THE CODE!!!
            //based on max_val, split the columns of data matrix into two clusters (based on the greater value in H)
            arma::uvec temp_subset_1 = arma::find(cluster_subset == 0);
            arma::uvec temp_subset_2 = arma::find(cluster_subset == 1);
            clusters(new_node_1) = split_subset(temp_subset_1); //cluster 1
            clusters(new_node_2) = split_subset(temp_subset_2); //cluster 2
            Ws.col(new_node_1) = W.col(0); //save the W matrix from the decomp of cluster 1 (first column)
            Ws.col(new_node_2) = W.col(1); // W matrix second column of cluster 2
            splits(i) = split_node; //record which node is being split
            is_leaf(new_node_1) = 1; //record if this is a permenant leaf node
            is_leaf(new_node_2) = 1;


            VEC subset = clusters(new_node_1);
            //[subset, W_buffer_one, H_buffer_one, priority_one] = trial_split(trial_allowance, unbalanced, min_priority, X, subset, W(:, 1), params);
            clusters(new_node_1) = subset;
            //W_buffer(new_node_1) = W_buffer_one;
            //H_buffer(new_node_1) = H_buffer_one;
            //priorities(new_node_1) = priority_one;

            subset = clusters(new_node_2);
            //[subset, W_buffer_one, H_buffer_one, priority_one] = trial_split(trial_allowance, unbalanced, min_priority, X, subset, W(:, 2), params);
            //clusters(new_node_2) = subset;
            //W_buffer(new_node_2) = W_buffer_one;
            //H_buffer(new_node_2) = H_buffer_one;
            //priorities(new_node_2)) = priority_one;
        }
        //change timings from clicks to seconds
        timings = timings / CLOCKS_PER_SEC;
    }

    ~R2NMF() {
        freeMatrices();
    }
};
#endif  // OPENMP_R2_HPP_
