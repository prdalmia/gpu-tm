/*
 * global_constants.h
 *
 *  Created on: Aug 2, 2010
 *      Author: arun
 */

#ifndef GLOBAL_CONSTANTS_H_
#define GLOBAL_CONSTANTS_H_

#define MAXITER 30
#define MAXTRANSZ 100

int MINSUPPORT;
double MINSUP_PER = 0.25;
int NUM_INSERT;
int NUM_ACTUAL_INSERT;

int DBASE_NUM_TRANS;
int DBASE_MAXITEM ;
int DBASE_AVG_TRANS_SZ;
int DBASE_BUFFER_SIZE = 8192;

// OBTAINED FROM PROFILING CODE ON CPU

int max_candidate_size=20;
int max_trans_size=100;
int max_large_list_size= 4000;
#define max_HashFunction 500
int threshold =2;
#endif /* GLOBAL_CONSTANTS_H_ */
