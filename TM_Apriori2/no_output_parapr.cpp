// Compiler options:-
// -DBALT     = make trees balanced
#include <omp.h>
#include <iostream>
#include <fcntl.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <math.h>
#include <string.h>


#include "pardhp.h"

extern int DBASE_NUM_TRANS;
extern int DBASE_MAXITEM ;
extern int DBASE_AVG_TRANS_SZ;
extern int DBASE_BUFFER_SIZE ;
extern double MINSUP_PER ;
extern int MINSUPPORT;
extern int threshold;

FILE *tmp;
int **lsupcnt;
int nBlocks;
int nthreads = 1;

struct timeval tp;



char infile[500], outfile[500], off_file[100];

FILE *summary;
uint64_t ts_uint, te_uint, *t_io_uint;
uint64_t max_io_time;

int **start, **enda;
int **hash_pos;
int *max_trans_sz;


int more;

int *hash_indx = NULL;
int tot_cand = 0;
int *local_tot_cand;


extern void CUDA_entrypoint(transaction *data_set,int *host_cnt_ary,int records_per_thread,int nBlocks,int nthreads);

void parse_args(int argc, char **argv)
{
  extern char * optarg;
  int c;

  if (argc < 5){
    cout << "usage: -i<infile> -s<support> -n<number of threads>\n";
    cout << "-t<threshold>\n";
    exit(3);
  }
  else{
    while ((c=getopt(argc,argv,"b:i:f:o:s:t:n:x:"))!=-1){
      switch(c){
      case 'b':
        DBASE_BUFFER_SIZE = atoi(optarg);
        break;
      case 'i':
        strcpy(infile,optarg);
        break;
      case 'f':
        strcpy(off_file, optarg);
        break;
      case 'o':
        strcpy(outfile, optarg);
        break;
      case 's':
        MINSUP_PER = atof(optarg);
        break;
      case 't':
        threshold = atoi(optarg);
        break;
      case 'n':
        nBlocks = atoi(optarg);
        break;
      case 'x':
        nthreads = atoi(optarg);
        break;
      }
    }
  }
}

/*



void alloc_mem_for_var(int pid)
{
  start[pid] = new int [MAXITER];
  enda[pid] = new int [MAXITER];
}

void sum_counts(int *destary, int **srcary, int pid, int ub)
{
  int i,j;

  for(i=1; i < nBlocks; i++){
    for(j=0; j < ub; j++){
      destary[j] += srcary[i][j];
    }
  }
}
*/



void *main_proc(void *)
{
  int vist, hash_function;
  //Dbase_Ctrl_Blk DCB;
  //HashTree *oldCand=NULL;
  int k, i, j,pid;
  int *buf, tid, numitem, idx;
  int *cnt_ary;
  char *trans_bitvec;
  int blk, lb, ub;
  int *offsets;
  int *file_offset;

  //ARUN
  int num_trans;
  int maxitem;
  int avg_trans_sz;
  int max_trans_sz = 1;
  int *Host_DB_Buffer;
  int db_size;
  int records_per_thread;
  int remaining_trans;
  FILE *Dbase_file,*off_fp;


  //Gokcen
  uint64_t parallel_sec_start_uint, parallel_sec_end_uint;

  int max_depth;
  int thr_id_max_depth;


  t_io_uint = (uint64_t *) calloc(nBlocks, sizeof(uint64_t));

  local_tot_cand=(int *) calloc(nBlocks, sizeof(int));
  file_offset = (int *) calloc(nBlocks+1, sizeof(int));


  //ts_uint = rdtsc();
  parallel_sec_start_uint = rdtsc();

  // READ DATABASE FILE AND TRANSFER TO GPU
  Dbase_file = fopen(infile,"r");
    if (Dbase_file < 0){
      printf("Error couldn't open tmp file\n");
      exit(-1);
    }
  fread(&num_trans,sizeof(int), 1, Dbase_file);
  fread(&maxitem,sizeof(int), 1, Dbase_file);
  fread(&avg_trans_sz,sizeof(int), 1, Dbase_file);

  DBASE_NUM_TRANS = num_trans;
  DBASE_MAXITEM = maxitem;
  DBASE_AVG_TRANS_SZ = avg_trans_sz;
  MINSUPPORT = (int)(MINSUP_PER*DBASE_NUM_TRANS+0.5);
  //ensure that support is at least 2
  if (MINSUPPORT < 2) MINSUPPORT = 2;

  cout<<"\nDBASE_NUM_TRANS:"<<DBASE_NUM_TRANS<<" DBASE_MAXITEM:"<<DBASE_MAXITEM<<" DBASE_AVG_TRANS_SZ:"<<DBASE_AVG_TRANS_SZ<<" MIN SUPPORT:"<<MINSUPPORT;
  // CALCULATE DATABASE OFFSETS BASED ON NUMBER OF THREADS
  records_per_thread =  ceil((double)DBASE_NUM_TRANS/(double)nBlocks);
  cout<<"records_per_thread:"<<records_per_thread;
  remaining_trans = DBASE_NUM_TRANS - ((nBlocks-1) * records_per_thread);


  file_offset[0] = 0;
  int temp_tid,temp_numOfItems,temp_item,trans_counter=0,thread_counter=1;
  int temp_size = 0,temp_index=0;
  int *host_item_offsets,*host_offsets;
  int offt =0,item_offset_counter=0;
  int host_cnt_ary[(maxitem * (maxitem - 1))/2];
  for (i=0; i<(maxitem * (maxitem - 1))/2; i++)
      host_cnt_ary[i]=0;
  transaction *host_data_set;

  host_item_offsets=(int*)calloc(DBASE_NUM_TRANS,sizeof(int));

  host_offsets = new int [DBASE_MAXITEM];
  for (i=DBASE_MAXITEM-1; i >= 0; i--)
  {
	host_offsets[DBASE_MAXITEM-i-1] = offt;
	offt += i;
  }
  host_data_set = (transaction *) calloc (num_trans, sizeof(transaction));

  size_t return_value;
  for(i=0;i<DBASE_NUM_TRANS;i++)
  {
	  //Read and count the tid and the num of item, count the num of items, continue till you have reached records_per_thread
	  	  fread(&temp_tid,sizeof(int),1,Dbase_file);
	  	  temp_size++;
	  	  host_item_offsets[i]=item_offset_counter;
	  	  return_value=fread(&temp_numOfItems,sizeof(int),1,Dbase_file);
	  	  if(return_value==0)
	  		  temp_numOfItems=0;
	  	  temp_size++;
	  	  host_data_set[i].tid = temp_tid;
		  host_data_set[i].numitem = temp_numOfItems;
		  temp_index=host_data_set[i].offset = host_item_offsets[i];

	  	  item_offset_counter+=temp_numOfItems;
	  	  host_data_set[i].item_list = (int *) calloc (host_data_set[i].numitem, sizeof(int));
	  	  for(j=0;j<temp_numOfItems;j++)
	  	  {
	  		  fread(&temp_item,sizeof(int),1,Dbase_file);
	  		  temp_size++;
	  		  host_data_set[i].item_list[j] = temp_item;

	  	  }
		   for (j=0; j < host_data_set[i].numitem -1; j++)
		   {
			   idx = host_offsets[host_data_set[i].item_list[j]]-host_data_set[i].item_list[j]-1;
			   for (k=j+1; k < host_data_set[i].numitem; k++)
			   {
				  host_cnt_ary[idx+host_data_set[i].item_list[k]]++;
			   }
		   }
		  trans_counter++;
		  // Last thread may have fewer than everyone else
		  if(trans_counter == records_per_thread )
		  {
			  file_offset[thread_counter]=temp_size;
			  trans_counter=0;
			  thread_counter++;
		  }
  }

  cout<<"\nthread_counter"<<thread_counter;
  if(records_per_thread != remaining_trans)
	  file_offset[thread_counter]=temp_size;
  cout<<"\nfile offset: done ";
  cout<<"\nitem_offset_counter:"<<item_offset_counter;

  if(thread_counter >= nBlocks)
	  db_size = file_offset[nBlocks];  // db_size is actually number of integers in the database
  else
	  db_size=file_offset[nBlocks];
/*
	FILE *count;
	count=fopen("SIMT_count.txt","w");
	for(int ll=0;ll<(DBASE_MAXITEM * (DBASE_MAXITEM -1)/2);ll++)
		fprintf(count,"\n%d",host_cnt_ary[ll]);
	fclose(count);*/

 CUDA_entrypoint(host_data_set,host_cnt_ary,records_per_thread,nBlocks,nthreads);


}

void init_var()
{
  int i;
  start = new int *[nBlocks];
  enda = new int *[nBlocks];
  hash_pos = new int *[nBlocks];
  for (i=0; i < nBlocks; i++) hash_pos[i] = NULL;

  max_trans_sz = new int[nBlocks];
  for (i=0; i < nBlocks; i++) max_trans_sz[i] = 0;

  lsupcnt = new int *[nBlocks];
  for (i=0; i < nBlocks; i++) lsupcnt[i] = NULL;
  more = 1;
}

void clean_up(){
  int i;

  delete [] hash_indx;

  for (i=0; i < nBlocks; i++){
    delete start[i];
    delete enda[i];
    delete hash_pos[i];
  }
  delete [] start;
  delete [] enda;
  delete [] hash_pos;
  for (i=0; i < nBlocks; i++) delete lsupcnt[i];
  delete [] lsupcnt;
}

int main(int argc, char **argv)
{
  char sumfile[100];
  parse_args(argc, argv);


  sprintf(sumfile, "temp");
  if ((summary = fopen (sumfile, "a+")) == NULL){
    printf("can't open %s\n", sumfile);
    exit(-1);
  }


  init_var();
  fprintf(summary, "Database %s sup %f\n",
      infile, MINSUP_PER);
  printf("Database %s sup %f\n",
      infile, MINSUP_PER);
  fprintf(summary, "nBlocks= %d\n ", nBlocks);
  printf("nBlocks= %d\n", nBlocks);

  ts_uint = rdtsc();
  if (more) main_proc(NULL);
  te_uint = rdtsc();

  //max_io_time = t_io_uint[0];
  //for (int j=1; j<nBlocks; j++)
 //   if (t_io_uint[j]>max_io_time) max_io_time = t_io_uint[j];

  printf("Cands= %d\n",tot_cand);
  printf("TOTAL_EXECUTION_TIME= %llu\n",te_uint-ts_uint);
  printf("IO_TIME= %llu\n",max_io_time);
  printf("COMPUTATION_TIME= %llu\n",te_uint-ts_uint-max_io_time);
  fflush(summary);
  fflush(stdout);
//  clean_up();
  fclose(summary);
  exit(0);
}



























