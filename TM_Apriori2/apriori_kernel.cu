
// To make CUDA Toolkit compatible with GCC 4.7
#undef _GLIBCXX_ATOMIC_BUILTINS

#include <iostream>
#include <fcntl.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <math.h>
#include <string.h>
#include <map>
#include <cutil_inline.h>


#include "Itemset.cu"
#include "ListItemset.cu"
#include "HashTree.cu"
#include "pardhp.h"
#include "global_constants.h"


struct check_indexes{
		int idx;
		int lb;
		int ub;
		int m;
		int blk;
	} *device_check_index,*check_index;

__device__ int g_barrier;
__device__ int temp_counter=0;
__device__ int device_hash_pos[8000][10000];
__device__ int trans_bitvec[8000][1001];
__device__ int start[8000][30];
__device__ int enda[8000][30];
//__device__ int *addr_touched[120][2000];
//__device__ int addr_counter[120];
//__device__ int addr_iteration;
__device__ int max_threshold;


__device__ __host__ int choose(int n, int k)
{
  int i;
  int val = 1;

  if (k >= 0 && k <= n){
	for (i=n; i > n-k; i--)
	  val *= i;
	for (i=2; i <= k; i++)
	  val /= i;
  }
  return val;
}

__device__ __host__ int get_hash_function(int num, int k)
	{
		int threshold =2 ;
	  int hash = (int)ceil(pow((float)num/threshold, (float)1.0/k));
	  if (hash < 1) hash = 1;
	  return hash;
	}

__device__ void gpu_interblock_sync(int goalVal)
{

	int tid = threadIdx.x;

	if (tid==0)
	{
		 atomicAdd(&g_barrier,1);
		//printf("\n%d     %d",blockIdx.x,oldVal);

	}

	while(g_barrier!=goalVal)
 	{

	}

}
__device__ __host__ void form_hash_indx( int *local_hash_index ,int hash_function, int maxitem)
{
  int i, cnt;
  i=0;

  printf("\n HASH_FUNCTION = %d", hash_function);
  if (hash_function == 1)
  {
    return;
  }

  while(i < maxitem){
    for(cnt = 0; i < maxitem && cnt < hash_function; i++)
      if (local_hash_index[i] == 0){
        local_hash_index[i] = cnt;
       //printf("\n i: %d, val:%d",i,hash_index[i]);
        cnt++;
      }
    for(cnt = hash_function-1;i < maxitem && cnt >= 0; i++)
      if (local_hash_index[i] == 0){
        local_hash_index[i] = cnt;
      //printf("\n i: %d, val:%d",i,hash_index[i]);
        cnt--;
      }
  }
}

__device__ int init_subsets(int *starts, int *endas, int num_item, int k_item)
{
  int i;

  if (num_item < k_item) return 0;

  for (i=0; i < k_item; i++){
    starts[i] = i;
    endas[i] = num_item - k_item + 1 + i;
  }
  return 1;
}

__device__ int get_next_subset(int *starts, int *endas, int k_item)
{
  int i,j;

  for (i=k_item-1; i >= 0; i--){
    starts[i]++;
    if (starts[i] < endas[i]){
      for (j=i+1; j < k_item; j++)
        starts[j] = starts[j-1]+1;
      return 1;
    }
  }
  return 0;
}


__device__ ListElement* find_in_list(Itemset *element, int sz, ListElement *head, int thread_id)
		{
		  for(;head; head = head->next()){
		    Itemset *curr = head->item();
		    for(int i=0; i < sz; i++){
		      int it = element->item((start[thread_id])[i]);
		      if (curr->item(i) < it)
		        break;
		      else if (curr->item(i) > it)
		        return NULL;
		      else if (i==sz-1) return head;
		    }
		  }
		  return NULL;
		}


__device__ int not_prune(ListElement *curr, int k, ListElement *beg, int thread_id)
{
  if (k+1 == 3){
    start[thread_id][0] = 1;
    start[thread_id][1] = 2;
    if ((beg = find_in_list(curr->Item, k, beg, thread_id)) == NULL) return 0;
  }
  else{
    int res = init_subsets(start[thread_id], enda[thread_id], curr->Item->numitems(), k);
    start[thread_id][k-2] = curr->Item->numitems()-2;
    start[thread_id][k-1] = curr->Item->numitems()-1;
    while (res){
      if ((beg = find_in_list(curr->Item, k, beg, thread_id)) == NULL) return 0;
      res = get_next_subset(start[thread_id], enda[thread_id], k);
    }
  }
  return 1;
}


__device__ int ass_cnt = 0;
__device__ void assign_lsupcnt_offset(HashTree *node, int &val)
{


	  if (node->is_leaf())
	  {
		ListItemset *list = node->list();

		/*printf("\n first: %p next: %p",list->first(),list->First->Next);
		printf("\n last: %p next: %p",list->last(),list->Last->Next);
		printf("\n numitem: %d",list->numitem);
		 */
		if(list && list->first())
		{
		  ListElement *iter = list->first();
		  for(;iter;iter = iter->next())
		  {
			  iter->item()->set_sup(val++);
		  }
		}
	  }
	  else
	  {
		for(int i=0; i < node->hash_function(); i++)
		  if (node->hash_table(i))
		  {
			  assign_lsupcnt_offset(node->hash_table(i), val);
		  }
	  }

}

__device__ inline void make_Itemset(Itemset *it, int *buf, int numitem, int tid)
{
  int j;

  it->set_tid(tid);
  it->set_numitems(numitem);
  for (j=0; j < numitem; j++){
    it->add_item(j, (int) buf[j]);
  }
}



__device__ void apriori_gen(int debug ,int total_threads,int k, int thread_id ,ListElement *tree_listElement,
							int *tree_listElement_cntr,ListItemset* Largelist,
							int*gen_check , HashTree * device_HashTree_FreeList,ListElement *temp_listElement)
{

	  //int ss,lb,ub,blk,index,counter=0,counter_2=0;
	  int lb,ub,blk;
	  blk = ceil((double)Largelist->numitem/(double)total_threads);

	  if(blk==0)
		  blk=1;

	  if(blk==1 && thread_id>=Largelist->numitem)
		lb = -1;
	  else
	  {
		  lb = thread_id *blk;
	  }

	  ub = min((thread_id+1)*blk, Largelist->numitem);

	  if(lb>=0 && ub<=Largelist->numitem &&lb<Largelist->numitem)
	  {
		 // if (debug==1)
		 //printf("\nApriori_gen thread_id= %d, blk= %d ,lb= %d, ub= %d, device_Largelist->numitems= %d nblocks= %d intraLock =%d", thread_id,blk, lb, ub, Largelist->numitem,total_threads,intraCTALock);
		  ListElement *L1iter = Largelist->node(lb);
		  for (int i=lb; i < ub && L1iter; i++, L1iter = L1iter->next())
		  {
			Itemset *temp = L1iter->item();

			//printf("\n thread_id: %d ,i:%d ,item: %d",thread_id,i,temp->item(0));
			ListElement *L2iter = L1iter->next();
			for(;L2iter; L2iter = L2iter->next())
			{
				if(debug==1)
					printf("\nbeg of inner loop,thread_id %d",thread_id);
			  Itemset *temp2 = L2iter->item();

			  if (temp->compare(*temp2,k-2) < 0) break;
			 else
			  {
				  if(debug==1)
				  		printf("\nelse part ,thread_id %d",thread_id);
				int k1 = temp->item(k-2);
				int k2 = temp2->item(k-2);
				if (k1 < k2)
				{
				// In order to save memory, we will use a temp listElement and if it passes the prune stage
				// we will allocate an actual listelement from the free list
				  ListElement *it_temp = &temp_listElement[thread_id];
				  it_temp->Item->set_numitems(k);
				  for (int l=0; l < temp->numitems(); l++)
					it_temp->Item->add_item(l,temp->item(l));
				  it_temp->Item->add_item(k-1, k2);
				  ListElement *beg = Largelist->first();

				  if(debug==1)
					printf("\n create item list  ,thread_id %d",thread_id);
				  if(k==2 || not_prune(it_temp, k-1, beg, thread_id))
				  {
					  if(debug==1)
						  printf("\n inside prune  ,thread_id %d",thread_id);
					  int val = atomicAdd(tree_listElement_cntr,1);

					  ListElement *it = &tree_listElement[val];
					  it->Item->set_numitems(k);
					  it->Item->support=-1;
					  for (int l=0; l < temp->numitems(); l++)
						  it->Item->add_item(l,temp->item(l));
					  it->Item->add_item(k-1, k2);
					  it->Item->set_sup(0);
					  if(debug==1)
						  printf("\nApriori_Kernel calls add element,thread_id:%d",thread_id);
					 debug=0;

					 device_Candidate->add_element(debug,it,thread_id,device_HashTree_FreeList);
					 debug=0;
					  if(debug==1)
						  printf("\nApriori_Kernel returns from add element,thread_id:%d",thread_id);

				  }
				}
			  }
			}
		  }
	  }
}


__device__ void increment(int debug,Itemset *trans, ListItemset *Clist, int *tbitvec, int *cnt_ary )
{


	if(Clist->first())
	{
		ListElement *head = Clist->first();
		for(;head; head = head->next())
		{
		  Itemset *temp = head->item();
		  if (temp->subsequence(tbitvec, trans->numitems()))
		  {
			  //atomicAdd(&cnt_ary[temp->sup()],1);
			  atomicAdd(&(temp->support),1);
		  }
		}

	}

}

__device__ void subset(int debug,Itemset *trans, int st, int en, int final,
    HashTree* node, int k, int level, int thread_id,
  int *tbitvec, int *vist, int hash_function, int *cnt_ary)
{

	int i;
  (*vist)++;
  int myvist = *vist;
  if (node == device_Candidate
      && node->is_leaf() && node->list() && node->list()->numitem>0)
  {
    increment(debug,trans, node->list(), tbitvec,cnt_ary);
  }
  else
  {
    for(i=st; i < en; i++)
    {
      int val = trans->item(i);
      int hashval = hash_index[val];
      if (hashval == -1) continue;

      if ((device_hash_pos[thread_id])[level*hash_function+hashval] != myvist)
      {
        (device_hash_pos[thread_id])[level*hash_function+hashval] = myvist;
        if (node->hash_table_exists() )
        {
			if(node->hash_table(hashval))

			{
			  if (node->hash_table(hashval)->is_leaf() &&
				  node->hash_table(hashval)->list() )
			  {
				  increment(debug,trans, node->hash_table(hashval)->list()
					, tbitvec,cnt_ary);
			  }
			  else if (en+1 <= final)
			  {

				 subset( debug,trans, i+1, en+1, final,node->hash_table(hashval),
					k, level+1, thread_id, tbitvec, vist, hash_function,cnt_ary);
			  }
			}
        }
      }
    }
  }
}



__device__ void form_large( int debug,
							int *device_counter_listElement,
							ListElement *device_free_listElement,
							ListItemset *device_Largelist ,
							HashTree *node,
							int k,
							int &cnt,
							int *cntary,
							int min_sup)
{



  if (node->is_leaf())
  {
	  ListItemset *list = node->list();
  if(list->numitem>0 && list->first())
    {
      ListElement *iter = list->first();
      for(;iter;iter = iter->next())
      {
    	  int temp_sup = iter->item()->sup() ;

    	  //iter->item()->set_sup(cntary[cnt++]);

       if (iter->item()->sup() >= min_sup)
        {

        	ListElement *element = &device_free_listElement[*device_counter_listElement];
        	(*device_counter_listElement)++;
        	element->Next=NULL;
        	element->Item->support = -1;
        	element->Item->copy(iter->item());
        	device_Largelist->sortedInsert(element, element->item());
/*
        	if(debug==1)
        	{
        		printf("\n Itemset:");
			for(int ss=0;ss<k;ss++)
			  {
				  printf("%d ",iter->item()->item(ss));
			  }
        	}*/
          for (int j=0; j < iter->item()->numitems(); j++)
           {
        	  hash_index[iter->item()->item(j)] = 0;

           }
        }
      }
    }
  }
  else{
    for(int i=0; i < node->hash_function(); i++)
      if (node->hash_table(i))
        form_large(debug,device_counter_listElement,device_free_listElement,device_Largelist,node->hash_table(i), k, cnt, cntary,min_sup);
  }

}




__global__ void Init_Kernel(int hash_function,
							int maxitem,
							HashTree *HashTree_FreeList,
							ListItemset *Largelist,
							ListElement *Largelist_listElement,
							int *Largelist_listElement_cntr,
							int *tree_listElement_cntr,
							int host_numitem)
{

	int threshold=2;
   for (int i = 0; i < NUM_HT_FREELIST; i++) 
      HashTree_FreeList_cntr[i] = max_HashFunction * i; 
	device_Candidate = &HashTree_FreeList[HashTree_FreeList_cntr[0]];
	HashTree_FreeList_cntr[0] += 1;
	device_Candidate->HashTree_Init(0, hash_function, threshold);
	Largelist->First= Largelist_listElement;
	Largelist->Last= Largelist_listElement + host_numitem-1;
	Largelist->numitem=host_numitem;
	*Largelist_listElement_cntr=host_numitem;
	/*int i;
	ListElement *temp = Largelist->First;
	for(i=0;i<host_numitem;i++)
	{
		printf("%d %d %d \n",temp->Item->theItemset[0],temp->Item->theItemset[1],temp->Item->support);
		temp=temp->Next;
	}
	*Largelist_listElement_cntr=host_numitem;
	printf("\n\nelement cntr: %d\n",*Largelist_listElement_cntr );
	printf("\n\nelement cntr: %d\n",*tree_listElement_cntr );*/
}



__global__ void AprioriGen_Kernel(int debug,
									int device_k,
									int total_threads,
									ListElement *tree_listElement,
									int *tree_listElement_cntr,
									ListItemset *Largelist,
									HashTree *HashTree_FreeList,
									int *gen_check,
									ListElement *temp_listElement)
{

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	debug=0;
	apriori_gen(debug,total_threads,device_k, thread_id,tree_listElement,tree_listElement_cntr,Largelist,gen_check,HashTree_FreeList,temp_listElement);



}

__global__ void Subset_Kernel( int debug,
								int device_k,
								int records_per_thread,
								int num_trans,
								int maxitem,
								transaction *data_set,
								Itemset *temp_Itemset,
								int *cnt_ary,
								ListItemset *Largelist)
{

	int blk,lb,ub,ll,i,vist,j,temp,device_more;
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	//if(thread_id==0)
		//printf("\nSubset_Kernel Start");
	blk = records_per_thread;
	lb = thread_id*blk;
	ub = min((thread_id+1)*blk, num_trans);
	vist = 1;

	device_more = (choose(Largelist->numitem,2)> 0);
	//printf("\n%d lb=%d ub=%d records_per_thread=%d",thread_id,lb,ub,records_per_thread);

	for (i=0; i <maxitem; i++){
	      trans_bitvec[thread_id][i]=0;
	}

	temp=((device_k)+1)*device_Candidate->Hash_function;

	for (ll=0; ll<temp; ll++)
			device_hash_pos[thread_id][ll]=0;
	if (device_more)
	{
		for(i=lb; i < ub;i++)
		{
			//printf("\n thread_id=%d lb=%d ub=%d records_per_thread=%d",thread_id,lb,ub,records_per_thread);
		  make_Itemset(&temp_Itemset[thread_id], data_set[i].item_list, data_set[i].numitem, data_set[i].tid);

		  for (j=0; j < temp_Itemset[thread_id].numitems(); j++)
			trans_bitvec[thread_id][temp_Itemset[thread_id].item(j)] = 1;

		  subset(debug,&temp_Itemset[thread_id], 0, temp_Itemset[thread_id].numitems()-(device_k)+1,temp_Itemset[thread_id].numitems(),
			 device_Candidate,device_k, 0, thread_id, trans_bitvec[thread_id], &vist, device_Candidate->hash_function(),cnt_ary);


		  for (j=0; j < temp_Itemset[thread_id].numitems(); j++)
			trans_bitvec[thread_id][temp_Itemset[thread_id].item(j)] = 0;

		}

	}


	if(thread_id==0)
		{
		printf("\nSubset_Kernel END");

		}

}

__device__ void print_large(ListItemset *device_Largelist)

{

	for(int jj=0;jj<device_Largelist->numitems();jj++)
	{
		ListElement *iter = device_Largelist->node(jj);
		 printf("\n Itemset:");
		  for(int ss=0;ss<iter->item()->numitems();ss++)
		  {
			  printf("%d ",iter->item()->item(ss));
		  }
		  printf("\n sup:%d",iter->item()->sup());

	}

}


__global__ void FormLargeList( int debug,
								int *device_more,
								int device_k,
								ListElement *Largelist_listElement,
								int *Largelist_listElement_cntr,
								ListItemset *Largelist,
								int maxitem,
								int min_sup,
								int *cnt_ary,
								HashTree *HashTree_FreeList)
{
	int ccnntt=0,i;
	int NUM_INSERT,hash_function,threshold=2;
	//printf("\nLargelist_element_counter: %d",*Largelist_listElement_cntr);
	*Largelist_listElement_cntr=0;
	//temp_counter=0;
	Largelist->numitem=0;
	Largelist->First=NULL;
	Largelist->Last=NULL;
	for(i=0;i<maxitem;i++)
		hash_index[i]=-1;
	if(device_k==4)
		debug=1;
	form_large(debug,Largelist_listElement_cntr,Largelist_listElement,Largelist,device_Candidate,device_k, ccnntt, cnt_ary,min_sup);
	printf("\n\n(%d .ITER)it= %d", device_k, Largelist->numitems());
	*device_more=(Largelist->numitems()>1);
	if(*device_more)
	{
      print_large(Largelist);
      NUM_INSERT = choose(Largelist->numitems(),2);
      hash_function = get_hash_function(NUM_INSERT,device_k+1);
      printf("\nHash :%d",hash_function);
      form_hash_indx(hash_index,hash_function,maxitem);
      for (int i = 0; i < NUM_HT_FREELIST; i++) 
         HashTree_FreeList_cntr[i] = max_HashFunction * i; 
      device_Candidate = &HashTree_FreeList[HashTree_FreeList_cntr[0]];
      HashTree_FreeList_cntr[0] += 1; 
      device_Candidate->HashTree_Init(0, hash_function, threshold);
	}
	else
	{
      print_large(Largelist);
	}
}

class stats {
	public:
		stats()
		{
			total=0;
			for(int i=0;i<120;i++)
				transactions[i]=0;

		}
		int total;
		int transactions[120];
	};

void CUDA_entrypoint(transaction *host_data_set,int *host_cnt_ary,int records_per_thread,int nBlocks,int nthreads)
{


	int *cnt_ary,*tree_listElement_cntr,*Largelist_listElement_cntr,*offsets;
	int *gen_check,*device_more,*host_offsets;
	int  device_k,k,more,total_threads,i,j,idx,total_combinations,debug,temp_int,offt=0;
	//int div_const=1024*1024;
	transaction *data_set;
	//size_t total,free=0;
	Itemset *temp_itemset,*temp_itemset_ptr ;
	void *temp_ptr,*temp_ptr_2;
	ListItemset *Largelist,*Host_Largelist,*temp_list;
	ListElement *tree_listElement,*Largelist_listElement,*temp_listElement;;
	HashTree *HashTree_FreeList, **temp_HashTree;
	int host_hash_index[1001];


	cudaSetDevice(0);

	cout<<"\nDBASE_NUM_TRANS:"<<DBASE_NUM_TRANS<<" DBASE_MAXITEM:"<<DBASE_MAXITEM<<" DBASE_AVG_TRANS_SZ:"<<DBASE_AVG_TRANS_SZ<<" MIN SUPPORT:"<<MINSUPPORT;
    cout<<" records_per_thread:"<<records_per_thread<<"\n";

    //cudaMemGetInfo(&free,&total);
    //cout<<"\n!!FREE!!:"<<free/(div_const);


	offt=0;
	host_offsets = new int [DBASE_MAXITEM];
	for (i=DBASE_MAXITEM-1; i >= 0; i--)
	{
		host_offsets[DBASE_MAXITEM-i-1] = offt;
		offt += i;
	}

    // offsets - Used to figure out the index of an candidate set for  the count array
	cutilSafeCall(cudaMalloc(&offsets,(size_t)DBASE_MAXITEM * sizeof(int)));
	cutilSafeCall(cudaMemcpy(offsets,host_offsets,(size_t)DBASE_MAXITEM * sizeof(int),cudaMemcpyHostToDevice));

    // data_set  Data set used to store items from DB
    cutilSafeCall(cudaMalloc(&data_set,(size_t)DBASE_NUM_TRANS * sizeof(transaction)));
    for(i=0;i<DBASE_NUM_TRANS;i++)
    {
    	cutilSafeCall(cudaMemcpy(&(data_set[i].tid),&host_data_set[i].tid,sizeof(int),cudaMemcpyHostToDevice));
    	cutilSafeCall(cudaMemcpy(&(data_set[i].numitem),&host_data_set[i].numitem,sizeof(int),cudaMemcpyHostToDevice));
    	temp_int=host_data_set[i].numitem;
    	//allocate space for item array and copy the items into it
    	cutilSafeCall(cudaMalloc(&temp_ptr,(size_t)temp_int *sizeof(int)));
    	cutilSafeCall(cudaMemcpy(&(data_set[i].item_list),&temp_ptr,sizeof(int*),cudaMemcpyHostToDevice));
    	cutilSafeCall(cudaMemcpy(temp_ptr,host_data_set[i].item_list,(size_t)temp_int*sizeof(int),cudaMemcpyHostToDevice));

    }

    printf("Done 1 ");

    // cnt_ary Array used to count the support for items in the hash tree
	total_combinations = DBASE_MAXITEM * (DBASE_MAXITEM -1)/2; // N *N-1/2, given N items it is the max. number of combination
	cutilSafeCall(cudaMalloc (&cnt_ary ,(size_t)(total_combinations)*sizeof(int)));
	cutilSafeCall(cudaMemcpy(cnt_ary,host_cnt_ary,(total_combinations)*sizeof(int),cudaMemcpyHostToDevice));





	// tree_listElement - Allocation mem for listElements used in hash tree
	cutilSafeCall(cudaMalloc(&(tree_listElement),((size_t)sizeof(ListElement)*max_large_list_size)));
	for(i=0;i<max_large_list_size;i++)
	{

		cutilSafeCall(cudaMalloc(&(temp_itemset),(size_t)sizeof(Itemset)));
		cutilSafeCall(cudaMemcpy(&(tree_listElement[i].Item),&(temp_itemset),sizeof(Itemset*),cudaMemcpyHostToDevice));

		// since the array is the first element of the Itemset class, we can use the location pointed to be temp_itemset to allocate space for this array

		cutilSafeCall(cudaMalloc(&(temp_ptr),(size_t)sizeof(int)*max_candidate_size));
		cutilSafeCall(cudaMemcpy(temp_itemset,&(temp_ptr),sizeof(int*),cudaMemcpyHostToDevice));

	}

	printf("Done 3 ");
	// tree_listElement_cntr - Counter for ListElements used in hash tree
	cutilSafeCall(cudaMalloc(&tree_listElement_cntr,(size_t)sizeof(int)));
	cutilSafeCall(cudaMemset(tree_listElement_cntr,0,sizeof(int)));


	// HashTree_FreeList - Allocating memory for the hash tree and the free list counters 
   int * h_HashTree_FreeList_cntr; 
   cutilSafeCall(cudaMalloc(&h_HashTree_FreeList_cntr,sizeof(int) * NUM_HT_FREELIST)); 
   cutilSafeCall(cudaMemcpyToSymbol("HashTree_FreeList_cntr", &h_HashTree_FreeList_cntr, sizeof(int*), 0, cudaMemcpyHostToDevice)); 
	cutilSafeCall(cudaMalloc(&HashTree_FreeList,(size_t)(sizeof(HashTree)*max_HashFunction*NUM_HT_FREELIST)));
   size_t total_HashTable_size = max_HashFunction * max_HashFunction * NUM_HT_FREELIST;
   HashTree ** all_HashTree = NULL; 
	cutilSafeCall(cudaMalloc( &(all_HashTree),(size_t)sizeof(HashTree*)*total_HashTable_size));
	cutilSafeCall(cudaMemset(all_HashTree,0,sizeof(HashTree*)*total_HashTable_size));
	printf("HashTreeFreeList pointer: %p, max_HashFunction = %d\n", HashTree_FreeList, max_HashFunction);
	for(i=0;i<max_HashFunction*NUM_HT_FREELIST;i++)
	{
      temp_HashTree = &(all_HashTree[i * max_HashFunction]); 
		cutilSafeCall(cudaMemcpy(&(HashTree_FreeList[i].Hash_table),&(temp_HashTree),sizeof(HashTree_FreeList[i].Hash_table),cudaMemcpyHostToDevice));

		cutilSafeCall(cudaMalloc(&(temp_list),(size_t)sizeof(ListItemset)));
		cutilSafeCall(cudaMemcpy( &(HashTree_FreeList[i].List_of_itemsets),&(temp_list),sizeof(ListItemset*),cudaMemcpyHostToDevice));

		//Init Values
		cutilSafeCall(cudaMemset(temp_list,0,sizeof(ListElement*)));
		cutilSafeCall(cudaMemset(&(temp_list->Last),0,sizeof(ListElement*)));
		cutilSafeCall(cudaMemset(&(temp_list->numitem),0,sizeof(int)));

      if (i % 100 == 0) printf("\t progressing %d\n", i); 
	}

	// set some initial values
	more=1;
	k=3;
	total_threads = nBlocks * nthreads;
	records_per_thread = ceil((double)DBASE_NUM_TRANS/(double)total_threads);


	// temp_listElement - allocating listElements which are used in apriori_gen to hold some temp values
	printf("Allocation: temp_listElement;  total_threads = %d\n", total_threads); 
	cutilSafeCall(cudaMalloc(&(temp_listElement),(size_t)sizeof(ListElement)*total_threads));
	for(i=0;i<total_threads;i++)
	{

		cutilSafeCall(cudaMalloc(&(temp_itemset),(size_t)sizeof(Itemset)));
		cutilSafeCall(cudaMemcpy(&(temp_listElement[i].Item),&(temp_itemset),sizeof(Itemset*),cudaMemcpyHostToDevice));

		// since the array is the first element of the Itemset class, we can use the location pointed to be temp_itemset to allocate space for this array

		cutilSafeCall(cudaMalloc(&(temp_ptr),(size_t)sizeof(int)*max_candidate_size));
		cutilSafeCall(cudaMemcpy(temp_itemset,&(temp_ptr),sizeof(int*),cudaMemcpyHostToDevice));

      if (i % 100 == 0) printf("\t progressing %d\n", i); 
	}

	// temp_itemset - allocating itemsets which are used in subset to hold some temp values
	printf("Allocation: temp_itemset;  total_threads = %d\n", total_threads); 
	cutilSafeCall(cudaMalloc(&(temp_itemset),(size_t)sizeof(Itemset)*total_threads));
	for(i=0;i<total_threads;i++)
	{
		cutilSafeCall(cudaMalloc(&temp_itemset_ptr,(size_t)sizeof(int)*max_trans_size));
		cutilSafeCall(cudaMemcpy(&temp_itemset[i],&temp_itemset_ptr,sizeof(temp_itemset_ptr),cudaMemcpyHostToDevice));
      if (i % 100 == 0) printf("\t progressing %d\n", i); 
	}

	// device_more - outer loop variable
	cutilSafeCall(cudaMalloc(&device_more,(size_t)sizeof(int)));

	//device_check_index - used for debugging purposes
	cutilSafeCall(cudaMalloc(&device_check_index , (size_t)(nBlocks* sizeof(check_indexes))));

	// gen_check - used for debugging purposes
	cutilSafeCall(cudaMalloc(&gen_check,(size_t)sizeof(int)*10000));

	//cudaMemGetInfo(&free,&total);
	//cout<<"\n!!FREE!!:"<<free/(div_const);

	Host_Largelist = new ListItemset();
	for(i=0; i <DBASE_MAXITEM; i++)
		host_hash_index[i] = -1;

	for(i=0; i < DBASE_MAXITEM-1; i++){
		idx = host_offsets[i]-i-1;
		for (j=i+1; j < DBASE_MAXITEM; j++)
		{
			if (host_cnt_ary[idx+j] >= MINSUPPORT)
			{
				host_hash_index[i] = 0;
				host_hash_index[j] = 0;
				Itemset *it =  new Itemset(2);
				it->set_numitems(2);
				it->add_item(0,i);
				it->add_item(1,j);
				it->set_sup(host_cnt_ary[idx+j]);
				ListElement *element = new ListElement(it);
				Host_Largelist->append(element);

			}
		}
	}

	ListElement *Host_TempListElement = Host_Largelist->First;

	int host_numitem = Host_Largelist->numitem;
	// Largelist_listElemenent - Allocating mem for listElements used in largelist
	printf("Allocation: Largelist_listelement;  max_large_list_size = %d\n", max_large_list_size); 
	cutilSafeCall(cudaMalloc(&(Largelist_listElement),((size_t)sizeof(ListElement)*max_large_list_size)));
	for(i=0;i<max_large_list_size;i++)
	{
		cutilSafeCall(cudaMalloc(&(temp_itemset_ptr),(size_t)sizeof(Itemset)));
		cutilSafeCall(cudaMemcpy(&(Largelist_listElement[i].Item),&(temp_itemset_ptr),sizeof(Itemset*),cudaMemcpyHostToDevice));
		// since the array is the first element of the Itemset class, we can use the location pointed to be temp_itemset to allocate space for this array
		cutilSafeCall(cudaMalloc(&(temp_ptr),(size_t)sizeof(int)*max_candidate_size));
		cutilSafeCall(cudaMemcpy(temp_itemset_ptr,&(temp_ptr),sizeof(int*),cudaMemcpyHostToDevice));
		if(i<host_numitem)
		{
			cutilSafeCall(cudaMemcpy(temp_ptr,Host_TempListElement->Item->theItemset,sizeof(int)*2,cudaMemcpyHostToDevice));
			temp_ptr_2 = &temp_itemset_ptr->theNumel;
			cutilSafeCall(cudaMemcpy(temp_ptr_2,&Host_TempListElement->Item->theNumel,sizeof(int),cudaMemcpyHostToDevice));
			temp_ptr_2 = &temp_itemset_ptr->support;
			cutilSafeCall(cudaMemcpy(temp_ptr_2,&Host_TempListElement->Item->support,sizeof(int),cudaMemcpyHostToDevice));
			Host_TempListElement = Host_TempListElement->Next;
		}


      if (i % 100 == 0) printf("\t progressing %d\n", i); 
	}

	//set the next pointers in the Largelist

	for(i=0;i<host_numitem;i++)
	{
		temp_ptr_2 = &(Largelist_listElement[i+1]);
		cutilSafeCall(cudaMemcpy(&(Largelist_listElement[i].Next),&(temp_ptr_2),sizeof(ListElement*),cudaMemcpyHostToDevice));
	}

	cutilSafeCall(cudaMemset(&(Largelist_listElement[host_numitem-1].Next),0,sizeof(ListElement*)));

	// Largelist_listElement_cntr - counter for listElements used in Largelist
	cutilSafeCall(cudaMalloc(&Largelist_listElement_cntr,(size_t)sizeof(int)));
	cutilSafeCall(cudaMemset(Largelist_listElement_cntr,host_numitem,sizeof(int)));

	// Largelist is linked list of candidate sets
	cutilSafeCall(cudaMalloc(&Largelist,(size_t)sizeof(ListItemset)));

	int NUM_INSERT = choose(host_numitem,2);
	int hash_function = get_hash_function(NUM_INSERT,2);

	form_hash_indx(host_hash_index,hash_function,DBASE_MAXITEM);

	//copy the host_hash_index to hash_index
	cutilSafeCall(cudaMemcpyToSymbol(hash_index,&host_hash_index,sizeof(int)*1001));

	printf("\n(2.ITER)it= %d", Host_Largelist->numitems());
	printf("\nNUM_INSERT: %d",NUM_INSERT);

	Init_Kernel<<<1,1>>>(hash_function,
						 DBASE_MAXITEM,
						 HashTree_FreeList,
						 Largelist,
						 Largelist_listElement,
						 Largelist_listElement_cntr,
						 tree_listElement_cntr,
						 host_numitem);

	cout<<"\n Loop Start";
	// cudaThreadSetLimit(cudaLimitStackSize,4096);
	for (k=3;more;k++)
	{
		debug=0;
		device_k=k;

		//cudaMemset(&g_barrier,0,sizeof(int));
		cutilSafeCall(cudaMemset(tree_listElement_cntr,0,sizeof(int)));

//		if(k==3)
//			cudaMemset(&addr_iteration,1,sizeof(int));
//		else
//			cudaMemset(&addr_iteration,0,sizeof(int));

		AprioriGen_Kernel<<<nBlocks,nthreads>>>(debug,
										 device_k,
										 total_threads,
									     tree_listElement,
									     tree_listElement_cntr,
									     Largelist,
									     HashTree_FreeList,
									     gen_check,
									     temp_listElement);

		cutilSafeCall(cudaMemset(cnt_ary,0,(total_combinations)*sizeof(int)));



		Subset_Kernel<<<nBlocks,nthreads>>>(debug,
									 device_k,
									 records_per_thread,
									 DBASE_NUM_TRANS,
									 DBASE_MAXITEM,
									 data_set,
									 temp_itemset,
									 cnt_ary,
									 Largelist);

		cudaThreadSynchronize();


		FormLargeList<<<1,1>>>( debug,
							   device_more,
							   device_k,
							   Largelist_listElement,
							   Largelist_listElement_cntr,
							   Largelist,
							   DBASE_MAXITEM,
							   MINSUPPORT,
							   cnt_ary,
							   HashTree_FreeList);

		cudaThreadSynchronize();


		cutilSafeCall(cudaMemcpy(&more,device_more,sizeof(int),cudaMemcpyDeviceToHost));

      printf("Cleanup HashTreeFreeList Hash_table\n"); 
      cutilSafeCall(cudaMemset(all_HashTree, 0, sizeof(HashTree*)*total_HashTable_size));
	}
	cudaThreadSynchronize();

	/*****************CACHE LINES TOUCHED BY TANSACTIONS **************************************************************/
//	int cache_line = 128; // bytes
//	int *host_addr_touched[120][2000],host_addr_counter[120];
//	unsigned long key;
//
//	std::map<unsigned long,stats>cacheline_trans;
//
//	for(i=0;i<120;i++)
//	{
//	cutilSafeCall(cudaMemcpy(host_addr_touched[i],addr_touched[i],2000*sizeof(int*),cudaMemcpyDeviceToHost));
//	}
//	cutilSafeCall(cudaMemcpy(host_addr_counter,addr_counter,120*sizeof(int),cudaMemcpyDeviceToHost));
//
//	printf("First memory locations: ");
//	for(i=0;i<120;i++)
//	printf("  %p",host_addr_touched[i][0]);
//
//	for(i=0;i<120;i++)
//	{
//		for(j=0;j<2000;j++)
//		{
//			key=(unsigned long)host_addr_touched[i][j];
//
//			if( (key%cache_line) == 0 ) // Start of Cache Line
//			{
//
//				cacheline_trans[key].total++;
//				cacheline_trans[key].transactions[i]=1;
//
//			}
//			else
//			{
//				key = (key-(key%128));// find the start of the cache line
//				cacheline_trans[key].total++;
//				cacheline_trans[key].transactions[i]=1;
//			}
//		}
//	}

	/*******************************************************************************************************************/
/*
	FILE *fp = fopen("cuda_elements_gen","w");
	host_gen_check =(int *)calloc(10000,sizeof(int));
	cutilSafeCall(cudaMemcpy(host_gen_check,gen_check,10000*sizeof(int),cudaMemcpyDeviceToHost));
	fprintf(fp,"\n");
	int kk=0;
	for(i=0;i<10000;i++)
	{
		if(host_gen_check[i]>0 && host_gen_check[i]<1000)
		{
			fprintf(fp,"%d ",host_gen_check[i]);
			kk+=1;
			if(kk==3)
			{
				fprintf(fp,"\n");
				kk=0;
			}
		}
	}
*/


/*	check_index = (check_indexes*)calloc(nBlocks,sizeof(check_indexes));
	cutilSafeCall(cudaMemcpy(check_index,device_check_index,nBlocks*sizeof(check_indexes),cudaMemcpyDeviceToHost));
	printf("\nIndexes: ");
	for(i=0;i<nBlocks;i++)
	{
		printf("\nidx: %d",check_index[i].idx);
		printf("\nlb: %d",check_index[i].lb);
		printf("\nub: %d",check_index[i].ub);
		printf("\nblk: %d",check_index[i].blk);
		printf("\nm: %d",check_index[i].m);
		printf("\n");

	} */


	/*	host_cnt_ary=(int*)calloc(total_combinations,sizeof(int));
		cutilSafeCall(cudaMemcpy(host_cnt_ary,cnt_ary,(total_combinations)*sizeof(int),cudaMemcpyDeviceToHost));
	FILE *output_file_count;
	output_file_count= fopen("cuda_output_count.txt","w");
	for(i=0;i<2867;i++)
	{
		fprintf(output_file_count,"\n%d",host_cnt_ary[i]);
	}
	fclose(output_file_count);*/

	/*FILE *output_file;
	output_file=fopen("cuda_output_data.txt","w");
	for(i=0;i<DBASE_NUM_TRANS;i++)
	  {
		  //  fprintf(output_file,"%d %d",host_data_set[i].tid,host_data_set[i].numitem);
		  for(j=0;j<host_data_set[i].numitem;j++)
		  {
			  int tmp_index =  host_item_offsets[i]+ j;
			fprintf(output_file,"%d ",host_item_array[tmp_index]);
		  }
		  fprintf(output_file,"\n");
	  }
	fprintf(output_file,"\n");
	fclose(output_file);*/

	printf("\nDONE\n");
}
