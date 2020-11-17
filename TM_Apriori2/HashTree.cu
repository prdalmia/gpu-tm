#ifndef __HASHTREE_H
#define __HASHTREE_H
#define BALT
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#define __noinline__
#endif
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
// #include <omp.h>

#include "pardhp.h"
#include "ListItemset.cu"

#define ACQUIRE 0
#define RELEASE 1

#define ACQUIRE_LOCK(location)\
	while(participated)\
	{\
		if ((atomicCAS(&(location),-1,thread_id) == -1) || (location == thread_id)) \
		{

#define RELEASE_LOCK(location)\
		atomicExch(&(location),-1);\
		participated=false;\
		}\
	}\

#define CCPD
#define YES 1
#define NO 0

#define debug1_printf(...) 
#define debug2_printf(...) 
#if DEBUG == 1
#define debug1_printf(...) printf(...)
#endif
#if DEBUG == 2
#define debug2_printf(...) printf(...)
#endif

// Only one of these should be enabled at a time. 
//#define TM_SYNC
//#define FGL_SYNC
#if defined(TM_SYNC) and defined(FGL_SYNC)
   #error Only one of TM_SYNC and FGL_SYNC may be enabled at a time 
#elif !defined(TM_SYNC) and !defined(FGL_SYNC)
   #error Need to enable either TM_SYNC or FGL_SYNC
#endif

class HashTree;
extern int * max_nested_depth;
// __device__ int device_lock_hashTree[4000];
__device__ int hash_index[1001];
#define NUM_HT_FREELIST 64
__device__ int *HashTree_FreeList_cntr;
__device__ HashTree *device_Candidate;
__device__ __shared__  int intraCTALock ;

/*__device__  int addr_iteration;
__device__  int *addr_touched[120][2000];
__device__  int addr_counter[120];*/

#ifdef TM_SYNC
__device__ __noinline__ void __tbegin() { } 
__device__ __noinline__ void __tcommit() { }
__device__ __host__ void __taccessmode(int readwritemode) { }
#else
// inlining the function to remove the function call overhead
__device__ __host__ void __taccessmode(int readwritemode) { }
#endif

class HashTree
{
public:

	void *operator new(size_t size)
	{
		HashTree * hashtree = (HashTree*)malloc(size);
		return (void *)hashtree;
	}


	void operator delete(void *ptr_hashtree)
	{
		HashTree * ptr = (HashTree*) ptr_hashtree;
		free(ptr);
	}


	void *operator new[] (size_t size)
	{
		HashTree * hashtree = (HashTree*)malloc(size);
		return (void *)hashtree;
	}


	void operator delete[] (void *ptr_hashtree)
	{
		HashTree * ptr = (HashTree*) ptr_hashtree;
		free(ptr);
	}


	__device__ __host__ void HashTree_Init (int Depth_P, int hash, int thresh)
	{
      __taccessmode(0x02); 
		Leaf = YES;
		Count = 0;
		Hash_function = hash;
		Depth = Depth_P;
		Threshold = thresh;
		List_of_itemsets->numitem = 0;
		List_of_itemsets->First = NULL;
		List_of_itemsets->Last = NULL;
		node_lock=-1;
      __taccessmode(0x13); 
	}

	__device__ void clear()
	{
		if (Leaf){
			if (List_of_itemsets)
				delete List_of_itemsets;
			List_of_itemsets = NULL;
		}
		else{
			if (Hash_table){
				for(int i=0; i < Hash_function; i++){
					if (Hash_table[i])
						delete Hash_table[i];
				}
				//      free(Hash_table);
				Hash_table = NULL;
			}
		}
		Leaf = YES;
		Count = 0;
		Hash_function = 0;
		Threshold = 0;
	}

	__device__ int hash(int Value)
	{
		if(Value != 0)
			return (Value%Hash_function);
		else
			return 0;
	}

	__device__ void rehash(int debug,int thread_id,HashTree *device_HashTree_FreeList )
	{
		Leaf = NO;

		while(!(List_of_itemsets->first() == NULL))
		{
			// iterate over current itemsets
			ListElement *temp = List_of_itemsets->remove();
         int this_depth = Depth; 
			debug2_printf("\nNumber Of Items: %d ,				Depth %d   thread_id %d",temp->Item->theNumel,Depth,thread_id);
			debug2_printf(" Item: %d",temp->Item->item(Depth));
			__taccessmode(0x02);
			int val = hash_index[temp->Item->item(this_depth)];//according to current Depth
			__taccessmode(0x13);
			if (Hash_table[val]==NULL)
			{
				// counter_unique = atomicAdd(&HashTree_FreeList_cntr,1);
				int counter_unique;
            int &freeCntr = HashTree_FreeList_cntr[thread_id & (NUM_HT_FREELIST - 1)];
				#if defined(FGL_SYNC)
				counter_unique = atomicAdd(&freeCntr, 1);
				#else
				counter_unique = freeCntr++;
				#endif
				Hash_table[val] = &device_HashTree_FreeList[counter_unique];
				//collect_stats(&(HashTree_FreeList_cntr),thread_id);
				//collect_stats(&(device_HashTree_FreeList[counter_unique]),thread_id);
				debug2_printf("\n REHASH: Counter OLD: %d NEW:%d \n", counter_unique, freeCntr);
				Hash_table[val]->HashTree_Init(Depth+1, Hash_function, Threshold);
			}
			debug2_printf("\n 		HASH VALUE: %d",val);
			Hash_table[val]->add_element(debug,temp, thread_id,device_HashTree_FreeList);
		}
		//printf("%d \n Rehash compete!!!!!!!!!!!!!!" ,thread_id);
	}

	__device__ void add_element(int debug,ListElement *Value, int thread_id,HashTree *device_HashTree_FreeList)
	{

			int counter_unique,val;
         // int value_item_depth = Value->Item->item(Depth); 
         int value_item_numitems = Value->Item->numitems();
			//val = hash_index[Value->Item->item(Depth)];
			bool participated=true;
			int flag=0;

			#if defined(FGL_SYNC)
			ACQUIRE_LOCK(this->node_lock)
			#elif defined(TM_SYNC)
			__tbegin(); 
			#endif

					if (Leaf)
					{
						debug1_printf("\n  %d IN LEAF",thread_id);
						List_of_itemsets->append(Value);
						debug2_printf("\n %d APPENDED %d" ,thread_id,Count);
						if(List_of_itemsets->numitems() > Threshold)
						{
							debug2_printf("....debug1...");
							if (Depth+1 > value_item_numitems) // !!! CHANGED FROM ORIGINAL CODE!!!!
							{
								Threshold++;
								debug2_printf("....debug2...");
							}
							else
							{
								debug2_printf("\n thread_id %d REHASH!!!!!!!!!",thread_id);
								rehash(debug,thread_id,device_HashTree_FreeList);
								debug2_printf("\n !!!!!!!!!!!REHASH COMPLETE");
							}  	//if so rehash
						}
					}
					else
					{
						debug2_printf("\n %d NOT LEAF ",thread_id);
						__taccessmode(0x02);
                  int value_item_depth = Value->Item->item(Depth); 
						val = hash_index[value_item_depth];
						__taccessmode(0x13);
						debug1_printf("\n 		HASH VALUE: %d",val);
						//TRANSACTION_BEGIN
						if (Hash_table[val] == NULL)
						{
							// counter_unique = atomicAdd(&HashTree_FreeList_cntr,1);
                     int &freeCntr = HashTree_FreeList_cntr[thread_id & (NUM_HT_FREELIST - 1)];
							#if defined(FGL_SYNC)
							counter_unique = atomicAdd(&freeCntr, 1);
							#else
							counter_unique = freeCntr++;
							#endif
							Hash_table[val] = &device_HashTree_FreeList[counter_unique];
							debug2_printf("\n ADDELMT: Counter OLD: %d NEW: %d \n", counter_unique, freeCntr);
							Hash_table[val]->HashTree_Init(Depth+1, Hash_function, Threshold);
						}
						//TRANSACTION_END
						//printf("\n Value:%p ,Itemset:%p , Item:%p, Item[0]:%d",Value,Value->Item,Value->Item->theItemset,Value->Item->theItemset[0]);
						flag=1;
						//Hash_table[val]->add_element(debug,Value, thread_id,device_HashTree_FreeList);
					}

					debug1_printf("\n  lock released");


			#if defined(FGL_SYNC)
			RELEASE_LOCK(this->node_lock)
			#elif defined(TM_SYNC)
			__tcommit(); 
			#endif

			if(flag==1)
				Hash_table[val]->add_element(debug,Value, thread_id,device_HashTree_FreeList);

	}

	__device__ void printHashTree()
	{
		  if (Depth == 0)
		    printf(" ROOT : C: %d ,H: %d ",Count,Hash_function);;
		  if (Leaf){
		    if (List_of_itemsets != NULL){
		      printf(" \nD:%d T:%d",Depth,Threshold);
		      printf(" Number of items: %d ",List_of_itemsets->numitems());
		      for(int ss=0;ss<List_of_itemsets->numitems();ss++)
			  {
				  printf("\nItemset:");
				  for(int ll=0;ll< List_of_itemsets->node(ss)->item()->numitems(); ll++)
				  {
					  printf("%d ",List_of_itemsets->node(ss)->item()->item(ll));
				  }
				  printf("sup:%d", List_of_itemsets->node(ss)->item()->support);
			  }
		    }
		  }
		  else{
		    for(int i=0; i < Hash_function; i++){
		      if (Hash_table[i]){
		    	printf("\n");
		        printf("Depth: %d child: %d",Depth,i );
		        Hash_table[i]->printHashTree();
		      }
		    }
		    printf("\nReturning from Depth %d",Depth);
		  }

		  printf("\n***********************************************");

	}

	__device__ int is_root()
	{
		return (Depth == 0);
	}

	__device__ __host__ inline int is_leaf()
	{
		return (Leaf == YES);
	}

	__device__ __host__ inline ListItemset * list()
	{
		return List_of_itemsets;
	}

	__device__ __host__ inline int hash_function()
	{
		return Hash_function;
	}

	__device__ inline int depth()
	{
		return Depth;
	}

	__device__ inline int hash_table_exists()
	{
		return (Hash_table != NULL);
	}

	__device__ __host__ inline HashTree *hash_table(int pos)
	{
		return Hash_table[pos];
	}


public:
	int Leaf;
	HashTree **Hash_table;
	int Hash_function;
	int Depth;
	ListItemset *List_of_itemsets;
	int Count;
	int Threshold;
	int node_lock;
};

typedef HashTree * HashTree_Ptr;
#endif //__HASHTREE_H
