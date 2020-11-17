
#ifndef __ITEMSET_H
#define __ITEMSET_H
#include "pardhp.h"
#include <malloc.h>
#include<stdio.h>
#include <stddef.h>
#if defined CCPD
#include "llsc.h"
#endif


#define DEFAULT -1

class Itemset
{

public:

	  Itemset (int sz)
	{
		theNumel = 0;
		if (sz <= 0) sz = 1;
		theSize = sz;
		theItemset=(int*)malloc(sizeof(int)*sz);
		for(int i=0; i < theSize; i++) theItemset[i] = DEFAULT;
		support =0;
		Tid = -1;
	}


	void clear()
	{
		free(theItemset);
		theNumel = 0;
		theSize = 0;
		support = 0;
	}


	~Itemset()
	{
		clear();
	}

	__device__ __host__  int item(int pos)
	{
		return theItemset[pos];
	}


	__device__ __host__  int numitems()
	{
		return theNumel;
	}


	__device__ void copy(Itemset *it){
		theNumel = it->theNumel;
		for (int i=0;i < theNumel; i++)
			theItemset[i] = it->theItemset[i];
		support = it->support;
	}

	__device__ int subsequence(Itemset& ar)
	{
		int i, j;

		if (theNumel > ar.theNumel) return 0;
		int start = 0;
		for(i=0; i < theNumel; i++){
			for(j=start; j < ar.theNumel; j++)
				if (theItemset[i] == ar.theItemset[j]){
					start = j+1;
					break;
				}
			if (j >= ar.theNumel) return 0;
		}
		return 1;
	}

	__device__ int subsequence(int *indx, int len)
	{
		if (theNumel > len) return 0;
		for (int i=0; i < theNumel; i++)
		{
			if(theItemset[i]>1000)
				printf("\noopsy subsequence itemset %d ",theItemset[i]);
			if (indx[theItemset[i]] != 1) return 0;
		}
		return 1;
	}

	__device__ int subsequence(Itemset& ar, int *pos_arr)
	{
		int i, j;

		if (theNumel > ar.theNumel) return 0;
		int start = 0;
		for(i=0; i < theNumel; i++){
			for(j=start; j < ar.theNumel; j++)
				if (theItemset[i] == ar.theItemset[j]){
					pos_arr[i] = j;
					start = j+1;
					break;
				}
			if (j >= ar.theNumel) return 0;
		}
		return 1;
	}


	__device__ __host__ int compare(Itemset& ar2)
	{
		int len = (theNumel > ar2.theNumel) ? ar2.theNumel:theNumel;
		for(int i=0; i < len; i++){
			if (theItemset[i] > ar2.theItemset[i]) return 1;
			else if (theItemset[i] < ar2.theItemset[i]) return -1;
		}
		if (theNumel > ar2.theNumel) return 1;
		else if (theNumel < ar2.theNumel) return -1;
		else return 0;
	}

	//len must be less than length of both Itemsets
	__device__ __host__  int compare(Itemset& ar2, int len)
	{
		for(int i=0; i < len; i++){
			if (theItemset[i] > ar2.theItemset[i]) return 1;
			else if (theItemset[i] < ar2.theItemset[i]) return -1;
		}
		return 0;
	}


	//ostream& operator << (ostream& outputStream, Itemset& arr){
	//  outputStream << "ITEM: ";
	//  for (int i=0; i < arr.theNumel; i++)
	//    outputStream << i << ":" << arr.theItemset[i] << " ";
	//  outputStream << " N:" << arr.theNumel << "(" << arr.theSize << ")"
	//  << " S:" << arr.support
	//  << " T:" << arr.Tid << "\n";
	//  return outputStream;
	//}

	__device__ __host__   inline void add_item(int pos, int val)
	{
		theItemset[pos] = val;
	}

	__device__ __host__ inline void set_numitems(int val)
	{
		theNumel = val;
	}

	__device__ inline int maxsize(){
		return theSize;
	}
	__device__ __host__ inline int sup(){
		return support;
	}
	__device__ inline void incr_sup()
	{
		support++;
	}
	__device__ __host__ inline void set_sup(int val)
	{
		support = val;
	}
	__device__ inline int tid(){
		return Tid;
	}
	__device__ inline void set_tid(int val)
	{
		Tid = val;
	}

	__device__ inline int *get_itemset()
	{
		return theItemset;
	}
public:
	int *theItemset;
	unsigned int theNumel;
	unsigned int theSize;
	int support;
	int Tid;
};

#endif //__ITEMSET_H







