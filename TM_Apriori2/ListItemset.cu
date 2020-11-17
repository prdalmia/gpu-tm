#ifndef LISTITEMSET_H
#define LISTITEMSET_H
#include <stddef.h>
#include <stdlib.h>
#include "Itemset.cu"

class ListElement {

public:

	__device__ __host__ ListElement(Itemset *itemPtr)
	{
		Item = itemPtr;
		Next = NULL;	// assume we'll put it at the end of the list
	}


	~ListElement()
	{
		Item = NULL;
		Next = NULL;
	}


//
//	void *operator new(size_t size)
//	{
//		ListElement * list_element = (ListElement*)malloc(size);
//		return (void *)list_element;
//	}


//	void operator delete(void * ptr_listelement)
//	{
//		ListElement * ptr = (ListElement*) ptr_listelement;
//		free(ptr);
//	}


	__device__  __host__ void set_next(ListElement *n)
	{
		Next = n;
	}


	__device__ __host__ Itemset* item()
		{
		return Item;
		}


	__device__ __host__   ListElement * next()
		{
		return Next;
		}

	__device__ __host__ inline void set_item(Itemset *it)
	{
		Item = it;
	}


public:
	ListElement *Next;		// next element on list,
	// NULL if this is the last
	Itemset *Item;
};

class ListItemset {

public:

	void * operator new(size_t size)
	{
		ListItemset * list_itemset = (ListItemset*)malloc(size);
		return (void *)list_itemset;
	}


	void operator delete(void * ptr_itemset)
	{
		ListItemset * ptr = (ListItemset*) ptr_itemset;
		free(ptr);
	}


	__device__ __host__ ListItemset()
	{
		First = NULL;
		Last = NULL;
		numitem =0;
	}


	__device__ __host__  void append(ListElement *element)
	{

		if (First == NULL) {		// list is empty
			First = element;
			First->set_next(NULL);
			Last = element;
			Last->set_next(NULL);
		} else {
			// else put it after Last
			Last->set_next(element);
			Last = element;
			Last->set_next(NULL);
		}
		numitem++;
	}


	__device__ __host__ int numitems()
	{
		return numitem;
	}


	__device__ __host__  ListElement* first()
		{
		return First;
		}



	__device__ void sortedInsert( ListElement *element, Itemset *item)
	{
		//cout << item << flush;
		ListElement *ptr;		// keep track

		numitem++;
		if (First == NULL) {	// if list is empty, put
			First = element;
			Last = element;
		}
		else if (item->compare(*First->item()) <= 0) {
			// item goes on front of list
			element->set_next(First);
			First = element;
		}
		else {		// look for First elt in list bigger than item
			for (ptr = First; ptr->next() != NULL; ptr = ptr->next()) {
				if (item->compare(*ptr->next()->item())<=0) {
					element->set_next(ptr->next());
					ptr->set_next(element);
					return;
				}
			}
			Last->set_next(element);		// item goes at end of list
			Last = element;
		}
	}


/*
	ListElement * sortedInsert( Itemset *item, ListElement *cpos)
		{
		//cout << item << flush;
		ListElement *element = new ListElement(item);
		ListElement *ptr;		// keep track

		numitem++;
		if (cpos == NULL) {
			First = element;
			Last = element;
			return First;
		}
		else if (cpos == First && item->compare(*First->item()) <= 0) {
			// item goes on front of list
			element->set_next(First);
			First = element;
			return First;
		}
		else {		// look for First elt in list bigger than item
			for (ptr = cpos; ptr->next() != NULL; ptr = ptr->next()) {
				if (item->compare(*ptr->next()->item())<=0) {
					element->set_next(ptr->next());
					ptr->set_next(element);
					return ptr->next();
				}
			}
			Last->set_next(element);		// item goes at end of list
			Last = element;
			return Last;
		}
		}
*/


	//----------------------------------------------------------------------
	// Remove
	//      Remove the First "item" from the front of a list.
	//
	//----------------------------------------------------------------------

	__device__ ListElement *remove()
		{
		ListElement *element = First;

		if (First == NULL)
			return NULL;

		if (First == Last) {	// list had one item, now has none
			First = NULL;
			Last = NULL;
		} else {
			First = element->next();
		}
		//delete element;
		numitem--;
		return element;
		}


	__device__ __host__ ListElement *node(int pos){
		ListElement *head = First;
		for (int i=0; i < pos && head; head = head->next(),i++);
		return head;
	}


	__device__ void  clearlist()
	{
		while (remove() != NULL);
		numitem = 0;
		First = Last = NULL;
	}


	__device__ inline ListElement *last()
	{
		return Last;
	}

public:
	ListElement *First;  	// Head of the list, NULL if list is empty
	ListElement *Last;		// Last element of list
	int numitem;
};


#endif // LISTITEMSET_H

