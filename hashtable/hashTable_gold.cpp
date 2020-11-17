#include <math.h>
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
int computeGold( int* gpuData, const int len);



////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! @param gpuData		resulting heap after gpu kernel has ran
//! @param len			blocks * threads
////////////////////////////////////////////////////////////////////////////////
int
computeGold(int* gpuData, const int len) 
{
	// basic idea: compute expected result, compare with gpuData
	
    return true;
}

