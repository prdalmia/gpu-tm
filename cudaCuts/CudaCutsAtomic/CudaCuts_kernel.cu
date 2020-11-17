
/***********************************************************************************************
 * * Implementing Graph Cuts on CUDA using algorithm given in CVGPU '08                       **
 * * paper "CUDA Cuts: Fast Graph Cuts on GPUs"                                               **
 * *                                                                                          **
 * * Copyright (c) 2008 International Institute of Information Technology.                    **
 * * All rights reserved.                                                                     **
 * *                                                                                          **
 * * Permission to use, copy, modify and distribute this software and its documentation for   **
 * * educational purpose is hereby granted without fee, provided that the above copyright     **
 * * notice and this permission notice appear in all copies of this software and that you do  **
 * * not sell the software.                                                                   **
 * *                                                                                          **
 * * THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR    **
 * * OTHERWISE.                                                                               **
 * *                                                                                          **
 * * Created By Vibhav Vineet.                                                                **
 * ********************************************************************************************/

#ifndef _PUSHRELABEL_KERNEL_CU_
#define _PUSHRELABEL_KERNEL_CU_

#include "CudaCuts.h"

__global__ void
setBlockCount(int * g_count_blocks, int count)
{
  /*
  // only 1 thread in this kernel
  // all of the arrays accessed in this kernel are either in the special region
  __denovo_setAcquireRegion(SPECIAL_REGION);
  __denovo_addAcquireRegion(RELAX_ATOM_REGION);
  __denovo_addAcquireRegion(READ_ONLY_REGION);
*/
  // unpaired atomic
  atomicExch(g_count_blocks, count);
/*
  // all of the arrays accessed in this kernel are either in the special region
  __denovo_gpuEpilogue(SPECIAL_REGION);
  __denovo_gpuEpilogue(RELAX_ATOM_REGION);
  __denovo_gpuEpilogue(READ_ONLY_REGION);
  */
}

/************************************************
 * Push operation is performed                 ** 
 * *********************************************/
__global__ void
kernel_push1_atomic( int *g_left_weight, int *g_right_weight,
                     int *g_down_weight, int *g_up_weight, int *g_sink_weight,
                     int *g_push_reser, int *g_pull_left, int *g_pull_right,
                     int *g_pull_down, int *g_pull_up, int *g_relabel_mask,
                     int *g_graph_height, int *g_height_write, int graph_size,
                     int width, int rows, int graph_size1, int width1,
                     int rows1)
{
  /*
    if (threadIdx.x == 0) {
      // all of the arrays accessed in this kernel are either in the special region or
      // the relaxed atomic region
      __denovo_setAcquireRegion(SPECIAL_REGION);
      __denovo_addAcquireRegion(RELAX_ATOM_REGION);
      __denovo_addAcquireRegion(READ_ONLY_REGION);
    }
    __syncthreads();
*/
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
    int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
    int thid = __umul24( y, width1 ) + x;
  /*
    // for use in inlined assembly
    int * pushTidMinus1Addr = &(g_push_reser[thid-1]);
    int * pushTidMinusWidth1Addr = &(g_push_reser[thid-width1]);
    int * pushTidAddr = &(g_push_reser[thid]);
    int * pushTidPlusWidth1Addr = &(g_push_reser[thid+width1]);
    int * pushTidPlus1Addr = &(g_push_reser[thid+1]);
    int * leftWeightTidAddr = &(g_left_weight[thid]);
    int * leftWeightTidPlus1Addr = &(g_left_weight[thid+1]);
    int * rightWeightTidMinus1Addr = &(g_right_weight[thid-1]);
    int * rightWeightTidAddr = &(g_right_weight[thid]);
    int * upWeightTidAddr = &(g_up_weight[thid]);
    int * upWeightTidPlusWidth1Addr = &(g_up_weight[thid+width1]);
    int * downWeightTidMinusWidth1Addr = &(g_down_weight[thid-width1]);
    int * downWeightTidAddr = &(g_down_weight[thid]);
*/
    __shared__ int height_fn[356];

    int temp_mult = __umul24(y1+1, 34 ) + x1 + 1, temp_mult1 = __umul24(y1,32) + x1;

    height_fn[temp_mult] = g_graph_height[thid];

    // handles corner cases of loading scratchpad
    (threadIdx.x == 31 && x < width1 - 1 ) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
    (threadIdx.x == 0 && x > 0 ) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
    (threadIdx.y == 7 && y < rows1 - 1 ) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
    (threadIdx.y == 0 && y > 0 ) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;
    __syncthreads();

    int flow_push = 0, min_flow_pushed = 0, neg_min_flow_pushed = 0;
    //flow_push = g_push_reser[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);

    if( thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width-1 && x > 0 && y < rows-1 && y > 0 )
    {
      int temp_weight = 0;

      temp_weight = g_sink_weight[thid];
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        g_sink_weight[thid] = temp_weight;
        atomicSub(&g_push_reser[thid], min_flow_pushed);
      }

      //flow_push = g_push_reser[thid];
      //min_flow_pushed = flow_push;
      //temp_weight = g_left_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_left_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 1] + 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_left_weight[thid], min_flow_pushed);
        atomicAdd(&g_right_weight[thid-1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid-1], min_flow_pushed);
      
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 a0;\n\t"    // temp reg a0 (min_flow_pushed)
                     ".reg .s32 a1;\n\t"    // temp reg a1 (negative min_flow_pushed)
                     ".reg .s32 a2;\n\t"    // temp reg a2 (atomicSub(leftWeightTidAddr) result)
                     ".reg .s32 a3;\n\t"    // temp reg a3 (atomicAdd(rightWeightMinus1Addr) result)
                     ".reg .s32 a4;\n\t"    // temp reg a4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 a5;\n\t"    // temp reg a5 (atomicAdd(pushTidMinus1Addr) result)
                     // PTX Instructions
                     "mov.s32 a0, %0;\n\t"
                     "mov.s32 a1, %1;\n\t"
                     "atom.add.s32 a2, [%2], a1;\n\t" // atomicSub for leftWeightTidAddr
                     "atom.add.s32 a3, [%3], a0;\n\t" // atomicAdd for rightWeightTidMinus1Addr
                     "atom.add.s32 a4, [%4], a1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 a5, [%5], a0;"     // atomicAdd for pushTidMinus1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(leftWeightTidAddr), "l"(rightWeightTidMinus1Addr),
                        "l"(pushTidAddr), "l"(pushTidMinus1Addr)
                     );
					*/					
					}else {
        atomicSub(&g_pull_left[thid-1], 1);
      }
      //flow_push = g_push_reser[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      min_flow_pushed = flow_push;
      //temp_weight = g_up_weight[thid];
      temp_weight = atomicAdd(&g_up_weight[thid], 0);

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 34] + 1)
      {
        (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;
 
       atomicSub(&g_up_weight[thid], min_flow_pushed);
        atomicAdd(&g_down_weight[thid-width1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid-width1], min_flow_pushed);
      }
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
       /*
        asm volatile(// Temp Registers
                     ".reg .s32 b0;\n\t"    // temp reg b0 (min_flow_pushed)
                      ".reg .s32 b1;\n\t"    // temp reg b1 (negative min_flow_pushed)
                     ".reg .s32 b2;\n\t"    // temp reg b2 (atomicSub(upWeightTidAddr) result)
                     ".reg .s32 b3;\n\t"    // temp reg b3 (atomicAdd(downWeightMinusWidth1Addr) result)
                     ".reg .s32 b4;\n\t"    // temp reg b4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 b5;\n\t"    // temp reg b5 (atomicAdd(pushTidMinusWidth1Addr) result)
                     // PTX Instructions
                     "mov.s32 b0, %0;\n\t"
                     "mov.s32 b1, %1;\n\t"
                     "atom.add.s32 b2, [%2], b1;\n\t" // atomicSub for upWeightTidAddr
                     "atom.add.s32 b3, [%3], b0;\n\t" // atomicAdd for downWeightTidMinusWidth1Addr
                     "atom.add.s32 b4, [%4], b1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 b5, [%5], b0;"     // atomicAdd for pushTidMinusWidth1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(upWeightTidAddr), "l"(downWeightTidMinusWidth1Addr),
                        "l"(pushTidAddr), "l"(pushTidMinusWidth1Addr)
                     );
      */
                    } else {
        atomicSub(&g_pull_up[thid - width1], 1);
      }

       //flow_push = g_push_reser[thid];
      //temp_weight = g_right_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_right_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 1] + 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_right_weight[thid], min_flow_pushed);
        atomicAdd(&g_left_weight[thid+1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid+1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 c0;\n\t"    // temp reg c0 (min_flow_pushed)
                     ".reg .s32 c1;\n\t"    // temp reg c1 (negative min_flow_pushed)
                     ".reg .s32 c2;\n\t"    // temp reg c2 (atomicSub(rightWeightTidAddr) result)
                     ".reg .s32 c3;\n\t"    // temp reg c3 (atomicAdd(leftWeightTidPlus1Addr) result)
                     ".reg .s32 c4;\n\t"    // temp reg c4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 c5;\n\t"    // temp reg c5 (atomicAdd(pushTidPlus1Addr) result)
                     // PTX Instructions
                     "mov.s32 c0, %0;\n\t"
                     "mov.s32 c1, %1;\n\t"
                     "atom.add.s32 c2, [%1], c1;\n\t" // atomicSub for leftWeightTidAddr
                     "atom.add.s32 c3, [%2], c0;\n\t" // atomicAdd for rightWeightTidMinus1Addr
                     "atom.add.s32 c4, [%3], c1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 c5, [%4], c0;" // atomicAdd for pushTidMinus1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(rightWeightTidAddr), "l"(leftWeightTidPlus1Addr),
                        "l"(pushTidAddr), "l"(pushTidPlus1Addr)
                     );
                     */
      } else {
        atomicSub( &g_pull_right[thid + 1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_down_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_down_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 34] + 1 )
      {
        (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_down_weight[thid], min_flow_pushed);
        atomicAdd(&g_up_weight[thid+width1], min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid+width1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 d0;\n\t"    // temp reg d0 (min_flow_pushed)
                     ".reg .s32 d1;\n\t"    // temp reg d1 (negative min_flow_pushed)
                     ".reg .s32 d2;\n\t"    // temp reg d2 (atomicSub(downWeightTidAddr) result)
                     ".reg .s32 d3;\n\t"    // temp reg d3 (atomicAdd(upWeightTidPlusWidth1Addr) result)
                     ".reg .s32 d4;\n\t"    // temp reg d4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 d5;\n\t"    // temp reg d5 (atomicAdd(pushTidPlusWidth1Addr) result)
                     // PTX Instructions
                     "mov.s32 d0, %0;\n\t"
                     "mov.s32 d1, %1;\n\t"
                     "atom.add.s32 d2, [%1], d1;\n\t" // atomicSub for downWeightTidAddr
                     "atom.add.s32 d3, [%2], d0;\n\t" // atomicAdd for upWeightTidPlusWidth1Addr
                     "atom.add.s32 d4, [%3], d1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 d5, [%4], d0;" // atomicAdd for pushTidPlusWidth1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(downWeightTidAddr), "l"(upWeightTidPlusWidth1Addr),
                        "l"(pushTidAddr), "l"(pushTidPlusWidth1Addr)
                     );
      } 
      */
      else {
        atomicSub( &g_pull_down[thid+width1], 1);
      }
    }
/*
    if (threadIdx.x == 0) {
      // all of the arrays accessed in this kernel are either in the special region or
      // the relaxed atomic region
      __denovo_gpuEpilogue(SPECIAL_REGION);
      __denovo_gpuEpilogue(RELAX_ATOM_REGION);
      __denovo_gpuEpilogue(READ_ONLY_REGION);
    }
   */ 
}

__global__ void
kernel_relabel_atomic( int *g_left_weight, int *g_right_weight,
                       int *g_down_weight, int *g_up_weight, int *g_sink_weight,
                       int *g_push_reser, int *g_pull_left, int *g_pull_right,
                       int *g_pull_down, int *g_pull_up, int *g_relabel_mask,
                       int *g_graph_height, int *g_height_write, int graph_size,
                       int width, int rows, int graph_size1, int width1,
                       int rows1 )
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int x1 = threadIdx.x;
  int y1 = threadIdx.y;
  int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
  int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
  int thid = __umul24( y, width1 ) + x;

  __shared__ int height_fn[356];

  int temp_mult = __umul24(y1+1, 34 ) + x1 + 1, temp_mult1 = __umul24(y1,32) + x1;

  height_fn[temp_mult] = g_graph_height[thid];

  (threadIdx.x == 31 && x < width1 - 1 ) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
  (threadIdx.x == 0 && x > 0 ) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
  (threadIdx.y == 7 && y < rows1 - 1 ) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
  (threadIdx.y == 0 && y > 0 ) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;

  __syncthreads();

  //int min_flow_pushed = g_left_weight[thid];
  //int flow_push = g_push_reser[thid];
  int min_flow_pushed = atomicAdd(&g_left_weight[thid], 0);
  int flow_push = atomicAdd(&g_push_reser[thid], 0);

  if (flow_push <= 0 ||
      (/*g_left_weight[thid]*/atomicAdd(&g_left_weight[thid], 0) == 0 &&
       /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) == 0 &&
       /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) == 0 &&
       /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) == 0 &&
       g_sink_weight[thid] == 0))
    g_relabel_mask[thid] = 2;
  else
  {
    ( flow_push > 0 && ( ( (height_fn[temp_mult] == height_fn[temp_mult-1] + 1 ) && /*g_left_weight[thid]*/atomicAdd(&g_left_weight[thid], 0) > 0  ) ||( (height_fn[temp_mult] == height_fn[temp_mult+1]+1 ) && /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0) || ( ( height_fn[temp_mult] == height_fn[temp_mult+34]+1 ) && /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0) || ( (height_fn[temp_mult] == height_fn[temp_mult-34]+1 ) && /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 ) || ( height_fn[temp_mult] == 1 && g_sink_weight[thid] > 0 )  ) ) ? g_relabel_mask[thid] = 1 : g_relabel_mask[thid] = 0;
  }

  __syncthreads();

  if(thid < graph_size1 && x < width - 1  && x > 0 && y < rows - 1  && y > 0  )
  {
    if(g_sink_weight[thid] > 0)
    {
      g_height_write[thid] = 1;
    }
    else
    {
      int min_height = graph_size;
      (min_flow_pushed > 0 && min_height > height_fn[temp_mult - 1] ) ? min_height = height_fn[temp_mult - 1] : 0;
      (/*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0 && min_height > height_fn[temp_mult + 1]) ? min_height = height_fn[temp_mult + 1] : 0;
      (/*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0 && min_height > height_fn[temp_mult + 34] ) ? min_height = height_fn[temp_mult + 34] : 0;
      (/*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 && min_height > height_fn[temp_mult - 34] ) ? min_height = height_fn[temp_mult - 34] : 0;
      g_height_write[thid] = min_height + 1;
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
*/
}

__global__ void
kernel_relabel_stochastic( int *g_left_weight, int *g_right_weight,
                           int *g_down_weight, int *g_up_weight,
                           int *g_sink_weight, int *g_push_reser,
                           int *g_pull_left, int *g_pull_right,
                           int *g_pull_down, int *g_pull_up,
                           int *g_relabel_mask, int *g_graph_height,
                           int *g_height_write, int graph_size, int width,
                           int rows, int graph_size1, int width1, int rows1,
                           int *d_stochastic, int g_block_num )
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  if(d_stochastic[blockIdx.y * g_block_num + blockIdx.x] == 1 )
  {
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
    int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
    int thid = __umul24( y, width1 ) + x;

    __shared__ int height_fn[356];

    int temp_mult = __umul24(y1+1, 34 ) + x1 + 1, temp_mult1 = __umul24(y1,32) + x1;

    height_fn[temp_mult] = g_graph_height[thid];

    (threadIdx.x == 31 && x < width1 - 1 ) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
    (threadIdx.x == 0 && x > 0 ) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
    (threadIdx.y == 7 && y < rows1 - 1 ) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
    (threadIdx.y == 0 && y > 0 ) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;

    __syncthreads();

    //int min_flow_pushed = g_left_weight[thid];
    int min_flow_pushed = atomicAdd(&g_left_weight[thid], 0);
    //int flow_push = g_push_reser[thid];
    int flow_push = atomicAdd(&g_push_reser[thid], 0);

    if(flow_push <= 0 ||
       (/*g_left_weight[thid]*/atomicAdd(&g_left_weight[thid], 0) == 0 &&
        /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) == 0 &&
        /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) == 0 &&
        /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) == 0 &&
        g_sink_weight[thid] == 0))
      g_relabel_mask[thid] = 2;
    else
    {
      ( flow_push > 0 && ( ( (height_fn[temp_mult] == height_fn[temp_mult-1] + 1 ) && /*g_left_weight[thid]*/atomicAdd(&g_left_weight[thid], 0) > 0  ) ||( (height_fn[temp_mult] == height_fn[temp_mult+1]+1 ) && /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0) || ( ( height_fn[temp_mult] == height_fn[temp_mult+34]+1 ) && /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0) || ( (height_fn[temp_mult] == height_fn[temp_mult-34]+1 ) && /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 ) || ( height_fn[temp_mult] == 1 && g_sink_weight[thid] > 0 )  ) ) ? g_relabel_mask[thid] = 1 : g_relabel_mask[thid] = 0;
    }

    __syncthreads();

    if(thid < graph_size1 && x < width - 1  && x > 0 && y < rows - 1  && y > 0  )
    {
      if(g_sink_weight[thid] > 0)
      {
        g_height_write[thid] = 1;
      }
      else
      {
        int min_height = graph_size;
        (min_flow_pushed > 0 && min_height > height_fn[temp_mult - 1] ) ? min_height = height_fn[temp_mult - 1] : 0;
        (/*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0 && min_height > height_fn[temp_mult + 1]) ? min_height = height_fn[temp_mult + 1] : 0;
        (/*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0 && min_height > height_fn[temp_mult + 34] ) ? min_height = height_fn[temp_mult + 34] : 0;
        (/*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 && min_height > height_fn[temp_mult - 34] ) ? min_height = height_fn[temp_mult - 34] : 0;
        g_height_write[thid] = min_height + 1;
      }
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

__global__ void
kernel_push2_atomic( int * g_left_weight, int * g_right_weight,
                     int * g_down_weight, int * g_up_weight,
                     int * g_sink_weight, int * g_push_reser, int * g_pull_left,
                     int * g_pull_right, int * g_pull_down, int * g_pull_up,
                     int * g_relabel_mask, int * g_graph_height,
                     int * g_height_write, int graph_size, int width, int rows,
                     int graph_size1, int width1, int rows1)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int x1 = threadIdx.x;
  int y1 = threadIdx.y;
  int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
  int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
  int thid = __umul24( y, width1 ) + x;
  // for use in inlined assembly
  /*
  int * pushTidMinus1Addr = &(g_push_reser[thid-1]);
  int * pushTidMinusWidth1Addr = &(g_push_reser[thid-width1]);
  int * pushTidAddr = &(g_push_reser[thid]);
  int * pushTidPlusWidth1Addr = &(g_push_reser[thid+width1]);
  int * pushTidPlus1Addr = &(g_push_reser[thid+1]);
  int * leftWeightTidAddr = &(g_left_weight[thid]);
  int * leftWeightTidPlus1Addr = &(g_left_weight[thid+1]);
  int * rightWeightTidMinus1Addr = &(g_right_weight[thid-1]);
  int * rightWeightTidAddr = &(g_right_weight[thid]);
  int * upWeightTidAddr = &(g_up_weight[thid]);
  int * upWeightTidPlusWidth1Addr = &(g_up_weight[thid+width1]);
  int * downWeightTidMinusWidth1Addr = &(g_down_weight[thid-width1]);
  int * downWeightTidAddr = &(g_down_weight[thid]);
*/
  __shared__ int height_fn[356];

  int temp_mult = __umul24(y1+1, 34 ) + x1 + 1, temp_mult1 = __umul24(y1,32) + x1;

  height_fn[temp_mult] = g_graph_height[thid];

  (threadIdx.x == 31 && x < width1 - 1 ) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
  (threadIdx.x == 0 && x > 0 ) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
  (threadIdx.y == 7 && y < rows1 - 1 ) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
  (threadIdx.y == 0 && y > 0 ) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;

  __syncthreads();

  int flow_push = 0, min_flow_pushed = 0, neg_min_flow_pushed = 0;
  //flow_push = g_push_reser[thid];
  flow_push = atomicAdd(&g_push_reser[thid], 0);

  if( thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width-1 && x > 0 && y < rows-1 && y > 0 )
  {
    int temp_weight = 0;

    temp_weight = g_sink_weight[thid];
    min_flow_pushed = flow_push;

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1 )
    {
      (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      g_sink_weight[thid] = temp_weight;
      atomicSub(&g_push_reser[thid], min_flow_pushed);
    }

    //flow_push = g_push_reser[thid];
    //temp_weight = g_left_weight[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);
    min_flow_pushed = flow_push;
    temp_weight = atomicAdd(&g_left_weight[thid], 0);

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 1] + 1 )
    {
      (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      neg_min_flow_pushed = 0 - min_flow_pushed;

      
      atomicSub(&g_left_weight[thid], min_flow_pushed);
      atomicAdd(&g_right_weight[thid-1],min_flow_pushed);
      atomicSub(&g_push_reser[thid], min_flow_pushed);
      atomicAdd(&g_push_reser[thid-1], min_flow_pushed);
      
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      /*
      asm volatile(// Temp Registers
                   ".reg .s32 e0;\n\t"    // temp reg e0 (min_flow_pushed)
                   ".reg .s32 e1;\n\t"    // temp reg e1 (negative min_flow_pushed)
                   ".reg .s32 e2;\n\t"    // temp reg e2 (atomicSub(leftWeightTidAddr) result)
                   ".reg .s32 e3;\n\t"    // temp reg e3 (atomicAdd(rightWeightTidMinus1Addr) result)
                   ".reg .s32 e4;\n\t"    // temp reg e4 (atomicSub(pushTidAddr) result)
                   ".reg .s32 e5;\n\t"    // temp reg e5 (atomicAdd(pushTidMinus1Addr) result)
                   // PTX Instructions
                   "mov.s32 e0, %0;\n\t"
                   "mov.s32 e1, %1;\n\t"
                   "atom.add.s32 e2, [%1], e1;\n\t" // atomicSub for leftWeightTidAddr
                   "atom.add.s32 e3, [%2], e0;\n\t" // atomicAdd for rightWeightTidMinus1Addr
                   "atom.add.s32 e4, [%3], e1;\n\t" // atomicSub for pushTidAddr
                   "atom.add.s32 e5, [%4], e0;"     // atomicAdd for pushTidMinus1Addr
                   // no outputs
                   // inputs
                   :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                      "l"(leftWeightTidAddr), "l"(rightWeightTidMinus1Addr),
                      "l"(pushTidAddr), "l"(pushTidMinus1Addr)
                   );
    */
                  } else {
      atomicSub(&g_pull_left[thid-1], 1);
    }

    //flow_push = g_push_reser[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);
    min_flow_pushed = flow_push;
    //temp_weight = g_up_weight[thid];
    temp_weight = atomicAdd(&g_up_weight[thid], 0);

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 34] + 1)
    {
      (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      neg_min_flow_pushed = 0 - min_flow_pushed;

      
      atomicSub(&g_up_weight[thid], min_flow_pushed);
      atomicAdd(&g_down_weight[thid-width1],min_flow_pushed);
      atomicSub(&g_push_reser[thid], min_flow_pushed);
      atomicAdd(&g_push_reser[thid-width1], min_flow_pushed);
      
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      /*
      asm volatile(// Temp Registers
                   ".reg .s32 g0;\n\t"    // temp reg g0 (min_flow_pushed)
                   ".reg .s32 g1;\n\t"    // temp reg g1 (negative min_flow_pushed)
                   ".reg .s32 g2;\n\t"    // temp reg g2 (atomicSub(upWeightTidAddr) result)
                   ".reg .s32 g3;\n\t"    // temp reg g3 (atomicAdd(downWeightTidMinusWidth1Addr) result)
                   ".reg .s32 g4;\n\t"    // temp reg g4 (atomicSub(pushTidAddr) result)
                   ".reg .s32 g5;\n\t"    // temp reg g5 (atomicAdd(pushTidMinusWidth1Addr) result)
                   // PTX Instructions
                   "mov.s32 g0, %0;\n\t"
                   "mov.s32 g1, %1;\n\t"
                   "atom.add.s32 g2, [%1], g1;\n\t" // atomicSub for upWeightTidAddr
                   "atom.add.s32 g3, [%2], g0;\n\t" // atomicAdd for downWeightTidMinusWidth1Addr
                   "atom.add.s32 g4, [%3], g1;\n\t" // atomicSub for pushTidAddr
                   "atom.add.s32 g5, [%4], g0;"     // atomicAdd for pushTidMinusWidth1Addr
                   // no outputs
                   // inputs
                   :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                      "l"(upWeightTidAddr), "l"(downWeightTidMinusWidth1Addr),
                      "l"(pushTidAddr), "l"(pushTidMinusWidth1Addr)
                   );
    */
                  } else {
      atomicSub(&g_pull_up[thid - width1], 1);
    }

    //flow_push = g_push_reser[thid];
    //temp_weight = g_right_weight[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);
    temp_weight = atomicAdd(&g_right_weight[thid], 0);
    min_flow_pushed = flow_push;

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 1] + 1 )
    {
      (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      neg_min_flow_pushed = 0 - min_flow_pushed;

      
      atomicSub(&g_right_weight[thid], min_flow_pushed);
      atomicAdd(&g_left_weight[thid+1],min_flow_pushed);
      atomicSub(&g_push_reser[thid], min_flow_pushed);
      atomicAdd(&g_push_reser[thid+1], min_flow_pushed);
      
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      /*
      asm volatile(// Temp Registers
                   ".reg .s32 h0;\n\t"    // temp reg h0 (min_flow_pushed)
                   ".reg .s32 h1;\n\t"    // temp reg h1 (negative min_flow_pushed)
                   ".reg .s32 h2;\n\t"    // temp reg h2 (atomicSub(rightWeightTidAddr) result)
                   ".reg .s32 h3;\n\t"    // temp reg h3 (atomicAdd(leftWeightTidPlus1Addr) result)
                   ".reg .s32 h4;\n\t"    // temp reg h4 (atomicSub(pushTidAddr) result)
                   ".reg .s32 h5;\n\t"    // temp reg h5 (atomicAdd(pushTidPlus1Addr) result)
                   // PTX Instructions
                   "mov.s32 h0, %0;\n\t"
                   "mov.s32 h1, %1;\n\t"
                   "atom.add.s32 h2, [%1], h1;\n\t" // atomicSub for rightWeightTidAddr
                   "atom.add.s32 h3, [%2], h0;\n\t" // atomicAdd for leftWeightTidPlus1Addr
                   "atom.add.s32 h4, [%3], h1;\n\t" // atomicSub for pushTidAddr
                   "atom.add.s32 h5, [%4], h0;"     // atomicAdd for pushTidPlus1Addr
                   // no outputs
                   // inputs
                   :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                      "l"(rightWeightTidAddr), "l"(leftWeightTidPlus1Addr),
                      "l"(pushTidAddr), "l"(pushTidPlus1Addr)
                   );
    */
    } 
    else {
      atomicSub( &g_pull_right[thid + 1], 1);
    }

    //flow_push = g_push_reser[thid];
    //temp_weight = g_down_weight[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);
    temp_weight = atomicAdd(&g_down_weight[thid], 0);
    min_flow_pushed = flow_push;

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 34] + 1 )
    {
      (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      neg_min_flow_pushed = 0 - min_flow_pushed;

      
      atomicSub(&g_down_weight[thid], min_flow_pushed);
      atomicAdd(&g_up_weight[thid+width1], min_flow_pushed);
      atomicSub(&g_push_reser[thid], min_flow_pushed);
      atomicAdd(&g_push_reser[thid+width1], min_flow_pushed);
      
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      /*
      asm volatile(// Temp Registers
                   ".reg .s32 i0;\n\t"    // temp reg i0 (min_flow_pushed)
                   ".reg .s32 i1;\n\t"    // temp reg i1 (negative min_flow_pushed)
                   ".reg .s32 i2;\n\t"    // temp reg i2 (atomicSub(downWeightTidAddr) result)
                   ".reg .s32 i3;\n\t"    // temp reg i3 (atomicAdd(upWeightTidPlusWidth1Addr) result)
                   ".reg .s32 i4;\n\t"    // temp reg i4 (atomicSub(pushTidAddr) result)
                   ".reg .s32 i5;\n\t"    // temp reg i5 (atomicAdd(pushTidPlusWidth1Addr) result)
                   // PTX Instructions
                   "mov.s32 i0, %0;\n\t"
                   "mov.s32 i1, %1;\n\t"
                   "atom.add.s32 i2, [%1], i1;\n\t" // atomicSub for downWeightTidAddr
                   "atom.add.s32 i3, [%2], i0;\n\t" // atomicAdd for upWeightTidPlusWidth1Addr
                   "atom.add.s32 i4, [%3], i1;\n\t" // atomicSub for pushTidAddr
                   "atom.add.s32 i5, [%4], i0;"     // atomicAdd for pushTidPlusWidth1Addr
                   // no outputs
                   // inputs
                   :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                      "l"(downWeightTidAddr), "l"(upWeightTidPlusWidth1Addr),
                      "l"(pushTidAddr), "l"(pushTidPlusWidth1Addr)
                   );
    */
  }
    else {
      atomicSub( &g_pull_down[thid+width1], 1);
    }
  }
  
  __syncthreads();
  //min_flow_pushed = g_left_weight[thid];
  //flow_push = g_push_reser[thid];
  min_flow_pushed = atomicAdd(&g_left_weight[thid], 0);
  flow_push = atomicAdd(&g_push_reser[thid], 0);

  if (flow_push <= 0 ||
      (/*g_left_weight[thid]*/atomicAdd(&g_left_weight[thid], 0) == 0 &&
       /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) == 0 &&
       /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) == 0 &&
       /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) == 0 &&
       g_sink_weight[thid] == 0))
  {
    g_relabel_mask[thid] = 2;
  }
  else
  {
    ( flow_push > 0 && ( ( (height_fn[temp_mult] == height_fn[temp_mult-1] + 1 ) && /*g_left_weight[thid]*/atomicAdd(&g_left_weight[thid], 0) > 0  ) ||( (height_fn[temp_mult] == height_fn[temp_mult+1]+1 ) && /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0) || ( ( height_fn[temp_mult] == height_fn[temp_mult+34]+1 ) && /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0) || ( (height_fn[temp_mult] == height_fn[temp_mult-34]+1 ) && /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 ) || ( height_fn[temp_mult] == 1 && g_sink_weight[thid] > 0 )  ) ) ? g_relabel_mask[thid] = 1 : g_relabel_mask[thid] = 0;
  }

  __syncthreads();

  if( thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width-1 && x > 0 && y < rows-1 && y > 0 )
  {
    int temp_weight = 0;

    temp_weight = g_sink_weight[thid];
    min_flow_pushed = flow_push;

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1 )
    {
      (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      g_sink_weight[thid] = temp_weight;
      atomicSub(&g_push_reser[thid], min_flow_pushed);
    }

    //flow_push = g_push_reser[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);

    min_flow_pushed = flow_push;

    //temp_weight = g_left_weight[thid];
    temp_weight = atomicAdd(&g_left_weight[thid], 0);

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 1] + 1 )
    {
      (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      neg_min_flow_pushed = 0 - min_flow_pushed;
      
      
      atomicSub(&g_left_weight[thid], min_flow_pushed);
      atomicAdd(&g_right_weight[thid-1],min_flow_pushed);
      atomicSub(&g_push_reser[thid], min_flow_pushed);
      atomicAdd(&g_push_reser[thid-1], min_flow_pushed);
      
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      /*
      asm volatile(// Temp Registers
                   ".reg .s32 j0;\n\t"    // temp reg j0 (min_flow_pushed)
                   ".reg .s32 j1;\n\t"    // temp reg j1 (negative min_flow_pushed)
                   ".reg .s32 j2;\n\t"    // temp reg j2 (atomicSub(leftWeightTidAddr) result)
                   ".reg .s32 j3;\n\t"    // temp reg j3 (atomicAdd(rightWeightTidMinus1Addr) result)
                   ".reg .s32 j4;\n\t"    // temp reg j4 (atomicSub(pushTidAddr) result)
                   ".reg .s32 j5;\n\t"    // temp reg j5 (atomicAdd(pushTidMinus1Addr) result)
                   // PTX Instructions
                   "mov.s32 j0, %0;\n\t"
                   "mov.s32 j1, %1;\n\t"
                   "atom.add.s32 j2, [%1], j1;\n\t" // atomicSub for leftWeightTidAddr
                   "atom.add.s32 j3, [%2], j0;\n\t" // atomicAdd for rightWeightTidMinus1Addr
                   "atom.add.s32 j4, [%3], j1;\n\t" // atomicSub for pushTidAddr
                   "atom.add.s32 j5, [%4], j0;"     // atomicAdd for pushTidMinus1Addr
                   // no outputs
                   // inputs
                   :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                      "l"(leftWeightTidAddr), "l"(rightWeightTidMinus1Addr),
                      "l"(pushTidAddr), "l"(pushTidMinus1Addr)
                   );
    */
                  } 
    else {
      atomicSub(&g_pull_left[thid-1], 1);
    }

    //flow_push = g_push_reser[thid];
    //temp_weight = g_up_weight[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);
    temp_weight = atomicAdd(&g_up_weight[thid], 0);
    min_flow_pushed = flow_push;

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 34] + 1)
    {
      (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      neg_min_flow_pushed = 0 - min_flow_pushed;

      
      atomicSub(&g_up_weight[thid], min_flow_pushed);
      atomicAdd(&g_down_weight[thid-width1],min_flow_pushed);
      atomicSub(&g_push_reser[thid], min_flow_pushed);
      atomicAdd(&g_push_reser[thid-width1], min_flow_pushed);
      
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      /*
      asm volatile(// Temp Registers
                   ".reg .s32 k0;\n\t"    // temp reg k0 (min_flow_pushed)
                   ".reg .s32 k1;\n\t"    // temp reg k1 (negative min_flow_pushed)
                   ".reg .s32 k2;\n\t"    // temp reg k2 (atomicSub(upWeightTidAddr) result)
                   ".reg .s32 k3;\n\t"    // temp reg k3 (atomicAdd(downWeightTidMinusWidth1Addr) result)
                   ".reg .s32 k4;\n\t"    // temp reg k4 (atomicSub(pushTidAddr) result)
                   ".reg .s32 k5;\n\t"    // temp reg k5 (atomicAdd(pushTidMinusWidth1Addr) result)
                   // PTX Instructions
                   "mov.s32 k0, %0;\n\t"
                   "mov.s32 k1, %1;\n\t"
                   "atom.add.s32 k2, [%1], k1;\n\t" // atomicSub for upWeightTidAddr
                   "atom.add.s32 k3, [%2], k0;\n\t" // atomicAdd for downWeightTidMinusWidth1Addr
                   "atom.add.s32 k4, [%3], k1;\n\t" // atomicSub for pushTidAddr
                   "atom.add.s32 k5, [%4], k0;"     // atomicAdd for pushTidMinusWidth1Addr
                   // no outputs
                   // inputs
                   :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                      "l"(upWeightTidAddr), "l"(downWeightTidMinusWidth1Addr),
                      "l"(pushTidAddr), "l"(pushTidMinusWidth1Addr)
                   );
    */
                  } else {
      atomicSub(&g_pull_up[thid - width1], 1);
    }

    //flow_push = g_push_reser[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);
    min_flow_pushed = flow_push;
    //temp_weight = g_right_weight[thid];
    temp_weight = atomicAdd(&g_right_weight[thid], 0);

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 1] + 1 )
    {
      (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      neg_min_flow_pushed = 0 - min_flow_pushed;

      
      atomicSub(&g_right_weight[thid], min_flow_pushed);
      atomicAdd(&g_left_weight[thid+1], min_flow_pushed);
      atomicSub(&g_push_reser[thid], min_flow_pushed);
      atomicAdd(&g_push_reser[thid+1], min_flow_pushed);
      
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      /*
      asm volatile(// Temp Registers
                   ".reg .s32 l0;\n\t"    // temp reg l0 (min_flow_pushed)
                   ".reg .s32 l1;\n\t"    // temp reg l1 (negative min_flow_pushed)
                   ".reg .s32 l2;\n\t"    // temp reg l2 (atomicSub(rightWeightTidAddr) result)
                   ".reg .s32 l3;\n\t"    // temp reg l3 (atomicAdd(leftWeightTidPlus1Addr) result)
                   ".reg .s32 l4;\n\t"    // temp reg l4 (atomicSub(pushTidAddr) result)
                   ".reg .s32 l5;\n\t"    // temp reg l5 (atomicAdd(pushTidPlus1Addr) result)
                   // PTX Instructions
                   "mov.s32 l0, %0;\n\t"
                   "mov.s32 l1, %1;\n\t"
                   "atom.add.s32 l2, [%1], l1;\n\t" // atomicSub for rightWeightTidAddr
                   "atom.add.s32 l3, [%2], l0;\n\t" // atomicAdd for leftWeightTidPlus1Addr
                   "atom.add.s32 l4, [%3], l1;\n\t" // atomicSub for pushTidAddr
                   "atom.add.s32 l5, [%4], l0;"     // atomicAdd for pushTidPlus1Addr
                   // no outputs
                   // inputs
                   :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                      "l"(rightWeightTidAddr), "l"(leftWeightTidPlus1Addr),
                      "l"(pushTidAddr), "l"(pushTidPlus1Addr)
                   );
    */
                  } else {
      atomicSub( &g_pull_right[thid + 1], 1);
    }

    //flow_push = g_push_reser[thid];
    //temp_weight = g_down_weight[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);
    temp_weight = atomicAdd(&g_down_weight[thid], 0);
    min_flow_pushed = flow_push;

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 34] + 1 )
    {
      (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      neg_min_flow_pushed = 0 - min_flow_pushed;

      
      atomicSub(&g_down_weight[thid], min_flow_pushed);
      atomicAdd(&g_up_weight[thid+width1], min_flow_pushed);
      atomicSub(&g_push_reser[thid], min_flow_pushed);
      atomicAdd(&g_push_reser[thid+width1], min_flow_pushed);
      
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      /*
      asm volatile(// Temp Registers
                   ".reg .s32 m0;\n\t"    // temp reg m0 (min_flow_pushed)
                   ".reg .s32 m1;\n\t"    // temp reg m1 (negative min_flow_pushed)
                   ".reg .s32 m2;\n\t"    // temp reg m2 (atomicSub(downWeightTidAddr) result)
                   ".reg .s32 m3;\n\t"    // temp reg m3 (atomicAdd(upWeightTidPlusWidth1Addr) result)
                   ".reg .s32 m4;\n\t"    // temp reg m4 (atomicSub(pushTidAddr) result)
                   ".reg .s32 m5;\n\t"    // temp reg m5 (atomicAdd(pushTidPlusWidth1Addr) result)
                   // PTX Instructions
                   "mov.s32 m0, %0;\n\t"
                   "mov.s32 m1, %1;\n\t"
                   "atom.add.s32 m2, [%1], m1;\n\t" // atomicSub for downWeightTidAddr
                   "atom.add.s32 m3, [%2], m0;\n\t" // atomicAdd for upWeightTidPlusWidth1Addr
                   "atom.add.s32 m4, [%3], m1;\n\t" // atomicSub for pushTidAddr
                   "atom.add.s32 m5, [%4], m0;"     // atomicAdd for pushTidPlusWidth1Addr
                   // no outputs
                   // inputs
                   :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                      "l"(downWeightTidAddr), "l"(upWeightTidPlusWidth1Addr),
                      "l"(pushTidAddr), "l"(pushTidPlusWidth1Addr)
                   );
    */} else {
      atomicSub( &g_pull_down[thid+width1], 1);
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
*/
}

__global__ void
kernel_End( int * g_stochastic, int * g_count_blocks, int g_counter)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region or
    // the special region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int thid = blockIdx.x * blockDim.x + threadIdx.x; 
  if( thid < g_counter )
  {
    if( g_stochastic[thid] == 1 ) {
      atomicAdd(g_count_blocks,1);
      //(*g_count_blocks) = (*g_count_blocks) + 1; 
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region or
    // the special region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

__global__ void
kernel_push1_start_atomic( int * g_left_weight, int * g_right_weight,
                           int * g_down_weight, int * g_up_weight,
                           int * g_sink_weight, int * g_push_reser,
                           int * g_relabel_mask, int * g_graph_height,
                           int * g_height_write, int graph_size, int width,
                           int rows, int graph_size1, int width1, int rows1,
                           int d_relabel, int * d_stochastic, int d_counter,
                           bool * d_finish)
{
  /*
    if (threadIdx.x == 0) {
      // all of the arrays accessed in this kernel are either in the read-only region,
      // special region or the relaxed atomic region
      __denovo_setAcquireRegion(SPECIAL_REGION);
      __denovo_addAcquireRegion(RELAX_ATOM_REGION);
      __denovo_addAcquireRegion(READ_ONLY_REGION);
    }
    __syncthreads();
*/
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
    int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
    int thid = __umul24( y, width1 ) + x;

    __shared__ int height_fn[356];

    int temp_mult = __umul24(y1+1, 34 ) + x1 + 1, temp_mult1 = __umul24(y1,32) + x1;

    height_fn[temp_mult] = g_graph_height[thid];

    (threadIdx.x == 31 && x < width1 - 1 ) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
    (threadIdx.x == 0 && x > 0 ) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
    (threadIdx.y == 7 && y < rows1 - 1 ) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
    (threadIdx.y == 0 && y > 0 ) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;
    __syncthreads();

    int flow_push = 0, min_flow_pushed = 0;
    //flow_push = g_push_reser[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);

    if( thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width-1 && x > 0 && y < rows-1 && y > 0 )
    {
      int temp_weight = 0;

      temp_weight = g_sink_weight[thid];
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        g_sink_weight[thid] = temp_weight;
        atomicSub(&g_push_reser[thid], min_flow_pushed);

        flow_push = flow_push - min_flow_pushed;
      }
    }

    __syncthreads();
    //min_flow_pushed = g_left_weight[thid];
    min_flow_pushed = atomicAdd(&g_left_weight[thid], 0);

    ( flow_push > 0 && ( ((height_fn[temp_mult] == height_fn[temp_mult-1] + 1 ) && min_flow_pushed > 0  ) ||( (height_fn[temp_mult] == height_fn[temp_mult+1]+1 ) && /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0) || ( ( height_fn[temp_mult] == height_fn[temp_mult+34]+1 ) && /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0) || ( (height_fn[temp_mult] == height_fn[temp_mult-34]+1 ) && /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 ) || ( height_fn[temp_mult] == 1 && g_sink_weight[thid] > 0 )  ) ) ? g_relabel_mask[thid] = 1 : g_relabel_mask[thid] = 0;

    if(thid < graph_size1 && x < width - 1  && x > 0 && y < rows - 1  && y > 0  )
    {
      if(g_sink_weight[thid] > 0)
      {
        g_height_write[thid] = 1;
      }
      else
      {
        int min_height = graph_size;
        (min_flow_pushed > 0 && min_height > height_fn[temp_mult - 1] ) ? min_height = height_fn[temp_mult - 1] : 0;
        (/*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0 && min_height > height_fn[temp_mult + 1]) ? min_height = height_fn[temp_mult + 1] : 0;
        (/*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0 && min_height > height_fn[temp_mult + 34] ) ? min_height = height_fn[temp_mult + 34] : 0;
        (/*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 && min_height > height_fn[temp_mult - 34] ) ? min_height = height_fn[temp_mult - 34] : 0;
        g_height_write[thid] = min_height + 1;
      }
    }
/*
    if (threadIdx.x == 0) {
      // all of the arrays accessed in this kernel are either in the read-only region,
      // special region or the relaxed atomic region
      __denovo_gpuEpilogue(SPECIAL_REGION);
      __denovo_gpuEpilogue(RELAX_ATOM_REGION);
      __denovo_gpuEpilogue(READ_ONLY_REGION);
    }
*/
  }

__global__ void
kernel_push1_stochastic( int * g_left_weight, int * g_right_weight,
                         int * g_down_weight, int * g_up_weight,
                         int * g_sink_weight, int * g_push_reser,
                         int * g_pull_left, int * g_pull_right,
                         int * g_pull_down, int * g_pull_up,
                         int * g_relabel_mask, int * g_graph_height,
                         int * g_height_write, int graph_size, int width,
                         int rows, int graph_size1, int width1, int rows1,
                         int * d_stochastic, int g_block_num )
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  if(d_stochastic[blockIdx.y * g_block_num + blockIdx.x] == 1 )
  {
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
    int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
    int thid = __umul24( y, width1 ) + x;
/*
    // for use in inlined assembly
    int * pushTidMinus1Addr = &(g_push_reser[thid-1]);
    int * pushTidMinusWidth1Addr = &(g_push_reser[thid-width1]);
    int * pushTidAddr = &(g_push_reser[thid]);
    int * pushTidPlusWidth1Addr = &(g_push_reser[thid+width1]);
    int * pushTidPlus1Addr = &(g_push_reser[thid+1]);
    int * leftWeightTidAddr = &(g_left_weight[thid]);
    int * leftWeightTidPlus1Addr = &(g_left_weight[thid+1]);
    int * rightWeightTidMinus1Addr = &(g_right_weight[thid-1]);
    int * rightWeightTidAddr = &(g_right_weight[thid]);
    int * upWeightTidAddr = &(g_up_weight[thid]);
    int * upWeightTidPlusWidth1Addr = &(g_up_weight[thid+width1]);
    int * downWeightTidMinusWidth1Addr = &(g_down_weight[thid-width1]);
    int * downWeightTidAddr = &(g_down_weight[thid]);
*/
    __shared__ int height_fn[356];
    
    int temp_mult = __umul24(y1+1, 34 ) + x1 + 1, temp_mult1 = __umul24(y1,32) + x1;

    height_fn[temp_mult] = g_graph_height[thid];

    (threadIdx.x == 31 && x < width1 - 1 ) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
    (threadIdx.x == 0 && x > 0 ) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
    (threadIdx.y == 7 && y < rows1 - 1 ) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
    (threadIdx.y == 0 && y > 0 ) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;

    __syncthreads();

    int flow_push = 0, min_flow_pushed = 0, neg_min_flow_pushed = 0;
    //flow_push = g_push_reser[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);

    if( thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width-1 && x > 0 && y < rows-1 && y > 0 )
    {
      int temp_weight = 0;

      temp_weight = g_sink_weight[thid];
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        g_sink_weight[thid] = temp_weight;
        atomicSub(&g_push_reser[thid], min_flow_pushed);
      }

      //flow_push = g_push_reser[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);

      min_flow_pushed = flow_push;

      //temp_weight = g_left_weight[thid];
      temp_weight = atomicAdd(&g_left_weight[thid], 0);

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 1] + 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_left_weight[thid], min_flow_pushed);
        atomicAdd(&g_right_weight[thid-1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid-1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
		/*
		asm volatile(// Temp Registers
                     ".reg .s32 n0;\n\t"    // temp reg n0 (min_flow_pushed)
                     ".reg .s32 n1;\n\t"    // temp reg n1 (negative min_flow_pushed)
                     ".reg .s32 n2;\n\t"    // temp reg n2 (atomicSub(leftWeightTidAddr) result)
                     ".reg .s32 n3;\n\t"    // temp reg n3 (atomicAdd(rightWeightTidMinus1Addr) result)
                     ".reg .s32 n4;\n\t"    // temp reg n4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 n5;\n\t"    // temp reg n5 (atomicAdd(pushTidMinus1Addr) result)
                     // PTX Instructions
                     "mov.s32 n0, %0;\n\t"
                     "mov.s32 n1, %1;\n\t"
                     "atom.add.s32 n2, [%1], n1;\n\t" // atomicSub for leftWeightTidAddr
                     "atom.add.s32 n3, [%2], n0;\n\t" // atomicAdd for rightWeightTidMinus1Addr
                     "atom.add.s32 n4, [%3], n1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 n5, [%4], n0;"     // atomicAdd for pushTidMinus1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(leftWeightTidAddr), "l"(rightWeightTidMinus1Addr),
                        "l"(pushTidAddr), "l"(pushTidMinus1Addr)
                     );
	  */
					} else {
        atomicSub(&g_pull_left[thid-1], 1);
      }

      //flow_push = g_push_reser[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      min_flow_pushed = flow_push;
      //temp_weight = g_up_weight[thid];
      temp_weight = atomicAdd(&g_up_weight[thid], 0);

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 34] + 1)
      {
        (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_up_weight[thid], min_flow_pushed);
        atomicAdd(&g_down_weight[thid-width1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid-width1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
		/*
		asm volatile(// Temp Registers
                     ".reg .s32 o0;\n\t"    // temp reg o0 (min_flow_pushed)
                     ".reg .s32 o1;\n\t"    // temp reg o1 (negative min_flow_pushed)
                     ".reg .s32 o2;\n\t"    // temp reg o2 (atomicSub(upWeightTidAddr) result)
                     ".reg .s32 o3;\n\t"    // temp reg o3 (atomicAdd(downWeightTidMinusWidth1Addr) result)
                     ".reg .s32 o4;\n\t"    // temp reg o4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 o5;\n\t"    // temp reg o5 (atomicAdd(pushTidMinusWidth1Addr) result)
                     // PTX Instructions
                     "mov.s32 o0, %0;\n\t"
                     "mov.s32 o1, %1;\n\t"
                     "atom.add.s32 o2, [%1], o1;\n\t" // atomicSub for upWeightTidAddr
                     "atom.add.s32 o3, [%2], o0;\n\t" // atomicAdd for downWeightTidMinusWidth1Addr
                     "atom.add.s32 o4, [%3], o1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 o5, [%4], o0;"     // atomicAdd for pushTidMinusWidth1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(upWeightTidAddr), "l"(downWeightTidMinusWidth1Addr),
                        "l"(pushTidAddr), "l"(pushTidMinusWidth1Addr)
                     );
	  */
					} else {
        atomicSub(&g_pull_up[thid - width1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_right_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      min_flow_pushed = flow_push;
      temp_weight = atomicAdd(&g_right_weight[thid], 0);

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 1] + 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

		
		atomicSub(&g_right_weight[thid], min_flow_pushed);
        atomicAdd(&g_left_weight[thid+1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid+1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
		/*
		asm volatile(// Temp Registers
                     ".reg .s32 q0;\n\t"    // temp reg q0 (min_flow_pushed)
                     ".reg .s32 q1;\n\t"    // temp reg q1 (negative min_flow_pushed)
                     ".reg .s32 q2;\n\t"    // temp reg q2 (atomicSub(rightWeightTidAddr) result)
                     ".reg .s32 q3;\n\t"    // temp reg q3 (atomicAdd(leftWeightTidPlus1Addr) result)
                     ".reg .s32 q4;\n\t"    // temp reg q4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 q5;\n\t"    // temp reg q5 (atomicAdd(pushTidPlus1Addr) result)
                     // PTX Instructions
                     "mov.s32 q0, %0;\n\t"
                     "mov.s32 q1, %1;\n\t"
                     "atom.add.s32 q2, [%1], q1;\n\t" // atomicSub for rightWeightTidAddr
                     "atom.add.s32 q3, [%2], q0;\n\t" // atomicAdd for leftWeightTidPlus1Addr
                     "atom.add.s32 q4, [%3], q1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 q5, [%4], q0;"     // atomicAdd for pushTidPlus1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(rightWeightTidAddr), "l"(leftWeightTidPlus1Addr),
                        "l"(pushTidAddr), "l"(pushTidPlus1Addr)
                     );
	  */
					} else {
        atomicSub( &g_pull_right[thid + 1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_down_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_down_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 34] + 1 )
      {
        (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_down_weight[thid], min_flow_pushed);
        atomicAdd(&g_up_weight[thid+width1], min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid+width1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
		/*
		asm volatile(// Temp Registers
                     ".reg .s32 s0;\n\t"    // temp reg s0 (min_flow_pushed)
                     ".reg .s32 s1;\n\t"    // temp reg s1 (negative min_flow_pushed)
                     ".reg .s32 s2;\n\t"    // temp reg s2 (atomicSub(downWeightTidAddr) result)
                     ".reg .s32 s3;\n\t"    // temp reg s3 (atomicAdd(upWeightTidPlusWidth1Addr) result)
                     ".reg .s32 s4;\n\t"    // temp reg s4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 s5;\n\t"    // temp reg s5 (atomicAdd(pushTidPlusWidth1Addr) result)
                     // PTX Instructions
                     "mov.s32 s0, %0;\n\t"
                     "mov.s32 s1, %1;\n\t"
                     "atom.add.s32 s2, [%1], s1;\n\t" // atomicSub for downWeightTidAddr
                     "atom.add.s32 s3, [%2], s0;\n\t" // atomicAdd for upWeightTidPlusWidth1Addr
                     "atom.add.s32 s4, [%3], s1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 s5, [%4], s0;"     // atomicAdd for pushTidPlusWidth1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(downWeightTidAddr), "l"(upWeightTidPlusWidth1Addr),
                        "l"(pushTidAddr), "l"(pushTidPlusWidth1Addr)
                     );
	  */
		} else {
        atomicSub( &g_pull_down[thid+width1], 1);
      }
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
 */ 
}

// ** UNUSED ...?
__global__ void
kernel_push2_stochastic( int * g_left_weight, int * g_right_weight,
                         int * g_down_weight, int * g_up_weight,
                         int * g_sink_weight, int * g_push_reser,
                         int * g_pull_left, int * g_pull_right,
                         int * g_pull_down, int * g_pull_up,
                         int * g_relabel_mask, int * g_graph_height,
                         int * g_height_write, int graph_size, int width,
                         int rows, int graph_size1, int width1, int rows1,
                         int d_relabel, int * d_stochastic, int * d_counter,
                         bool * d_finish )
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  if(d_stochastic[blockIdx.y * 20 + blockIdx.x] == 1 )
  {
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
    int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
    int thid = __umul24( y, width1 ) + x;
/*
    // for use in inlined assembly
    int * pushTidMinus1Addr = &(g_push_reser[thid-1]);
    int * pushTidMinusWidth1Addr = &(g_push_reser[thid-width1]);
    int * pushTidAddr = &(g_push_reser[thid]);
    int * pushTidPlusWidth1Addr = &(g_push_reser[thid+width1]);
    int * pushTidPlus1Addr = &(g_push_reser[thid+1]);
    int * leftWeightTidAddr = &(g_left_weight[thid]);
    int * leftWeightTidPlus1Addr = &(g_left_weight[thid+1]);
    int * rightWeightTidMinus1Addr = &(g_right_weight[thid-1]);
    int * rightWeightTidAddr = &(g_right_weight[thid]);
    int * upWeightTidAddr = &(g_up_weight[thid]);
    int * upWeightTidPlusWidth1Addr = &(g_up_weight[thid+width1]);
    int * downWeightTidMinusWidth1Addr = &(g_down_weight[thid-width1]);
    int * downWeightTidAddr = &(g_down_weight[thid]);
*/
    __shared__ int height_fn[356];

    int temp_mult = __umul24(y1+1, 34 ) + x1 + 1, temp_mult1 = __umul24(y1,32) + x1;

    height_fn[temp_mult] = g_graph_height[thid];

    (threadIdx.x == 31 && x < width1 - 1 ) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
    (threadIdx.x == 0 && x > 0 ) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
    (threadIdx.y == 7 && y < rows1 - 1 ) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
    (threadIdx.y == 0 && y > 0 ) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;

    __syncthreads();

    int flow_push = 0, min_flow_pushed = 0, neg_min_flow_pushed = 0;
    //flow_push = g_push_reser[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);

    if( thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width-1 && x > 0 && y < rows-1 && y > 0 )
    {
      int temp_weight = 0;

      temp_weight = g_sink_weight[thid];
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        g_sink_weight[thid] = temp_weight;
        atomicSub(&g_push_reser[thid], min_flow_pushed);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_left_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_left_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 1] + 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_left_weight[thid], min_flow_pushed);
        atomicAdd(&g_right_weight[thid-1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid-1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 t0;\n\t"    // temp reg t0 (min_flow_pushed)
                     ".reg .s32 t1;\n\t"    // temp reg t1 (negative min_flow_pushed)
                     ".reg .s32 t2;\n\t"    // temp reg t2 (atomicSub(leftWeightTidAddr) result)
                     ".reg .s32 t3;\n\t"    // temp reg t3 (atomicAdd(rightWeightTidMinus1Addr) result)
                     ".reg .s32 t4;\n\t"    // temp reg t4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 t5;\n\t"    // temp reg t5 (atomicAdd(pushTidMinus1Addr) result)
                     // PTX Instructions
                     "mov.s32 t0, %0;\n\t"
                     "mov.s32 t1, %1;\n\t"
                     "atom.add.s32 t2, [%1], t1;\n\t" // atomicSub for leftWeightTidAddr
                     "atom.add.s32 t3, [%2], t0;\n\t" // atomicAdd for rightWeightTidMinus1Addr
                     "atom.add.s32 t4, [%3], t1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 t5, [%4], t0;"     // atomicAdd for pushTidMinus1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(leftWeightTidAddr), "l"(rightWeightTidMinus1Addr),
                        "l"(pushTidAddr), "l"(pushTidMinus1Addr)
                     );
      */
                    } else {
        atomicSub(&g_pull_left[thid-1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_up_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_up_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 34] + 1)
      {
        (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_up_weight[thid], min_flow_pushed);
        atomicAdd(&g_down_weight[thid-width1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid-width1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 u0;\n\t"    // temp reg u0 (min_flow_pushed)
                     ".reg .s32 u1;\n\t"    // temp reg u1 (negative min_flow_pushed)
                     ".reg .s32 u2;\n\t"    // temp reg u2 (atomicSub(upWeightTidAddr) result)
                     ".reg .s32 u3;\n\t"    // temp reg u3 (atomicAdd(downWeightTidMinusWidth1Addr) result)
                     ".reg .s32 u4;\n\t"    // temp reg u4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 u5;\n\t"    // temp reg u5 (atomicAdd(pushTidMinusWidth1Addr) result)
                     // PTX Instructions
                     "mov.s32 u0, %0;\n\t"
                     "mov.s32 u1, %1;\n\t"
                     "atom.add.s32 u2, [%1], u1;\n\t" // atomicSub for upWeightTidAddr
                     "atom.add.s32 u3, [%2], u0;\n\t" // atomicAdd for downWeightTidMinusWidth1Addr
                     "atom.add.s32 u4, [%3], u1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 u5, [%4], u0;"     // atomicAdd for pushTidMinusWidth1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(upWeightTidAddr), "l"(downWeightTidMinusWidth1Addr),
                        "l"(pushTidAddr), "l"(pushTidMinusWidth1Addr)
                     );
      */
      } else {
        atomicSub(&g_pull_up[thid - width1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_right_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_right_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 1] + 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_right_weight[thid], min_flow_pushed);
        atomicAdd(&g_left_weight[thid+1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid+1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 v0;\n\t"    // temp reg v0 (min_flow_pushed)
                     ".reg .s32 v1;\n\t"    // temp reg v1 (negative min_flow_pushed)
                     ".reg .s32 v2;\n\t"    // temp reg v2 (atomicSub(rightWeightTidAddr) result)
                     ".reg .s32 v3;\n\t"    // temp reg v3 (atomicAdd(leftWeightTidPlus1Addr) result)
                     ".reg .s32 v4;\n\t"    // temp reg v4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 v5;\n\t"    // temp reg v5 (atomicAdd(pushTidPlus1Addr) result)
                     // PTX Instructions
                     "mov.s32 v0, %0;\n\t"
                     "mov.s32 v1, %1;\n\t"
                     "atom.add.s32 v2, [%1], v1;\n\t" // atomicSub for rightWeightTidAddr
                     "atom.add.s32 v3, [%2], v0;\n\t" // atomicAdd for leftWeightTidPlus1Addr
                     "atom.add.s32 v4, [%3], v1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 v5, [%4], v0;"     // atomicAdd for pushTidPlus1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(rightWeightTidAddr), "l"(leftWeightTidPlus1Addr),
                        "l"(pushTidAddr), "l"(pushTidPlus1Addr)
                     );
      */               
      } else {
        atomicSub( &g_pull_right[thid + 1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_down_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_down_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 34] + 1 )
      {
        (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_down_weight[thid], min_flow_pushed);
        atomicAdd(&g_up_weight[thid+width1], min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid+width1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 w0;\n\t"    // temp reg w0 (min_flow_pushed)
                     ".reg .s32 w1;\n\t"    // temp reg w1 (negative min_flow_pushed)
                     ".reg .s32 w2;\n\t"    // temp reg w2 (atomicSub(downWeightTidAddr) result)
                     ".reg .s32 w3;\n\t"    // temp reg w3 (atomicAdd(upWeightTidPlusWidth1Addr) result)
                     ".reg .s32 w4;\n\t"    // temp reg w4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 w5;\n\t"    // temp reg w5 (atomicAdd(pushTidPlusWidth1Addr) result)
                     // PTX Instructions
                     "mov.s32 w0, %0;\n\t"
                     "mov.s32 w1, %1;\n\t"
                     "atom.add.s32 w2, [%1], w1;\n\t" // atomicSub for downWeightTidAddr
                     "atom.add.s32 w3, [%2], w0;\n\t" // atomicAdd for upWeightTidPlusWidth1Addr
                     "atom.add.s32 w4, [%3], w1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 w5, [%4], w0;"     // atomicAdd for pushTidPlusWidth1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(downWeightTidAddr), "l"(upWeightTidPlusWidth1Addr),
                        "l"(pushTidAddr), "l"(pushTidPlusWidth1Addr)
                     );
      */
                    } else {
        atomicSub( &g_pull_down[thid+width1], 1);
      }
    }

    __syncthreads(); 
    //min_flow_pushed = g_left_weight[thid];
    min_flow_pushed = atomicAdd(&g_left_weight[thid], 0);
    //flow_push = g_push_reser[thid];
    flow_push = atomicAdd(&g_push_reser[thid], 0);

    if (flow_push <= 0 ||
        (/*g_left_weight[thid]*/atomicAdd(&g_left_weight[thid], 0) == 0 &&
         /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) == 0 &&
         /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) == 0 &&
         /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) == 0 &&
         g_sink_weight[thid] == 0))
      g_relabel_mask[thid] = 2;
    else
    {
      ( flow_push > 0 && ( ( (height_fn[temp_mult] == height_fn[temp_mult-1] + 1 ) && /*g_left_weight[thid]*/atomicAdd(&g_left_weight[thid], 0) > 0  ) ||( (height_fn[temp_mult] == height_fn[temp_mult+1]+1 ) && /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0) || ( ( height_fn[temp_mult] == height_fn[temp_mult+34]+1 ) && /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0) || ( (height_fn[temp_mult] == height_fn[temp_mult-34]+1 ) && /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 ) || ( height_fn[temp_mult] == 1 && g_sink_weight[thid] > 0 )  ) ) ? g_relabel_mask[thid] = 1 : g_relabel_mask[thid] = 0;
    }
    __syncthreads();

    if( thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width-1 && x > 0 && y < rows-1 && y > 0 )
    {
      int temp_weight = 0;

      temp_weight = g_sink_weight[thid];
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        g_sink_weight[thid] = temp_weight;
        atomicSub(&g_push_reser[thid], min_flow_pushed);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_left_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_left_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 1] + 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_left_weight[thid], min_flow_pushed);
        atomicAdd(&g_right_weight[thid-1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid-1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 x0;\n\t"    // temp reg x0 (min_flow_pushed)
                     ".reg .s32 x1;\n\t"    // temp reg x1 (negative min_flow_pushed)
                     ".reg .s32 x2;\n\t"    // temp reg x2 (atomicSub(leftWeightTidAddr) result)
                     ".reg .s32 x3;\n\t"    // temp reg x3 (atomicAdd(rightWeightTidMinus1Addr) result)
                     ".reg .s32 x4;\n\t"    // temp reg x4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 x5;\n\t"    // temp reg x5 (atomicAdd(pushTidMinus1Addr) result)
                     // PTX Instructions
                     "mov.s32 x0, %0;\n\t"
                     "mov.s32 x1, %1;\n\t"
                     "atom.add.s32 x2, [%1], x1;\n\t" // atomicSub for leftWeightTidAddr
                     "atom.add.s32 x3, [%2], x0;\n\t" // atomicAdd for 
                     "atom.add.s32 x4, [%3], x1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 x5, [%4], x0;"     // atomicAdd for pushTidMinus1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(leftWeightTidAddr), "l"(rightWeightTidMinus1Addr),
                        "l"(pushTidAddr), "l"(pushTidMinus1Addr)
                     );
      */
                    } else {
        atomicSub(&g_pull_left[thid-1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_up_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_up_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 34] + 1)
      {
        (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_up_weight[thid], min_flow_pushed);
        atomicAdd(&g_down_weight[thid-width1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid-width1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 y0;\n\t"    // temp reg y0 (min_flow_pushed)
                     ".reg .s32 y1;\n\t"    // temp reg y1 (negative min_flow_pushed)
                     ".reg .s32 y2;\n\t"    // temp reg y2 (atomicSub(upWeightTidAddr) result)
                     ".reg .s32 y3;\n\t"    // temp reg y3 (atomicAdd(downWeightTidMinusWidth1Addr) result)
                     ".reg .s32 y4;\n\t"    // temp reg y4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 y5;\n\t"    // temp reg y5 (atomicAdd(pushTidMinusWidth1Addr) result)
                     // PTX Instructions
                     "mov.s32 y0, %0;\n\t"
                     "mov.s32 y1, %1;\n\t"
                     "atom.add.s32 y2, [%1], y1;\n\t" // atomicSub for upWeightTidAddr
                     "atom.add.s32 y3, [%2], y0;\n\t" // atomicAdd for downWeightTidMinusWidth1Addr
                     "atom.add.s32 y4, [%3], y1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 y5, [%4], y0;"     // atomicAdd for pushTidMinusWidth1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(upWeightTidAddr), "l"(downWeightTidMinusWidth1Addr),
                        "l"(pushTidAddr), "l"(pushTidMinusWidth1Addr)
                     );
      */
                    } else {
        atomicSub(&g_pull_up[thid - width1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_right_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_right_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 1] + 1 )
      {
        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_right_weight[thid], min_flow_pushed);
        atomicAdd(&g_left_weight[thid+1],min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid+1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 z0;\n\t"    // temp reg z0 (min_flow_pushed)
                     ".reg .s32 z1;\n\t"    // temp reg z1 (negative min_flow_pushed)
                     ".reg .s32 z2;\n\t"    // temp reg z2 (atomicSub(rightWeightTidAddr) result)
                     ".reg .s32 z3;\n\t"    // temp reg z3 (atomicAdd(leftWeightTidPlus1Addr) result)
                     ".reg .s32 z4;\n\t"    // temp reg z4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 z5;\n\t"    // temp reg z5 (atomicAdd(pushTidPlus1Addr) result)
                     // PTX Instructions
                     "mov.s32 z0, %0;\n\t"
                     "mov.s32 z1, %1;\n\t"
                     "atom.add.s32 z2, [%1], z1;\n\t" // atomicSub for rightWeightTidAddr
                     "atom.add.s32 z3, [%2], z0;\n\t" // atomicAdd for leftWeightTidPlus1Addr
                     "atom.add.s32 z4, [%3], z1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 z5, [%4], z0;"     // atomicAdd for pushTidPlus1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(rightWeightTidAddr), "l"(leftWeightTidPlus1Addr),
                        "l"(pushTidAddr), "l"(pushTidPlus1Addr)
                     );
      */
                    } else {
        atomicSub( &g_pull_right[thid + 1], 1);
      }

      //flow_push = g_push_reser[thid];
      //temp_weight = g_down_weight[thid];
      flow_push = atomicAdd(&g_push_reser[thid], 0);
      temp_weight = atomicAdd(&g_down_weight[thid], 0);
      min_flow_pushed = flow_push;

      if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 34] + 1 )
      {
        (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0;
        temp_weight = temp_weight - min_flow_pushed;
        neg_min_flow_pushed = 0 - min_flow_pushed;

        
        atomicSub(&g_down_weight[thid], min_flow_pushed);
        atomicAdd(&g_up_weight[thid+width1], min_flow_pushed);
        atomicSub(&g_push_reser[thid], min_flow_pushed);
        atomicAdd(&g_push_reser[thid+width1], min_flow_pushed);
        
        // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
        // same temp reg names
        /*
        asm volatile(// Temp Registers
                     ".reg .s32 aa0;\n\t"    // temp reg aa0 (min_flow_pushed)
                     ".reg .s32 aa1;\n\t"    // temp reg aa1 (negative min_flow_pushed)
                     ".reg .s32 aa2;\n\t"    // temp reg aa2 (atomicSub(downWeightTidAddr) result)
                     ".reg .s32 aa3;\n\t"    // temp reg aa3 (atomicAdd(upWeightTidPlusWidth1Addr) result)
                     ".reg .s32 aa4;\n\t"    // temp reg aa4 (atomicSub(pushTidAddr) result)
                     ".reg .s32 aa5;\n\t"    // temp reg aa5 (atomicAdd(pushTidPlusWidth1Addr) result)
                     // PTX Instructions
                     "mov.s32 aa0, %0;\n\t"
                     "mov.s32 aa1, %1;\n\t"
                     "atom.add.s32 aa2, [%1], aa1;\n\t" // atomicSub for downWeightTidAddr
                     "atom.add.s32 aa3, [%2], aa0;\n\t" // atomicAdd for upWeightTidPlusWidth1Addr
                     "atom.add.s32 aa4, [%3], aa1;\n\t" // atomicSub for pushTidAddr
                     "atom.add.s32 aa5, [%4], aa0;"     // atomicAdd for pushTidPlusWidth1Addr
                     // no outputs
                     // inputs
                     :: "r"(min_flow_pushed), "r"(neg_min_flow_pushed),
                        "l"(downWeightTidAddr), "l"(upWeightTidPlusWidth1Addr),
                        "l"(pushTidAddr), "l"(pushTidPlusWidth1Addr)
                     );
      */
                    } else {
        atomicSub( &g_pull_down[thid+width1], 1);
      }
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

__global__ void
kernel_bfs_t(int *g_push_reser, int  *g_sink_weight, int *g_graph_height,
             bool *g_pixel_mask, int vertex_num, int width, int height,
             int vertex_num1, int width1, int height1)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region or
    // the special region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int thid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if(thid < vertex_num && g_pixel_mask[thid] == true )
  {
    int col = thid % width1, row = thid / width1;

    //if(col > 0 && row > 0 && col < width - 1 && row < height - 1 && g_push_reser[thid] > 0 )
    if(col > 0 && row > 0 && col < width - 1 && row < height - 1 && atomicAdd(&g_push_reser[thid], 0) > 0 )
    {
      g_graph_height[thid] = 1;
      g_pixel_mask[thid] = false;
    }
    else
    {
      if(g_sink_weight[thid] > 0)
      {
        g_graph_height[thid] = -1;
        g_pixel_mask[thid] = false;
      }
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region or 
    // the special region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

__global__ void
kernel_push_stochastic1( int *g_push_reser, int *s_push_reser,
                         int *g_count_blocks, bool *g_finish, int g_block_num,
                         int width1)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the special region or
    // the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
  int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
  int thid = __umul24( y, width1 ) + x;
  //s_push_reser[thid] = g_push_reser[thid];
  s_push_reser[thid] = atomicAdd(&g_push_reser[thid], 0);

  if( thid == 0 )
  {
    if(/*(*g_count_blocks)*/atomicAdd(g_count_blocks, 0) < 50 )
      (*g_finish) = false; 
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the special region or
    // the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

__global__ void
kernel_push_stochastic2( int *g_push_reser, int *s_push_reser,
                         int *d_stochastic, int g_block_num, int width1)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region or
    // the special region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
  int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
  int thid = __umul24( y, width1 ) + x;

  int stochastic = 0;
  
  //stochastic = ( s_push_reser[thid] - g_push_reser[thid]);
  stochastic = ( s_push_reser[thid] - atomicAdd(&g_push_reser[thid], 0));
  if(stochastic != 0)
  {
    d_stochastic[blockIdx.y * g_block_num + blockIdx.x] = 1;
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only
    // region or the special region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

__global__ void
kernel_push1_start_stochastic( int *g_left_weight, int *g_right_weight,
                               int *g_down_weight, int *g_up_weight,
                               int *g_sink_weight, int *g_push_reser,
                               int *g_relabel_mask, int *g_graph_height,
                               int *g_height_write, int graph_size,
                               int width, int rows, int graph_size1, int width1,
                               int rows1, int d_relabel, int *d_stochastic,
                               int d_counter, bool *d_finish )
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int x1 = threadIdx.x;
  int y1 = threadIdx.y;
  int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
  int y  = __umul24( blockIdx.y, blockDim.y ) + threadIdx.y;
  int thid = __umul24( y, width1 ) + x;

  __shared__ int height_fn[356];

  int temp_mult = __umul24(y1+1, 34 ) + x1 + 1, temp_mult1 = __umul24(y1,32) + x1;

  height_fn[temp_mult] = g_graph_height[thid];

  (threadIdx.x == 31 && x < width1 - 1 ) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
  (threadIdx.x == 0 && x > 0 ) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
  (threadIdx.y == 7 && y < rows1 - 1 ) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
  (threadIdx.y == 0 && y > 0 ) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;

  __syncthreads();

  int flow_push = 0, min_flow_pushed = 0;
  //flow_push = g_push_reser[thid];
  flow_push = atomicAdd(&g_push_reser[thid], 0);

  if( thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width-1 && x > 0 && y < rows-1 && y > 0 )
  {
    int temp_weight = 0;


    temp_weight = g_sink_weight[thid];
    min_flow_pushed = flow_push;

    if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1 )
    {
      (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
      temp_weight = temp_weight - min_flow_pushed;
      g_sink_weight[thid] = temp_weight;
      atomicSub(&g_push_reser[thid], min_flow_pushed);

      flow_push = flow_push - min_flow_pushed;
    }
  }

  __syncthreads();
  //min_flow_pushed = g_left_weight[thid];
  min_flow_pushed = atomicAdd(&g_left_weight[thid], 0);

  ( flow_push > 0 && ( ((height_fn[temp_mult] == height_fn[temp_mult-1] + 1 ) && min_flow_pushed > 0  ) ||( (height_fn[temp_mult] == height_fn[temp_mult+1]+1 ) && /*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0) || ( ( height_fn[temp_mult] == height_fn[temp_mult+34]+1 ) && /*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0) || ( (height_fn[temp_mult] == height_fn[temp_mult-34]+1 ) && /*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 ) || ( height_fn[temp_mult] == 1 && g_sink_weight[thid] > 0 )  ) ) ? g_relabel_mask[thid] = 1 : g_relabel_mask[thid] = 0;

  if(thid < graph_size1 && x < width - 1  && x > 0 && y < rows - 1  && y > 0  )
  {
    if(g_sink_weight[thid] > 0)
    {
      g_height_write[thid] = 1;
    }
    else
    {
      int min_height = graph_size;
      (min_flow_pushed > 0 && min_height > height_fn[temp_mult - 1] ) ? min_height = height_fn[temp_mult - 1] : 0;
      (/*g_right_weight[thid]*/atomicAdd(&g_right_weight[thid], 0) > 0 && min_height > height_fn[temp_mult + 1]) ? min_height = height_fn[temp_mult + 1] : 0;
      (/*g_down_weight[thid]*/atomicAdd(&g_down_weight[thid], 0) > 0 && min_height > height_fn[temp_mult + 34] ) ? min_height = height_fn[temp_mult + 34] : 0;
      (/*g_up_weight[thid]*/atomicAdd(&g_up_weight[thid], 0) > 0 && min_height > height_fn[temp_mult - 34] ) ? min_height = height_fn[temp_mult - 34] : 0;
      g_height_write[thid] = min_height + 1;
    } 
 }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

__global__ void
kernel_bfs(int *g_left_weight, int *g_right_weight, int *g_down_weight,
           int *g_up_weight, int *g_graph_height, bool *g_pixel_mask,
           int vertex_num, int width, int height, int vertex_num1, int width1,
           int height1, bool *g_over, int g_counter)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the special
    // region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  /*******************************
   *threadId is calculated ******
   *****************************/
  int thid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if(thid < vertex_num && g_pixel_mask[thid] == true)
  {
    int col = thid % width1, row = thid / width1;

    if(col < width - 1 && col > 0 && row < height - 1 && row > 0 )
    {
      int height_l = 0, height_d = 0, height_u = 0, height_r = 0;
      height_r = g_graph_height[thid+1];
      height_l = g_graph_height[thid-1];
      height_d = g_graph_height[thid+width1];
      height_u = g_graph_height[thid-width1];

      if(((height_l == g_counter &&
           /*g_right_weight[thid-1]*/atomicAdd(&g_right_weight[thid-1], 0) > 0)) ||
         ((height_d == g_counter &&
           /*g_up_weight[thid+width1]*/atomicAdd(&g_up_weight[thid+width1], 0) > 0) ||
          ( height_r == g_counter &&
            /*g_left_weight[thid+1]*/atomicAdd(&g_left_weight[thid+1], 0) > 0 ) ||
          ( height_u == g_counter &&
            /*g_down_weight[thid-width1]*/atomicAdd(&g_down_weight[thid-width1], 0) > 0 ) ))
      {
        g_graph_height[thid] = g_counter + 1;
        g_pixel_mask[thid] = false;
        *g_over = true;
      }
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the special
    // region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
*/  
}

/************************************************************
 * functions to construct the graph on the device          **
 * *********************************************************/
__device__
void add_edge(int from, int to, int cap, int rev_cap, int type,
              int *d_left_weight, int *d_right_weight, int *d_down_weight,
              int *d_up_weight)
{
  if(type==1)
  {
    //d_left_weight[from] = d_left_weight[from]+cap;
    //d_right_weight[to] = d_right_weight[to]+rev_cap;
    atomicAdd(&d_left_weight[from], cap);
    atomicAdd(&d_right_weight[to], rev_cap);
    /*
      // ** NOTE: This function is inlined and called from multiple places, so
      // asm block is duplicated and code won't compile.  Make function not
      // inlined?
    int * leftWeightFromAddr = &(d_left_weight[from]);
    int * rightWeightToAddr = &(d_right_weight[to]);
    // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
    // same temp reg names
    asm volatile(// Temp Registers
                 ".reg .s32 ab0;\n\t"    // temp reg ab0 (cap)
                 ".reg .s32 ab1;\n\t"    // temp reg ab1 (rev_cap)
                 ".reg .s32 ab2;\n\t"    // temp reg ab2 (atomicAdd(leftWeightFromAddr) result)
                 ".reg .s32 ab3;\n\t"    // temp reg ab3 (atomicAdd(rightWeightToAddr) result)
                 // PTX Instructions
                 "mov.s32 ab0, %0;\n\t"
                 "mov.s32 ab1, %1;\n\t"
                 "atom.add.s32 ab2, [%1], ab0;\n\t" // atomicAdd for leftWeightFromAddr
                 "atom.add.s32 ab3, [%2], ab1;"     // atomicAdd for rightWeightToAddr
                 // no outputs
                 // inputs
                 :: "r"(cap), "r"(rev_cap), "l"(leftWeightFromAddr),
                    "l"(rightWeightToAddr)
                 );
    */
  }

  if(type==2)
  {
    //d_right_weight[from] = d_right_weight[from]+cap;
    //d_left_weight[to] = d_left_weight[to]+rev_cap;
    atomicAdd(&d_right_weight[from], cap);
    atomicAdd(&d_left_weight[to], rev_cap);
    /*
      // ** NOTE: This function is inlined and called from multiple places, so
      // asm block is duplicated and code won't compile.  Make function not
      // inlined?
    int * rightWeightFromAddr = &(d_right_weight[from]);
    int * lefttWeightToAddr = &(d_left_weight[to]);
    // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
    // same temp reg names
    asm volatile(// Temp Registers
                 ".reg .s32 ac0;\n\t"    // temp reg ac0 (cap)
                 ".reg .s32 ac1;\n\t"    // temp reg ac1 (rev_cap)
                 ".reg .s32 ac2;\n\t"    // temp reg ac2 (atomicAdd(rightWeightFromAddr) result)
                 ".reg .s32 ac3;\n\t"    // temp reg ac3 (atomicAdd(lefttWeightToAddr) result)
                 // PTX Instructions
                 "mov.s32 ac0, %0;\n\t"
                 "mov.s32 ac1, %1;\n\t"
                 "atom.add.s32 ac2, [%1], ac0;\n\t" // atomicAdd for rightWeightFromAddr
                 "atom.add.s32 ac3, [%2], ac1;"     // atomicAdd for lefttWeightToAddr
                 // no outputs
                 // inputs
                 :: "r"(cap), "r"(rev_cap), "l"(rightWeightFromAddr),
                    "l"(lefttWeightToAddr)
                 );
    */
  }

  if(type==3)
  {
    //d_down_weight[from] = d_down_weight[from]+cap;
    //d_up_weight[to] = d_up_weight[to]+rev_cap;
    atomicAdd(&d_down_weight[from], cap);
    atomicAdd(&d_up_weight[to], rev_cap);
    /*
      // ** NOTE: This function is inlined and called from multiple places, so
      // asm block is duplicated and code won't compile.  Make function not
      // inlined?
    int * downWeightFromAddr = &(d_down_weight[from]);
    int * upWeightToAddr = &(d_up_weight[to]);
    // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
    // same temp reg names
    asm volatile(// Temp Registers
                 ".reg .s32 ad0;\n\t"    // temp reg ad0 (cap)
                 ".reg .s32 ad1;\n\t"    // temp reg ad1 (rev_cap)
                 ".reg .s32 ad2;\n\t"    // temp reg ad2 (atomicAdd(downWeightFromAddr) result)
                 ".reg .s32 ad3;\n\t"    // temp reg ad3 (atomicAdd(upWeightToAddr) result)
                 // PTX Instructions
                 "mov.s32 ad0, %0;\n\t"
                 "mov.s32 ad1, %1;\n\t"
                 "atom.add.s32 ad2, [%1], ad0;\n\t" // atomicAdd for downWeightFromAddr
                 "atom.add.s32 ad3, [%2], ad1;"     // atomicAdd for upWeightToAddr
                 // no outputs
                 // inputs
                 :: "r"(cap), "r"(rev_cap), "l"(downWeightFromAddr),
                    "l"(upWeightToAddr)
                 );
    */
  }

  if(type==4)
  {
    //d_up_weight[from] = d_up_weight[from]+cap;
    //d_down_weight[to] = d_down_weight[to]+cap;
    atomicAdd(&d_up_weight[from], cap);
    atomicAdd(&d_down_weight[to], cap);
    /*
      // ** NOTE: This function is inlined and called from multiple places, so
      // asm block is duplicated and code won't compile.  Make function not
      // inlined?
    int * upWeightFromAddr = &(d_up_weight[from]);
    int * downWeightToAddr = &(d_down_weight[to]);
    // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
    // same temp reg names
    asm volatile(// Temp Registers
                 ".reg .s32 ae0;\n\t"    // temp reg ae0 (cap)
                 ".reg .s32 ae1;\n\t"    // temp reg ae1 (rev_cap)
                 ".reg .s32 ae2;\n\t"    // temp reg ae2 (atomicAdd(upWeightFromAddr) result)
                 ".reg .s32 ae3;\n\t"    // temp reg ae3 (atomicAdd(downWeightToAddr) result)
                 // PTX Instructions
                 "mov.s32 ae0, %0;\n\t"
                 "mov.s32 ae1, %1;\n\t"
                 "atom.add.s32 ae2, [%1], ae0;\n\t" // atomicAdd for upWeightFromAddr
                 "atom.add.s32 ae3, [%2], ae1;"     // atomicAdd for downWeightToAddr
                 // no outputs
                 // inputs
                 :: "r"(cap), "r"(rev_cap), "l"(upWeightFromAddr),
                    "l"(downWeightToAddr)
                 );
    */
  }
}

__device__
void add_tweights(int i, int cap_source, int  cap_sink, int *d_push_reser,
                  int *d_sink_weight)
{
  int diff = cap_source - cap_sink;

  if(diff>0)
  {
    //d_push_reser[i] = d_push_reser[i] + diff;
    atomicAdd(&d_push_reser[i], diff);
  }
  else
  {
    d_sink_weight[i] = d_sink_weight[i] - diff;
  }
}

__device__
void add_term1(int i, int A, int B, int *d_push_reser, int *d_sink_weight)
{
  add_tweights(i,B,A, d_push_reser, d_sink_weight);
}

__device__
void add_t_links_Cue(int alpha_label, int thid, int *d_left_weight,
                     int *d_right_weight, int *d_down_weight, int *d_up_weight,
                     int *d_push_reser, int *d_sink_weight, int *dPixelLabel,
                     int *dDataTerm, int width, int height, int num_labels)
{
  {
    if(dPixelLabel[thid]!=alpha_label) {
      add_term1(thid, dDataTerm[thid*num_labels+alpha_label], dDataTerm[thid * num_labels + dPixelLabel[thid]], d_push_reser, d_sink_weight  );
    }
  }
}

__device__
void add_t_links(int alpha_label, int thid, int *d_left_weight,
                 int *d_right_weight, int *d_down_weight, int *d_up_weight,
                 int *d_push_reser, int *d_sink_weight, int *dPixelLabel,
                 int *dDataTerm, int width, int height, int num_labels)
{
  {
    if(dPixelLabel[thid]!=alpha_label) {
      add_term1(thid, dDataTerm[thid*num_labels+alpha_label], dDataTerm[thid * num_labels + dPixelLabel[thid]], d_push_reser, d_sink_weight  );
    }
  }
}

__device__
void add_term2(int x, int y, int A, int B, int C, int D, int type,
               int *d_left_weight, int *d_right_weight, int *d_down_weight,
               int *d_up_weight, int *d_push_reser, int *d_sink_weight  )
{
  if ( A+D > C+B) {
    int delta = A+D-C-B;
    int subtrA = delta/3;

    A = A-subtrA;
    C = C+subtrA;
    B = B+(delta-subtrA*2);
#ifdef COUNT_TRUNCATIONS
    truncCnt++;
#endif
  }
#ifdef COUNT_TRUNCATIONS
  totalCnt++;
#endif

  add_tweights(x, D, A, d_push_reser, d_sink_weight);

  B -= A; C -= D;

  if (B < 0)
  {
    add_tweights(x, 0, B, d_push_reser, d_sink_weight);
    add_tweights(y, 0, -B, d_push_reser, d_sink_weight );
    add_edge(x, y, 0, B+C,type, d_left_weight, d_right_weight, d_down_weight, d_up_weight );
  }
  else if (C < 0)
  {
    add_tweights(x, 0, -C, d_push_reser, d_sink_weight);
    add_tweights(y, 0, C, d_push_reser, d_sink_weight);
    add_edge(x, y, B+C, 0,type, d_left_weight, d_right_weight, d_down_weight, d_up_weight);
  }
  else
  {
    add_edge(x, y, B, C,type, d_left_weight, d_right_weight, d_down_weight, d_up_weight);
  }
}

__device__
void set_up_expansion_energy_G_ARRAY(int alpha_label, int thid, int *d_left_weight,
                                     int *d_right_weight, int *d_down_weight,
                                     int *d_up_weight, int *d_push_reser,
                                     int *d_sink_weight, int *dPixelLabel,
                                     int *dDataTerm, int *dSmoothTerm, int width,
                                     int height, int num_labels )
{
  int x,y,nPix;

  int weight;

  int i = thid;
  {
    if(dPixelLabel[i]!=alpha_label)
    {
      y = i/width;
      x = i - y*width;

      if ( x < width - 1 )
      {
        nPix = i + 1;
        weight = 1;
        if ( dPixelLabel[nPix] != alpha_label )
        {
          add_term2(i,nPix,
                    ( dSmoothTerm[alpha_label + alpha_label * num_labels]) * weight,
                    ( dSmoothTerm[alpha_label + dPixelLabel[nPix]*num_labels]) * weight,
                    ( dSmoothTerm[ dPixelLabel[i] +  alpha_label * num_labels] ) * weight,
                    ( dSmoothTerm[ dPixelLabel[i] +  dPixelLabel[nPix] * num_labels] )  * weight,
                    2, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight); // 1-left, 2-right, 3-down, 4-up
        }
        else   add_term1(i,
                         ( dSmoothTerm[alpha_label + dPixelLabel[nPix] * num_labels]) * weight,
                         ( dSmoothTerm[dPixelLabel[i] + alpha_label*num_labels]) * weight,
                         d_push_reser, d_sink_weight);
      }

      if ( y < height - 1 )
      {
        nPix = i + width;
        weight = 1;
        if ( dPixelLabel[nPix] != alpha_label )
        {
          add_term2(i,nPix,
                    ( dSmoothTerm[alpha_label + alpha_label * num_labels]) * weight,
                    ( dSmoothTerm[alpha_label + dPixelLabel[nPix]*num_labels]) * weight,
                    ( dSmoothTerm[ dPixelLabel[i] +  alpha_label * num_labels] ) * weight,
                    ( dSmoothTerm[ dPixelLabel[i] +  dPixelLabel[nPix] * num_labels] )  * weight,
                    3, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight );
        }
        else   add_term1(i,
                         ( dSmoothTerm[alpha_label + dPixelLabel[nPix] * num_labels]) * weight,
                         ( dSmoothTerm[dPixelLabel[i] + alpha_label*num_labels]) * weight,
                         d_push_reser, d_sink_weight);
      }
      if ( x > 0 )
      {
        nPix = i - 1;
        weight = 1;
        if ( dPixelLabel[nPix] == alpha_label )
          add_term1(i,
                    ( dSmoothTerm[alpha_label + dPixelLabel[nPix] * num_labels]) * weight,
                    ( dSmoothTerm[dPixelLabel[i] + alpha_label*num_labels]) * weight,
                    d_push_reser, d_sink_weight );
      }

      if ( y > 0 )
      {
        nPix = i - width;
        weight = 1;
        if ( dPixelLabel[nPix] == alpha_label )
        {
          add_term1(i,
                    ( dSmoothTerm[alpha_label + alpha_label * num_labels]) * weight,
                    ( dSmoothTerm[dPixelLabel[i] + alpha_label*num_labels]) * weight,
                    d_push_reser, d_sink_weight);
        }
      }
    }
  }
}

__device__
void set_up_expansion_energy_G_ARRAY_Cue(int alpha_label, int thid,
                                         int *d_left_weight,
                                         int *d_right_weight,
                                         int *d_down_weight,
                                         int *d_up_weight, int *d_push_reser,
                                         int *d_sink_weight, int *dPixelLabel,
                                         int *dDataTerm, int *dSmoothTerm,
                                         int *dHcue, int *dVcue, int width,
                                         int height, int num_labels )
{
  int x,y,nPix;

  int weight;

  int i = thid;
  {
    if(dPixelLabel[i]!=alpha_label)
    {
      y = i/width;
      x = i - y*width;

      if ( x < width - 1 )
      {
        nPix = i + 1;
        weight=dHcue[i];
        if ( dPixelLabel[nPix] != alpha_label )
        {
          add_term2(i,nPix,
                    ( dSmoothTerm[alpha_label + alpha_label * num_labels]) * weight,
                    ( dSmoothTerm[alpha_label + dPixelLabel[nPix]*num_labels]) * weight,
                    ( dSmoothTerm[ dPixelLabel[i] +  alpha_label * num_labels] ) * weight,
                    ( dSmoothTerm[ dPixelLabel[i] +  dPixelLabel[nPix] * num_labels] )  * weight,
                    2, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight); // 1-left, 2-right, 3-down, 4-up
        }
        else   add_term1(i,
                         ( dSmoothTerm[alpha_label + dPixelLabel[nPix] * num_labels]) * weight,
                         ( dSmoothTerm[dPixelLabel[i] + alpha_label*num_labels]) * weight,
                         d_push_reser, d_sink_weight);
      }


      if ( y < height - 1 )
      {
        nPix = i + width;
        weight=dVcue[i];
        if ( dPixelLabel[nPix] != alpha_label )
        {
          add_term2(i,nPix,
                    ( dSmoothTerm[alpha_label + alpha_label * num_labels]) * weight,
                    ( dSmoothTerm[alpha_label + dPixelLabel[nPix]*num_labels]) * weight,
                    ( dSmoothTerm[ dPixelLabel[i] +  alpha_label * num_labels] ) * weight,
                    ( dSmoothTerm[ dPixelLabel[i] +  dPixelLabel[nPix] * num_labels] )  * weight,
                    3, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight );
        }
        else   add_term1(i,
                         ( dSmoothTerm[alpha_label + dPixelLabel[nPix] * num_labels]) * weight,
                         ( dSmoothTerm[dPixelLabel[i] + alpha_label*num_labels]) * weight,
                         d_push_reser, d_sink_weight);
      }
      if ( x > 0 )
      {
        nPix = i - 1;
        weight=dHcue[nPix];
        if ( dPixelLabel[nPix] == alpha_label )
          add_term1(i,
                    ( dSmoothTerm[alpha_label + dPixelLabel[nPix] * num_labels]) * weight,
                    ( dSmoothTerm[dPixelLabel[i] + alpha_label*num_labels]) * weight,
                    d_push_reser, d_sink_weight );
      }

      if ( y > 0 )
      {
        nPix = i - width;
        weight = dVcue[nPix];
        if ( dPixelLabel[nPix] == alpha_label )
        {
          add_term1(i,
                    ( dSmoothTerm[alpha_label + alpha_label * num_labels]) * weight,
                    ( dSmoothTerm[dPixelLabel[i] + alpha_label*num_labels]) * weight,
                    d_push_reser, d_sink_weight);
        }
      }
    }
  }
}

__global__
void CudaWeightCue(int alpha_label, int *d_left_weight, int *d_right_weight,
                   int *d_down_weight, int *d_up_weight, int *d_push_reser,
                   int *d_sink_weight, int *dPixelLabel, int *dDataTerm,
                   int *dSmoothTerm, int *dHcue, int *dVcue, int width,
                   int height, int num_labels)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int thid = blockIdx.x * 256 + threadIdx.x;

  add_t_links_Cue(alpha_label, thid, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight, dPixelLabel, dDataTerm, width, height, num_labels);

  set_up_expansion_energy_G_ARRAY_Cue(alpha_label, thid, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight, dPixelLabel, dDataTerm, dSmoothTerm, dHcue, dVcue, width, height, num_labels);
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

__global__
void CudaWeight(int alpha_label, int *d_left_weight, int *d_right_weight,
                int *d_down_weight, int *d_up_weight, int *d_push_reser,
                int *d_sink_weight, int *dPixelLabel, int *dDataTerm,
                int *dSmoothTerm, int width, int height, int num_labels)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int thid = blockIdx.x * 256 + threadIdx.x;

  add_t_links(alpha_label, thid, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight, dPixelLabel, dDataTerm, width, height, num_labels);

  set_up_expansion_energy_G_ARRAY(alpha_label, thid, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight, dPixelLabel, dDataTerm, dSmoothTerm, width, height, num_labels);
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

/*********************************************************
 * function which adjusts the array size for efficiency **
 * consideration                                        **
 * ******************************************************/
__global__
void adjustedgeweight(int *d_left_weight, int *d_right_weight,
                      int *d_down_weight, int *d_up_weight, int *d_push_reser,
                      int *d_sink_weight, int *temp_left_weight,
                      int *temp_right_weight, int *temp_down_weight,
                      int *temp_up_weight, int *temp_push_reser,
                      int *temp_sink_weight, int width, int height,
                      int graph_size, int width1, int height1, int graph_size1)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int thid = blockIdx.x * 256 + threadIdx.x;

  if( thid < graph_size1 )
  {
    int row = thid / width1, col = thid % width1;
    if(row < height && col < width)
    {
      temp_left_weight[row* width1 + col] = atomicAdd(&d_left_weight[row * width + col], 0);
      temp_right_weight[row * width1 + col] = atomicAdd(&d_right_weight[row * width + col], 0);
      temp_down_weight[row * width1 + col] = atomicAdd(&d_down_weight[row * width + col], 0);
      temp_up_weight[row * width1 + col] = atomicAdd(&d_up_weight[row * width + col], 0);
      temp_push_reser[row * width1 + col] = atomicAdd(&d_push_reser[row * width + col], 0);
      temp_sink_weight[row * width1 + col] = d_sink_weight[row * width + col];
    }
    else
    {
      temp_left_weight[row * width1 + col] = 0;
      temp_right_weight[row * width1 + col] = 0;
      temp_down_weight[row * width1 + col] = 0;
      temp_up_weight[row * width1 + col] = 0;
      temp_push_reser[row * width1 + col] = 0;
      temp_sink_weight[row * width1 + col] = 0;
    }
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
  */
}

/************************************************************
 * Intializes memory on the gpu                            **
 * ********************************************************/
__global__
void copyedgeweight( int *d_left_weight, int *d_right_weight,
                     int *d_down_weight, int *d_up_weight, int *d_push_reser,
                     int *d_sink_weight, int *temp_left_weight,
                     int *temp_right_weight, int *temp_down_weight,
                     int *temp_up_weight, int *temp_push_reser,
                     int *temp_sink_weight, int *d_pull_left, int *d_pull_right,
                     int *d_pull_down, int *d_pull_up, int *d_relabel_mask,
                     int *d_graph_heightr, int *d_graph_heightw, int width,
                     int height, int graph_size, int width1, int height1,
                     int graph_size1)
{
  /*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(RELAX_ATOM_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
  }
  __syncthreads();
*/
  int thid = blockIdx.x * 256 + threadIdx.x;

  if( thid < graph_size1 )
  {
    /*
    d_left_weight[thid] = temp_left_weight[thid];
    d_right_weight[thid] = temp_right_weight[thid];
    d_down_weight[thid] = temp_down_weight[thid];
    d_up_weight[thid] = temp_up_weight[thid];
    d_push_reser[thid] = temp_push_reser[thid];
    */
    atomicExch(&d_left_weight[thid], temp_left_weight[thid]);
    atomicExch(&d_right_weight[thid], temp_right_weight[thid]);
    atomicExch(&d_down_weight[thid], temp_down_weight[thid]);
    atomicExch(&d_up_weight[thid], temp_up_weight[thid]);
    atomicExch(&d_push_reser[thid], temp_push_reser[thid]);
    d_sink_weight[thid] = temp_sink_weight[thid];

    atomicExch(&d_pull_left[thid], 0);
    atomicExch(&d_pull_right[thid], 0);
    atomicExch(&d_pull_down[thid], 0);
    atomicExch(&d_pull_up[thid], 0);
    d_relabel_mask[thid] = 0;
    d_graph_heightr[thid] = 1;
    d_graph_heightw[thid] = 1;
  }
/*
  if (threadIdx.x == 0) {
    // all of the arrays accessed in this kernel are either in the read-only region,
    // special region or the relaxed atomic region
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(RELAX_ATOM_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
  }
 */ 
}

#endif
