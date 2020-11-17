
#######################################################################################
#                                                                      		      #			
#         Software implements Graph Cuts on CUDA                      		      #	
#								      	 	      #	
#								      		      #	
#   										      #	
#										      #	
#   Copyright (c) 2008 International Institute of Information Technology.	      #	
#   All rights reserved.							      #	
#										      #	
#   Permission to use, copy, modify and distribute this software and its	      #	
#   documentation for  research purpose is hereby granted without fee, 	              #	
#   provided that the  above copyright notice and this permission notice 	      #	
#   appear in all copies of this software  and that you do  not sell the 	      #	
#   software.									      #	
#										      #	
#   THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, 		      #	
#   EXPRESS, IMPLIED OR  OTHERWISE.					      	      #	
# 										      #	
#   Please report any issue to Vibhav Vineet ( vibhavvinet@research.iiit.ac.in)	      #	
#    										      #
#   Please cite following papers, if you use this software for research		      # 	
#   purpose                                                                           #
#	                                                                              #
#   "CUDA Cuts: Fast Graph Cuts on the GPU"                                           #
#   Vibhav Vineet and P. J. Narayanan                                                 #
#   Proceedings of CVPR workshop on Visual Computer Visions on the GPUs, June         #
#   2008		                                                              #
#	                                                                              #
#   "CUDA Cuts: Fast Graph Cuts on the GPU"                                           #
#   Vibhav Vineet and P. J. Narayanan                                                 #
#   Technical Report, International Institute of Information Technology,              #
#   Hyderabd	                                                                      #
#		                                                                      #
######################################################################################

=======================================================================================================================                                                                            
1. Using the Code in the Linux Environment
   1. Download and unzip CudaCuts.zip from the website.
      It contains CudaCuts directory. 
      Subdirectories:
	  CudaCutsAtomic/ -- This include the programs which require hardware with atomic compatibility. 
	  CudaCutsNonAtomic/ -- This include the programs which does not require hardware with atomic compatibility.
  
  2. Make sure you have CUDA installed on your system.
  4. Include CudaCuts.cu from any of the subdirectories into your main program. 
     For more information look into example code.
  5. make to compile the code. 
 
=======================================================================================================================

2. Example Code

  Example code is given as a help to understand how to use the code.
  
  1. "Example.cu" is the main program. 
  2. make to compile the example code.
  3. Exucutable file "cudaCuts" is created.

=======================================================================================================================

3. Here is the brief description of how to use the code. 

  Our code will solve the MRF problems involving 4-connectivity for a grid graph.

  The default type for energy and edge-weights is int. 

step 1 : Intialize the Grid

	Width and height of the grid is set in this function. Number of 
	labels is also set through this function call:
		 
	1. void cudaCutsInit(int width, int height, int numOfLabels) ; 

step 2 : Set up the Energy Function

	Energy functions are specified by dataTerm and smoothnessTerm. As in our
	case, when the connectivity is 4, special weights, hCue and vCue can also be
	specified. dataTerm is stored in an array "dataCost" of size width * height *numOfLabels. 
	smoothnessTerm is stored in an array "smoothnessCost" of size numOfLabels * numOfLabels.
	hCue and vCue are stored in arrays of sizes width * height in "hcue" and "vcue"
	arrays respectively. 

	Users can pass the pointers to the program for dataTerm, smoothnessTerm, hCue, and 
	vCue through a series of function calls. These functions set the energy functions
	and returns 0 on success otherwise returns a -1 as an indication of failure.

	1. int cudaCutsSetupDataTerm( energyType *dCost ):
	2. int cudaCutsSetupSmoothTerm( energyType *smoothCost);
	3. int cudaCutsSetupHCue( energyType *hCue );
	4. int cudaCutsSetupVCue( energyType *vCue );

step 3 : Constructing the Graph and Invoking the CudaCuts Algorithm
	
	The graph is constructed after specifying the energy function as explained
	earlier. The "cudaCutsSetupGraph()" constructs the graph and "cudaCutsOptimize()" calls the
	cudacuts optimization algorithm. Pixels get their labels through the call "cudaCutsGetResult()". 
	The function "cudaCutsGetResult()" updates a global array "pixelLabels", of
	size width x height, which has the binary label assigned to each pixel. cudaCutsGetEnergy()
	returns the total energy of the current configuration of the MRF. 

	1. int cudaCutsSetupGraph();
	2. int cudaCutsGcOptimize();
	3. int cudaCutsGetResult();
	4. energyType cudaCutsGetEnergy() ;

step 4 : Free Allocated Memory 
	
	cudaCutsFreeMem() function deletes the allocated memory on both the host and the device. 
	

=======================================================================================================================

4. Overall Cuda Cuts Algorithm
	
	1.  Include CudaCuts.cu to your main program. 
	
	Invoke following series of function calls:

	2.  cudaCutsInit( width,  height , numOfLabels);
	3.  cudaCutsSetupDataTerm( dataTerm ) ;
	4.  cudaCutsSetupSmoothTerm( smoothTerm ) ; 
	5.  cudaCutsSetupHCue( hCue ) ; 
	6.  cudaCutsSetupVCue( vCue ) ; 
	7.  cudaCutsSetupGraph() ; 
	8.  cudaCutsGcOptimize() ; 
	9.  cudaCutsGetResult() ; 
	10  cudaCutsGetEnergy();
	11. cudaCutsFreeMem() ; 

 	The interface gives the result for one image. However, if the segmentation
	operation has to be performed on a series of frames or on a video, the steps 7 to
	9 should be called for each frame and steps 3 to 6, which ever is changing frame to frame, 
	considering the widht and height of the grid remains same.
	
=======================================================================================================================
                                                                                                     
 5.  Using the Code in Windows Environment                                                           
                                                                                                     
    The step wise process required for using the code on windows remains same                        
    as in linux environment. One example code is given which can help the users                    
    working in windows environment.                                                                 
=======================================================================================================================
