[![Documentation Status](https://readthedocs.org/projects/smolyak-qmc/badge/?version=latest)](https://smolyak-qmc.readthedocs.io/en/latest/?badge=latest)


Smolyak_qmc 

This project was implemented to approximates the integral of multidimensional functions using the Smolyak algorithm. 
For this the one dimensional Quadratures itself, not the differences are used. The user could choose between 5 one-dimensional quadratures. 
These are 1: Newton-Cotes,
	  2: Trapezoidal, 
	  3: Monte Carlo (nested), 
	  4: Monte Carlo (non-nested) and 
	  5:Quasi-Monte Carlo. 

Regarding the use of the project, the easiest  way is to  call the controller_smolyak method. 

The input arguments are:
	function_string: String of function, you want be approximate. The function is made callable by 
		    	 the use of the Sympy package. Hence, problems might occur for less common functions. 

	variables_string: A sting with variables used in the function to avoid confusion. 
		          The form the input is needed in is: "(x_1, x_2,..., x_n)"
	
	quadrature: Sting of the quadrature you want to use. The options are "Newton-Cotes", "Trapezoidal", 
		    "Monte Carlo (nested)", "Monte Carlo (non-nested)" and  "Quasi-Monte Carlo". 
	
	q: Degree of approximation. 

Also $\epsilon$-error can be calculated for the different one-dimensional quadratures. The problem is, that  the error  
approximation turned out to be relatively inexact. Hence, the q given back is relatively high.

For the implementation of the Quasi-Monte Carlo option we use the  qmcpy.Lattice, to get the one dimensional points. 

Also a View could be used to make it easier to enter the arguments. 

The files in this folder are structured in the following way. 

Folders: 
	Methodes: The python files with the implementation of the Smolyak algorithm with several supplementary methods. 	
		  One methods uses one, the other one three points for, if the one dimensional degree of approximation equals 1.

	Data: For the examples several calculations were used that took several hours. For reasons of convenience these calculations were made before uploading the 
	      files, so that the largest part of the compilation of the jupyter files is creating plots. The code used to calculate these files could be found in the
	      bottom part of every document. The code is commented out.  
	      (The filename extension .pkl is used to make clear, that they need to be opened using the !! read_pickel !! method of the !! pandas !! package.)

  docs: Folder containing the docummentation, which was made using the sphinx package. The most important file of this folder is index.html. It could be found in
        the relative directory: docs/_build/html. 
  
  
  examples: 
		  Grid_points: Displays the the structure of the points used for the approximation of the approximation.
		  
		  Proof_of_principle: First  a simple, two-dimensional, analytical function is approximated using all five quadratures. 
		  		      Then the error for the approximation of a constant function is plotted for several degrees of approximation and dimensions
		  		      to verify the correctness of the implementation of the algorithm. This error can be explained by the computational epsilon.	
		  		      After this an easy polynomial is approximated in different dimension, to get a feeling for the error of the non.deterministic 
		  		      quadratures in higher dimension.
		  		    
		  Stat_evaluation_for_prob_quad: For the non-deterministic quadratures the error and the runtime for the approximation of an analytical function and 
		  				 with respect to the degree of approximation are compared. 
		  				 
	  
		 Perpetuated_Coulomb_Potential: The example 18.3.8 in the book Tractability of Multivariate Problems Vol. 2 is implemented for different alpha. Because it is 
		 	      		       not trivial to calculate the result the integral, the result is approximated for bigger alpha. For small alpha, the result of 
		 	      		       the approximation is varying by several orders of magnitude, depending on how many points with big values are used for the calculation.
		 	      		       The approximation even sometimes is negative, even though the function itself is strictly positive. The methods for multidimensional 
		 	      		       integration implemented in the scipy package show similar problems. 
		 	      		  
		View: Short  file calling the implemented GUI of the project. Further more a list of functions implemented in the .py file is shown. Regarding the way how to use the 
		      functions and what they return descriptions could be found in the .py files. 
		  

Comment regarding the implementation: During the implementation of this project I learned a lot about the syntax and best practice with respect to programming using python. Hence, a lot 
 	 			      a lot of not quite elegant syntax is used in this project. An example is the use of the range function. Here the enumerate function would have been
 	 			      easier to use and to read in many situations. Also the structure of programming could be improved. I will not change these points due to a lack of time. 
				      One other point that should be changed, is the way how the evaluations are stored. The way used  so far tends to fill the cache quite fastly.
