This repository contain the code that was used in a research at Tel Aviv university, in collaboration with Tel Aviv Sourasky medical center. The main objective was to cluster Covid-19 patients, using echocardiography in addition to standard clinical measurement. The clustering was done with consensus clustering using the k-prototype algorithm.
The data that the project is based on is not publicly available and may be made available upon reasonable request to Dr. Yan Topolsky (yant@tlvmc.gov.il).  

This repository contains the following python files: 
make_full_data_frames.py – creates and saves dataframes that are ready to be used as input to the clustering algorithm, after imputation and normalization. Receives four arguments:  
1.	A csv file of the full data
2.	A list of all continuous variables
3.	A list of all categorical variables
4.	Path to save the data frames that are ready to be used as input

cluster_and_analyses.py – cluster the data using k-prototype algorithm and perform several analyses: compute p-values for all variables to test if they vary significantly among clusters, compute c-index and log rank p-value for cox model of patients mortality, analyze the change in c-index and log rank p-value when the effect of the echo variables is discarded, or equivalent number of non-echo variables. Receives four arguments: 
1.	A csv file of the full data
2.	A csv file of the imputed continuous data
3.	A csv file of the imputed categorical data
4.	List of outcomes variables to add to the analysis (not as input for clustering). 

consensus_with_kprototype.py - runs the k-prototype algorithm in a consensus clustering frame work. This file is originally found in github.com/ZigaSajovic/Consensus_Clustering under the name:consensusClustering.py and the copyrights are saved to Žiga Sajovic, XLAB 2019. 
Changes were made by us to adjust the consensuse framework to k-prototypes algorithm.  
