# Shreya Kapoor's Master Thesis
## Extracting Most Predictive Subgraphs from Models of Human Brain Connectivity
--------------------------------------------------------------------------------------
Thesis submission date: 16 November 2020
---------------------------------------------------------------------------------------
Directory structure:
a) preprocessing
	- Conversion of the HCP file data to connectivity matrices in .csv format
	- dMRI files from s900 processed using the pipeline given on https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/ismrm_hcp_tutorial.html?highlight=HCP#ismrm-tutorial-structural-connectome-for-human-connectome-project-hcp
	- Files needed from the HCP repository: 
		- bvals
		- bvecs
		- data.nii.gz
		- nodif_brain_mask.nii.gz
		- aparc+aseg.nii.gz
		- T1w_acpc_dc_restore_brain.nii.gz
	- Main file for running the scripts: pipeline.py
	- fsl_5.0 required
	
b) graphtools
	Contains all the experiments done after the preprocessing
	- Performs classification according to prepared matrix 
c) gmwcs-solver
	- Modified from the provided version
