# Shreya Kapoor's Master Thesis

## Extracting Most Predictive Subgraphs from Models of Human Brain Connectivity

**Thesis Submission Date:** November 16, 2020

---

## Overview

This repository contains the research code, experiments, and documentation for a Master's thesis focused on extracting the most predictive subgraphs from models of human brain connectivity. The work combines dMRI preprocessing, graph theory, and machine learning to identify significant brain network patterns.

## Repository Structure

### 📁 **preprocessing/**
Handles dMRI data processing using the [MRtrix pipeline](https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/ismrm_hcp_tutorial.html)

**Required Dependencies:**
- FSL 5.0
- MRtrix

**Required HCP Data Files:**
- `bvals` / `bvecs` - Diffusion gradient information
- `data.nii.gz` - Raw DWI data
- `nodif_brain_mask.nii.gz` - Brain mask
- `aparc+aseg.nii.gz` - Anatomical segmentation
- `T1w_acpc_dc_restore_brain.nii.gz` - Structural image

**Main Entry Point:** `pipeline.py`

### 📁 **graphtools/**
Comprehensive experiments conducted post-preprocessing, including:
- Graph-based classification
- Feature extraction from network matrices
- AutoML leaderboard with multiple model comparisons
- Detailed performance metrics and visualizations

### 📁 **gmwcs-solver/** & **gmwcs-solver2/**
Modified implementations of solvers for graph-based optimization tasks

### 📁 **CoBundleMAP/**
Specialized graph analysis submodule (external reference)

### 📁 **ms_thesis/**
Complete thesis document and supplementary materials

---

## Key Results

### AutoML Model Performance
The best performing model was **XGBoost** with:
- **Logloss:** 0.20092
- **Training Time:** 335.61 seconds

Other models tested:
- Baseline: 0.47933 logloss
- Decision Tree: 2.19603 logloss
- Neural Network: 0.419515 logloss
- Random Forest: 0.273811 logloss
- Ensemble: 0.20092 logloss

For detailed performance analysis, see `graphtools/fc_data/AutoML_1/README.md`

---

## Installation

### Requirements
- Python 3.7+
- FSL 5.0
- MRtrix3
- scikit-learn
- XGBoost
- nilearn
- networkx

### Setup

```bash
# Clone the repository
git clone https://github.com/ShreyaKapoor18/Thesis.git
cd Thesis

# Install Python dependencies
pip install -r requirements.txt
```



---
