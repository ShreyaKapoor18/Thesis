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

## Usage

### Running Preprocessing Pipeline

```bash
cd preprocessing
python pipeline.py --input-dir /path/to/hcp/data --output-dir /path/to/output
```

### Running Graph Analysis & Classification

```bash
cd graphtools
python main_classifier.py --data-path /path/to/preprocessed/graphs
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{kapoor2020thesis,
  author={Kapoor, Shreya},
  title={Extracting Most Predictive Subgraphs from Models of Human Brain Connectivity},
  school={Friedrich-Alexander-Universität Erlangen-Nürnberg},
  year={2020}
}
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## References

- [MRtrix Documentation](https://mrtrix.readthedocs.io/)
- [FSL - FMRIB Software Library](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
- [Nilearn - Neuroimaging Analysis](https://nilearn.github.io/)
- [Human Connectome Project](https://www.humanconnectomeproject.org/)

---

## Contact

**Author:** Shreya Kapoor  
**Email:** shreya.kapoor@fau.de  
**GitHub:** [@ShreyaKapoor18](https://github.com/ShreyaKapoor18)

---

## Acknowledgments

This thesis was conducted at Friedrich-Alexander-Universität Erlangen-Nürnberg. Special thanks to the Human Connectome Project for providing the dMRI data and to the open-source community for the tools and libraries used in this research.
