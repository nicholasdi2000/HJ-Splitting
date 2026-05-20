# HJ-Prox

Implementation of the paper "Operator Splitting with Hamilton–Jacobi-based Proximals", ICML 2026.

The Hamilton–Jacobi proximal operator (HJ-Prox) estimates `prox_{t·f}(x)` for a possibly non-smooth `f` using only function values of `f`, via a Monte Carlo softmax average derived from the viscous Hamilton–Jacobi PDE. Plugging HJ-Prox into existing operator-splitting schemes (PGD, DRS, PDHG, DYS) replaces difficult analytical proximal calculations with a zeroth-order approximation.


## Notebooks

| Notebook | What it does |
|----------|--------------|
| `pgd_lasso.ipynb` | LASSO with PGD vs PGD-HJ vs PPM-HJ |
| `drs_fused_lasso.ipynb` | Fused LASSO on Doppler signal with DRS / DRS-HJ / PPM-HJ |
| `pdhg_total_variation.ipynb` | TV deblurring with PDHG / PDHG-HJ |
| `drs_multitask_learning.ipynb` | Multitask learning with DRS / DRS-HJ |
| `dys_sparse_group_lasso.ipynb` | Sparse-group LASSO with DYS / DYS-HJ |
| `dys_sparse_group_lasso_reviewer.ipynb` | Sparse-group LASSO, alternate δ annealing (Openreview Response) |
| `dys_constrained_lasso.ipynb` | Non-negative LASSO with DYS / DYS-HJ |
| `dys_real_world_application.ipynb` | Overlapping group LASSO on GSE2034 breast-cancer data |
| `foglasso_vs_hjprox.ipynb` | HJ-Prox vs the FoGLasso analytical dual solver (Openreview Response) |

Open one, **Run All**.

`hj_prox` depends only on **numpy** and **torch**. The notebooks additionally use **matplotlib**, **scipy**, **pandas**, **scikit-learn**, and (for the GSE2034 experiment) **GEOparse** and **gseapy**. The GSE2034 notebook downloads ~50 MB of microarray data from NCBI GEO on first run.


Disclaimer: Both ChatGPT and Claude aided in writing simulations, polishing, and organizing the git repo.
