# Multi-task-TCGA-learning
Multi-task learning of TCGA data.

### 1. Data Source: TCGA
last downloaded from bop: `/projects/sysbio/users/TCGA/PANCANATLAS/Data/` on Oct-23-2017 

### 2. Running Jupyter Notebook remotely:
- In dlpc, run `jupyter notebook --no-browser --port 8888`  
- In laptop, run `ssh -NL 8888:localhost:8888 molly@dlpc` to forward the port from dlpc to laptop
