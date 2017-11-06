# Multi-task-TCGA-learning
Multi-task learning of TCGA data. This is the code implementing the NIPS woman in machine learning poster titled "Multi-task learning of tumor genomics with deep neural network", abstract for this poster can be found [here](https://drive.google.com/file/d/0Bzoozx2KZAPmbUM2WVBmcV9DRms/view?usp=sharing)


### 1. Data Source: TCGA
last downloaded from bop: `/projects/sysbio/users/TCGA/PANCANATLAS/Data/` on Oct-23-2017 

### 2. Running Jupyter Notebook remotely:
- In dlpc, run `jupyter notebook --no-browser`  
- In laptop, run `ssh -NL 8800:localhost:8888 molly@eduroam-169-233-227-149.ucsc.edu`, this command has alias `dlpc-jupyter`
