# Multi-task-TCGA-learning
Multi-task learning of TCGA data. This is the code implementing the NIPS woman in machine learning poster titled "Multi-task learning of tumor genomics with deep neural network", abstract for this poster can be found [here](https://drive.google.com/file/d/0Bzoozx2KZAPmbUM2WVBmcV9DRms/view?usp=sharing)


### Data Source: TCGA
last downloaded from bop: `/projects/sysbio/users/TCGA/PANCANATLAS/Data/` on Oct-23-2017 

#### Running Jupyter Notebook remotely:
1. In dlpc, run `jupyter notebook --no-browser`  
2. In laptop, run `ssh -NL 8800:localhost:8888 molly@eduroam-169-233-227-149.ucsc.edu`, this command has alias `dlpc-jupyter`
