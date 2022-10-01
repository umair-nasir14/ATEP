# ATEP
This repository contains code to reproduce all of our experiments.
The file structure is as follows:
```
├── atep                    -> The main ATEP code
├── dataframes              -> Contains csv files to reproduce the figures
├── demos                   -> Demo videos
├── figures                 -> The actual figures
├── jobscript.sh            -> Main jobscript to reproduce our experiments
├── neat                    -> Main code for neat
```
To create the environment, use `environment.yml`
```
conda env create -f environment.yml
```
To reproduce the experiments, run 
```
qsub jobscript.sh
```

To visualise and recreate the plots, use `visuals.ipynb`.
