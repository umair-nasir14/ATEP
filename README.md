# ATEP
This repository contains code for [Augmentative Topology Agents For Open-Ended Learning](https://dl.acm.org/doi/abs/10.1145/3583133.3590576).
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

If you use the repository, please cite:
```
@inproceedings{nasir2023augmentative,
  title={Augmentative topology agents for open-ended learning},
  author={Nasir, Muhammad and Beukman, Michael and James, Steven and Cleghorn, Christopher},
  booktitle={Proceedings of the Companion Conference on Genetic and Evolutionary Computation},
  pages={671--674},
  year={2023}
}
```
