#!/bin/bash
#PBS -N atep_run
#PBS -l select=11:ncpus=24:mpiprocs=1
#PBS -P xxx1111
#PBS -q xxxx
#PBS -l walltime=96:00:00
#PBS -m abe
#PBS -M xxxx@yyyy.com


ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID

cd $PBS_O_WORKDIR

echo "$PBS_O_WORKDIR/$PBS_JOBID"

jobnodes=`uniq -c ${PBS_NODEFILE} | awk -F. '{print $1 }' | awk '{print $2}' | paste -s -d " "`

thishost=`uname -n | awk -F. '{print $1.}'`
thishostip=`hostname -i`
rayport=6379
 
thishostNport="${thishostip}:${rayport}"
echo "Allocate Nodes = <$jobnodes>"
export thishostNport
echo "HERE is pbs nodelist"
cat $PBS_NODEFILE 
echo "set up ray cluster..." 
J=0
for n in `echo ${jobnodes}`
do
        if [[ ${n} == "${thishost}" ]]
        then
                echo "first allocate node - use as headnode ..."
                module load chpc/python/anaconda/3-2019.10
                source /apps/chpc/chem/anaconda3-2019.10/etc/profile.d/conda.sh
                conda activate atep-env
		ray start --head
		echo "bash::started ray on headnode"
                sleep 5
        else
		echo "Running now on other node ${n} -- ${thishostNport} -- $J"
		pbsdsh -n $J -- $PBS_O_WORKDIR/startWorkerNode.pbs ${thishostNport} & 
                sleep 5
        fi
J=$((J+1))
done 

experiment=experiment_name

mkdir -p /ipp/$experiment
mkdir -p /logs/$experiment

python -u master.py \
  /logs/$experiment \
  --delta_threshold=3.0 \
  --neat_population=1000 \
  --c1=1.0 \
  --c2=1.0 \
  --c3=3.7 \
  --max_stagnation=60 \
  --crossover_probability=0.3 \
  --connection_weight_probability=0.95 \
  --mutation_probability_node=0.15 \
  --weight_mutate_large_probability=0.85 \
  --bias_mutation_large_probability=0.85 \
  --master_seed=24582922 \
  --mc_lower=25 \
  --mc_upper=340 \
  --repro_threshold=200 \
  --max_num_envs=20 \
  --adjust_interval=6 \
  --steps_before_transfer=25 \
  --num_workers=24 \
  --checkpointing \
  --start_from=None \
  --start_from_checkpointing=False \
  --n_iterations=20000 2>&1 | tee /ipp/$experiment/run.log



