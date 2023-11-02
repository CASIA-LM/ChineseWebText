#!/bin/bash
#SBATCH --job-name=finetune  # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=1     # total number of tasks per node
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH -p ai_training
#SBTACH --exclusive
##SBATCH --nodelist=dx-ai-node7

#SBATCH -o slumnlog/%x.o%j
#SBATCH -e slumnlog/%x.e%j

echo slurm_job_node_list: $SLURM_JOB_NODELIST
echo slurm_ntasks_per_node: $SLURM_NTASKS_PER_NODE
##sh run_train_pretrain.sh
sh run.sh