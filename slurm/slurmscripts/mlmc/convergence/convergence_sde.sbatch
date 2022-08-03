#!/bin/bash

#SBATCH -o ./../../output/out_C1sde-%j.out # STDOUT

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=05:00:00

##SBATCH --partition=hmem

. /etc/profile
module purge
module load GCCcore/11.2.0 Python/3.9.6

export PYTHONPATH="${PYTHONPATH}$HOME/code/rpylib/"

cd "$HOME"/code/rpylib/slurm/pythonscripts || exit

srun python mlmc_giles_convergence_sde.py -model="$1"

## for example:
## sbatch -J Csde convergence_sde.sbatch cgmy02_cgmy04
