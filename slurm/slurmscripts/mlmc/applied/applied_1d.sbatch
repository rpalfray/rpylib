#!/bin/bash

#SBATCH -o ./../../output/out_A1d-%j.out # STDOUT

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00

##SBATCH --partition=hmem

. /etc/profile
module purge
module load GCCcore/11.2.0 Python/3.9.6

export PYTHONPATH="${PYTHONPATH}$HOME/code/rpylib/"

cd "$HOME"/code/rpylib/slurm/pythonscripts || exit

python mlmc_giles_applied_1d.py -model="$1" -rmses="$2"

## for example:
## sbatch -J A1d applied_1d.sbatch -model=cgmy15 -rmses=0.04

##  R0  R1    R2     R3   R4    R5
## 0.1 0.07 0.05 0.04 0.03 0.02
