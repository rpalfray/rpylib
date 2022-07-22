#!/bin/bash

#SBATCH -o ./../../output/out_CQ1d-%j.out # STDOUT

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=00:05:00

. /etc/profile
module purge
module load GCCcore/11.2.0 Python/3.9.6

export PYTHONPATH="${PYTHONPATH}$HOME/code/rpylib/"

cd $HOME/code/rpylib/slurm/pythonscripts || exit

srun python mlmc_giles_convergence_1d.py -model="$1" -max_level=2 -mc_paths=5

## for example:
## sbatch -J CQgmy15 convergence_1d_quick.sbatch cgmy15
