#!/bin/bash

#SBATCH -o ./../../output/out_CQ3d-%j.out # STDOUT

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3700
#SBATCH --time=00:10:00

. /etc/profile
module purge
module load GCCcore/11.2.0 Python/3.9.6

export PYTHONPATH="${PYTHONPATH}$HOME/code/rpylib/"

cd "$HOME"/code/rpylib/slurm/pythonscripts || exit

srun python mlmc_giles_convergence_copulas.py -model="$1" -max_level=2 -mc_paths=5

## for example:
## sbatch -J CQ3d convergence_copulas_quick.sbatch hem_vg_cgmy12
