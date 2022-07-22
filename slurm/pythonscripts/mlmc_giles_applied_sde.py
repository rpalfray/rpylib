# add working directory to PYTHONPATH as this is not supported by Slurm
import sys
from pathlib import Path
root_path = Path.cwd().parent.parent
sys.path.append(str(root_path))

import argparse

from scripts.mlmc.application.mclm_applied_to_sde import coupling_sde_cost_and_levels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for weak/strong convergence graphs as in Giles for'
                                                 'Lévy-driven SDE')
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-rmses', nargs='*', default=None, type=float, required=True)
    args = parser.parse_args()

    coupling_sde_cost_and_levels(name=args.model, rmses=args.rmses)
