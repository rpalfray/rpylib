# add working directory to PYTHONPATH as this is not supported by Slurm
import sys
from pathlib import Path
root_path = Path.cwd().parent.parent
sys.path.append(str(root_path))

import argparse

from scripts.mlmc.tools.utils import *
from scripts.mlmc.convergence.mlmc_convergence_1d import coupling_variances


def giles_variances_graphs_1d(model_str: str, max_level: int, mc_paths: int, h0: float):
    my_maturity = cases_data['maturity']
    model = calibrated_models[model_str]
    result_folder_name = Path(Path(__file__).cwd().parent, 'results/giles_convergence/1d')

    coupling_variances(name=model_str, model=model, mc_paths=mc_paths, max_level=max_level,
                       maturity=my_maturity, h0=h0, result_folder_name=result_folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for weak/strong convergence graphs as in Giles for '
                                                 'one-dimensional Levy processes')
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-h0', type=float, default=cases_data['h0'])
    parser.add_argument('-max_level', type=int, default=cases_data['max_level'])
    parser.add_argument('-mc_paths', type=int, default=cases_data['mc_paths'])
    args = parser.parse_args()

    giles_variances_graphs_1d(model_str=args.model, max_level=args.max_level, mc_paths=args.mc_paths, h0=args.h0)
