"""Numerical example in our paper: benchmark against the series representation
"""

# add working directory to PYTHONPATH as this is not supported by Slurm
import sys
from pathlib import Path
root_path = Path.cwd().parent.parent
sys.path.append(str(root_path))


from rpylib.grid.spatial import CTMCUniformGrid

from scripts.mlmc.tools.utils import *
from scripts.benchmark.series_representation import benchmark_series_representation


def generate_data_series_representation():
    my_strikes = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05])
    my_mc_paths = 200_000
    my_tau = 4_000

    def my_ctmc_grid_function(model):
        return CTMCUniformGrid(h=1e-6, model=model)

    my_result_folder_name = Path(Path().cwd().parent, 'results/series/')

    benchmark_series_representation(strikes=my_strikes, mc_paths=my_mc_paths, tau=my_tau,
                                    ctmc_grid_function=my_ctmc_grid_function, result_folder_name=my_result_folder_name)


if __name__ == '__main__':
    generate_data_series_representation()
