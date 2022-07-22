# add working directory to PYTHONPATH as this is not supported by Slurm
import sys
from pathlib import Path
root_path = Path.cwd().parent.parent
sys.path.append(str(root_path))


from rpylib.grid.spatial import CTMCUniformGrid

from scripts.mlmc.tools.utils import *

from scripts.benchmark.cds import cds_spread
from scripts.benchmark.first_to_default import first_to_default_spread


def generate_benchmark_singe_name_cds():
    my_maturity = 1/2
    name = 'hem'
    model = exp_of_levy_models[name]
    mc_paths = 1_000_000
    cds_spreads_bps = [100, 150, 200, 250, 300, 400, 500]
    result_folder_name = Path(Path().cwd().parent, 'results/credit/')

    def my_ctmc_grid_function(my_h, my_levels_a):
        return CTMCUniformGrid(h=my_h, model=model)

    cds_spread(name=name, model=model, maturity=my_maturity, mc_paths=mc_paths, cds_spreads_bps=cds_spreads_bps,
               ctmc_grid_function=my_ctmc_grid_function, result_folder_name=result_folder_name)


def generate_benchmark_first_to_default_cds():
    my_maturity = 1/2
    copula_clayton = create_clayton_copula(theta=0.70, eta=0.30)
    model1 = create_exponential_of_levy_model(ModelType.HEM)(spot=50, intensity=5, sigma=0.05)
    model2 = create_exponential_of_levy_model(ModelType.HEM)(spot=100, intensity=10, sigma=0.05)
    model3 = create_exponential_of_levy_model(ModelType.HEM)(spot=150, intensity=20, sigma=0.05)
    my_models = [model1, model2, model3]

    copula = copula_clayton
    my_cds_spreads_bps = [100, 150, 200, 250, 300, 400, 500]
    my_mc_paths = 1_000_000
    name = 'hem_hem_hem'
    result_folder_name = Path(Path().cwd().parent, 'results/credit/')
    first_to_default_spread(name=name, copula=copula, models=my_models, cds_spreads_bps=my_cds_spreads_bps,
                            maturity=my_maturity, mc_paths=my_mc_paths, result_folder_name=result_folder_name)


if __name__ == '__main__':
    generate_benchmark_singe_name_cds()
    generate_benchmark_first_to_default_cds()
