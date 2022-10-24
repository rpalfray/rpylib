### Background

The data in the paper was generated via the scripts in the `slurmscripts` folder written in [Slurm](https://slurm.schedmd.com/documentation.html) 
and sent to a supercomputer.

The scipts of `slurmscripts` call the ones in the `pythonscripts` folder:
> Benchmark: 
> 1. `benchmark_credit.py`: the benchmark  of the CTMC against the closed-form formula by Garreau-Kercheval for a First-to-Default CDS
> 2. `benchmark_series_representation.py`: the benchmark of the CTMC against the series representation 

> MLMC:
> 
> Scripts testing the convergence rates of the MLMC:
> 1. `mlmc_giles_convergence_1d.py`: one-dimensional case
> 2. `mlmc_giles_convergence_copulas.py`: Lévy copula case
> 3. `mlmc_giles_convergence_sde.py`: Lévy-driven SDE case
> 
> Scripts testing the applications of the MLMC (calculation of the complexity of the algorithm):
> 1. `mlmc_giles_applied_1d.py`: one-dimensional case
> 2. `mlmc_giles_applied_copulas.py`: Lévy copula case
> 3. `mlmc_giles_applied_sde.py`: Lévy-driven SDE case

