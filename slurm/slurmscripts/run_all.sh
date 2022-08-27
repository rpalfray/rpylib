#!/bin/bash


#############################
## Variances/Means scripts ##
#############################
cd "$HOME"/code/rpylib/slurm/slurmscripts/mlmc/convergence || exit

## 'quick' scripts
sbatch -J CQgmy15 convergence_1d_quick.sbatch cgmy15
sbatch -J CQ3d convergence_copulas_quick.sbatch hem_vg_cgmy11
sbatch -J CQsde convergence_sde_quick.sbatch cgmy02_cgmy04

## 'regular' scripts
sbatch -J Cgmy15 convergence_1d.sbatch cgmy15
sbatch -J C3d convergence_copulas.sbatch hem_vg_cgmy11
sbatch -J Csde convergence_sde.sbatch cgmy02_cgmy04

sleep 300 ## wait a few minutes so that all the data has been generated (with the '_quick' scripts)
## for the 'applied' scripts

##########################
## Applied MLMC scripts ##
##########################
cd "$HOME"/code/rpylib/slurm/slurmscripts/mlmc/applied || exit

## 1d
COUNT_RMSE_1D=0
for RMSE in 0.1 0.07 0.05 0.04 0.03 0.02; do
    export RMSE_1D
    sbatch -J A1d#${COUNT_RMSE_1D} applied_1d.sbatch cgmy15 ${RMSE}
    (( COUNT_RMSE_1D++ ))
    sleep 0.1
done

## COPULA
COUNT_RMSE_C=0
for RMSE in 0.002 0.003 0.006 0.01 0.015 0.03; do
  sbatch -J A3d#${COUNT_RMSE_C} applied_copulas.sbatch hem_vg_cgmy11 ${RMSE}
  (( COUNT_RMSE_C++ ))
  sleep 0.1
done

## SDE
COUNT_RMSE_SDE=0
for RMSE in 0.006 0.005 0.004 0.003 0.002 0.001; do
  sbatch -J Asde#${COUNT_RMSE_SDE} applied_sde.sbatch cgmy02_cgmy04 ${RMSE}
  (( COUNT_RMSE_SDE++ ))
  sleep 0.1
done

###################
## Credit/Series ##
###################
cd "$HOME"/code/rpylib/slurm/slurmscripts/ || exit
sbatch -J BSeries benchmark_series_representation.sbatch
sbatch -J BCredit benchmark_credit.sbatch

