# rpylib

[![Documentation Status](https://readthedocs.org/projects/rpylib/badge/?version=latest)](https://rpylib.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://app.travis-ci.com/rpalfray/rpylib.svg?branch=master)](https://app.travis-ci.com/rpalfray/rpylib)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Scope

This Python pricing library was developed as a companion code for the paper:  
**_"A Weak MLMC Scheme for Lévy-copula-driven SDEs with Applications to the Pricing of Credit, Equity and Interest Rate Derivatives"_** 
([**link to the paper to be added later**](https://www.google.com)).
 
The results consist of the numerical analysis of:
- the benchmark of the Continuous-Time Markov Chain (CTMC) scheme approximation against the series representation<sup>[1]</sup>
- the benchmark of the CTMC scheme against the closed-form formula for First-to-Default CDS<sup>[2]</sup>
- the weak and strong convergence of the multilevel CTMC scheme as well as the convergence rate of the cost w.r.t 
the _rmse_ compared to the standard Monte-Carlo; these results mimic those of Giles<sup>[3][4]</sup> for diffusion processes  
The different convergence rates considered in our case are dependent on the Blumenthal-Getoor index of the underlying 
Levy process.

## Results

The main results are presented in the form of 4 graphs (as in Giles<sup>[3][4]</sup>):    
- _log<sub>2</sub>(v<sub>l</sub>)_, the log level variances in function of the level _l_
- _log<sub>2</sub>|m<sub>l</sub>|_ the log level  means in function of the level _l_
- _N<sub>l</sub>_ (optimal number of Monte-Carlo paths for the level _l_) in function of the level _l_
- the total costs of the multilevel Monte-Carlo and the standard Monte-Carlo in function of the _rmse_ (root-mean square error)

MLMC applied to CGMY with beta=1.5: 

[<img src="https://github.com/rpalfray/rpylib/blob/master/docs/pics/cgmy15.jpg?raw=true" width="700" alt="MLMC applied to CGMY with beta=1.5" />](https://github.com/rpalfray/rpylib/blob/master/docs/pics/cgmy15.jpg?raw=true)


## Scripts

#### For the paper:
See the _slurm_ folder.

### Additional scripts:

Other scripts are available in _scripts/statistics_. These scripts allow to plot the distribution of the spot 
underlying of the Levy process simulated by Monte-Carlo (either directly from the SDE or from the CTMC scheme).  



 [1]: _[**Lévy Copulas: Review of Recent Results**](https://link.springer.com/chapter/10.1007/978-3-319-25826-3_7)_, P. Tankov  
 [2]: _[**A Structural Jump Threshold Framework for Credit Risk**](https://epubs.siam.org/doi/10.1137/140993892)_, P. Garreau, A. Kercheval  
 [3]: _[**Multilevel Monte Carlo Path Simulation**](https://people.maths.ox.ac.uk/gilesm/files/OPRE_2008.pdf)_, M.B. Giles  
 [4]: _[**Multilevel Monte Carlo methods**](https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf)_, M.B. Giles


***
#### Contact:
Any feedback on this project will be appreciated, please log a new Issue or email [me](mailto:romain.palfray+rpylib@gmail.com).
