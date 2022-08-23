rpylib
======

|Documentation Status| |Build Status| |Code style: black|

Scope
-----

| This Python pricing library was developed as a companion code for the
  paper:
| **"A weak Multilevel Monte-Carlo scheme for multidimensional Levy-type
  processes"** (`link to the paper to be added
  later <https://www.google.com>`__).

| The results consist of the numerical analysis of:

- the benchmark of
  the Continuous-Time Markov Chain (CTMC) scheme approximation against
  the series representation [1]_
- the benchmark of the CTMC scheme against the closed-form formula for
  First-to-Default
  CDS [2]_
- the weak and strong convergence of the multilevel CTMC scheme as
  well as the convergence rate of the cost w.r.t the *rmse* compared to
  the standard Monte-Carlo; these results mimic those of
  Giles [3]_  for diffusion processes

| The different convergence rates considered in our case are dependent
  on the Blumenthal-Getoor index of the underlying Levy process.

Results
-------

| The main results are presented in the form of 4 graphs (as in
  Giles [3]_ [4]_) |

- *log2(vl)*, the log level variances in function of the level *l*
- *log2\|ml\|* the log level means in function of the level *l*
- *Nl* (optimal number of Monte-Carlo paths for the level *l*) in function of
  the level *l*
- the total costs of the multilevel Monte-Carlo and the
  standard Monte-Carlo in function of the *rmse* (root-mean square
  error)

.. image:: https://github.com/rpalfray/rpylib/blob/master/docs/pics/cgmy15.jpg
   :width: 700
   :alt: MLMC applied to CGMY with :math:`\\beta=1.5`


Scripts
-------

For the paper:
~~~~~~~~~~~~~~

The scripts below were run on a supercomputer via ``Slurm``; the Slurm
scripts are saved in *rpylib/slurm/slurmscripts* which call the ``python``
scripts in */slurm/pythonscripts* themselves calling the following
scripts:

Benchmark of the CTMC scheme in *rpylib/scripts/benchmark/* :

1. ``cds.py``
2. ``first_to_default.py``
3. ``series_representation.py``

MLMC weak/strong convergence rates in *rpylib/scripts/mlmc/convergence*
:

4. ``mlmc_convergence_1d.py``
5. ``mlmc_convergence_copulas.py``
6. ``mlmc_convergence_sde.py``

and multilevel Monte-Carlo (MLMC) applications to unidimensional Levy
processes, Levy copulas and Levy-driven SDE in
*rpylib/scripts/mlmc/applied* :

7. ``mlmc_applied_1d.py``
8. ``mlmc_applied_copulas.py``
9. ``mlmc_applied_sde.py``

Other scripts:
~~~~~~~~~~~~~~

Other scripts are available in *rpylib/scripts/statistics*. These
scripts allow to plot the distribution of the spot underlying of the
Levy process simulated by Monte-Carlo (either directly from the SDE or
from the CTMC scheme).

--------------

Contact:
^^^^^^^^

Any feedback on this project will be appreciated, please log a new Issue
or email `me <mailto:romain.palfray+rpylib@gmail.com>`__.



.. [1] `Levy Copulas: Review of Recent Results <https://link.springer.com/chapter/10.1007/978-3-319-25826-3_7>`_, P. Tankov
.. [2] `A Structural Jump Threshold Framework for Credit Risk <https://epubs.siam.org/doi/10.1137/140993892>`_, P. Garreau, A. Kercheval
.. [3] `Multilevel Monte Carlo Path Simulation <https://people.maths.ox.ac.uk/gilesm/files/OPRE_2008.pdf>`_, M.B. Giles
.. [4] `Multilevel Monte Carlo methods <https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf>`_, M.B. Giles


.. |Documentation Status| image:: https://readthedocs.org/projects/rpylib/badge/?version=latest
   :target: https://rpylib.readthedocs.io/en/latest/?badge=latest
.. |Build Status| image:: https://app.travis-ci.com/rpalfray/rpylib.svg?branch=master
   :target: https://app.travis-ci.com/rpalfray/rpylib
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
