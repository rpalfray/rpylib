rpylib
======

Scope
-----

| This Python pricing library was developed as a companion code for the
  paper:
| **“A weak Multilevel Monte-Carlo scheme for multidimensional Levy-type
  processes”** (`link to the paper to be added later`_).

The results consist of the numerical analysis of: - the benchmark of the
Continuous-Time Markov Chain (CTMC) scheme approximation against the
series representation\ `1`_\  - the benchmark of the CTMC scheme against
the closed-form formula for First-to-Default CDS\ `2`_\  - the weak and
strong convergence of the multilevel CTMC scheme as well as the
convergence rate of the cost w.r.t the *rmse* compared to the standard
Monte-Carlo; these results mimic those of Giles\ `3,4,5`_\ for diffusion
processes The different convergence rates considered in our case are
dependent on the Blumenthal-Getoor index of the underlying Levy process.

Results
-------

| The results are presented in the form of 4 graphs (as in
  Giles\ `3,4,5`_\):
| - *log2(variance)* in function of the level *l* - *log2\|mean\|* in
  function of the level *l* - *Nl* (optimal number of Monte-Carlo paths
  for the level *l*) in function of the level *l* - the total costs of
  the multilevel Monte-Carlo and the standard Monte-Carlo in function of
  the *rmse* (root mean square error)

Scripts
-------

For the paper:
^^^^^^^^^^^^^^

The scripts used to generate the results can be found in
*rpylib/scripts/benchmark/* : 1. ``cds.py`` 2. ``first_to_default.py``
3. ``series_representation.py``

*rpylib/scripts/mlmc/convergence* : 4. ``mlmc_convergence_1d`` 5.
``mlmc_convergence_copulas`` 6. ``mlmc_convergence_sde``

and *rpylib/scripts/mlmc/applied* : 7. ``mlmc_applied_1d`` 8.
``mlmc_applied_copulas`` 9. ``mlmc_applied_sde``

Other scripts:
~~~~~~~~~~~~~~

Other scripts are available in *rpylib/scripts/statistics*. These
scripts allow to plot the distribution of the spot underlying of the
Levy process simulated by Monte-Carlo (either directly from the SDE or
from the CTMC scheme).

.. _link to the paper to be added later: https://www.google.com
.. _1: _%5B**Lévy%20Copulas:%20Review%20of%20Recent%20Results**%5D(https://link.springer.com/chapter/10.1007/978-3-319-25826-3_7)_,%20P.%20Tankov
.. _2: _%5B**A%20Structural%20Jump%20Threshold%20Framework%20for%20Credit%20Risk**%5D(https://epubs.siam.org/doi/10.1137/140993892)_,%20P.%20Garreau,%20A.%20Kercheval
.. _3: _%5B**Multilevel%20Monte%20Carlo%20Path%20Simulation**%5D(https://people.maths.ox.ac.uk/gilesm/files/OPRE_2008.pdf)_,%20M.B.%20Giles
.. _4: _%5B**Multilevel%20Monte%20Carlo%20methods**%5D(https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf)_,%20M.B.%20Giles
.. _5: _%5B**Multilevel%20path%20simulation%20for%20jump-diffusion%20SDEs**%5D(https://people.maths.ox.ac.uk/gilesm/files/mcqmc10_xia.pdf)_,%20Y.%20Xia,%20M.B.%20Giles


