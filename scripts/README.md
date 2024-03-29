# Background

The scripts in this folder illustrate the use of this library which was originally developed as a companion code of 
the paper [**_"A Weak MLMC Scheme for Lévy-copula-driven SDEs with Applications to the Pricing of Credit, Equity and Interest Rate Derivatives"_**](https://arxiv.org/abs/2211.02528)
but the library itself aims to be generic and the implementation strives to keep concepts as orthogonal as possible, 
hence giving the opportunity to external users to easily integrate their own features 
(pricing models, engines, approximation schemes, etc).

## benchmark
Benchmark of the CTMC against the series representation (`series_representation.py`) and the closed-form formulas for a 
single-name CDS as well as a first-to-default CDS given by the modelling framework of Garreau and Kercheval.

## mlmc:
Generation of data to check the MLMC algorithm with regard to the weak and strong convergence rates. 
The `convergence` and `application` folders contains examples of the MLMC applied to a CGMY process, a 3d Lévy copula and a 
2d Levy-driven SDE.

## statistics:
### CTMC
Examples of the Continuous-Time Markov chain approximation applied to:
> `levy_process.py`: one-dimensional Lévy process

> `levy_copula.py`: Lévy copulas.

### Direct Simulation
> `direct.py`:
illustrates the Monte-Carlo statistics when the model can be simulated directly via Monte-Carlo; the example
at hand prices a call option with the Kou model and compare the result to the theoretical price.

### Series
Example for the series representation algorithm.
