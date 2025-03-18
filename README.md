# unbinned-inference-paper

This is the code used for the studies described in **Unbinned Inference with Correlated Events** (arXiv:xxxx).  Detector resolution effects in experimental data can be statistically removed using unfolding techniques.  Advances in machine learning have enabled the unfolding of unbinned, high-dimensional data, where the output is a set of simulated events with event weights that are assigned by the unfolding.  In our paper, we present studies of how statistical correlations in the event weights affect inference tasks performed on the unfolding output. 

Here are the main pieces of code with brief descriptions.

- **run-kde-study-1d.ipynb** :  Generates toy samples of 1D data and performs unbinned unfolding using OmniFold with a simple Kernel Density Estimator for the pdfs inside OmniFold.
- **run-nn-study-nd.ipynb** : Generates toy samples of 1D, 2D, 4D, or 6D data and performs unbinned unfolding using OmniFold where the pdf ratios are estimated using Neural Networks.
- **process-nd-output.ipynb** :  Runs all of the post-unfolding analysis including binned and unbinned fits and evaluations of the weight correlations.
- **GenerateInputSamples.py** :  Generates samples for analysis with Iterative Bayesian Unfolding.
- **iMinuitFits.py** :  Performs the IBU and binned chi2 fits of the IBU output.
- **paper-plots....ipynb** :  Makes the plots in the paper.
- **RooMultiVarGaussian2e...** :  Modified version of some RooFit code to allow fitting all covariance parameters of the multi variate Gaussian.
- **validate-unbinned-ml-fits.ipynb** :  Code to check RooMultiVarGaussian2e to make sure that the asymptotic uncertainties are correct when the events are statistically independent.

