# pyDRTtools

We are pleased to introduce the pyDRTtools, is the python version of DRTtools for computing distribution relaxation times (DRT) from electrochemical impedance spectroscopy (EIS) data. 

**What is the pyDRTtools? Why would I want it?**

pyDRTtools is a Python GUI that analyzes EIS data via the DRT model. pyDRTtools includes:

- an intuitive GUI for computing DRT based on Tikhonov regularization

- several options for optimizing the estimation of the DRT

- a sampler that allows you to determine the credible intervals of your DRT
  
- an optimal selection of the regularization parameter

- Hilbert-transform subroutines that allow you to assess and score the quality of your data

Hopefully, by now you are inclined to think that this toolbox may be useful to the interpretation of your EIS data. If you are interested, you will find an explanation of the toolbox's capabilities it in the user's guide as well as in the references below.

## Distribution and Release Information

pyDRTtools is freely available under the MIT license from this site.

**System requirements**

To install and run pyDRTtools, you need: Python >= 3

**Installation details**

For details about the installation procedures, you can consult the user manual [manual](manual)

**Run the following on anaconda prompt:**
```
conda create --name DRT pip ipython pandas matplotlib scikit-learn spyder
conda activate DRT
pip install cvxopt PyQt5
pip install pyDRTtools
```
**ipython**
```
!launchGUI
```
Detailed installation instructions are available in the DRT toolbox user's guide (also included with the standard distribution).

**How to cite this work?**

Refer to the GitHub repository (https://github.com/ciuccislab/pyDRTtools) so that you can cite each function appropriately.

[1] Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499.*

Link: https://doi.org/10.1016/j.electacta.2015.09.097

if you want to add more details about standard regularization methods for computing the regularization parameter used in ridge regression, you should cite the following references also:

*[2] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the Regularization Parameter in the Distribution of Relaxation Times, Journal of the Electrochemical Society, 170 (2023) 030502.*

Link: https://doi.org/10.1149/1945-7111/acbca4

if you are presenting the *Bayesian credible intervals* generated by the pyDRTtools in any of your academic works, you should cite the following references also:

[3] Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454.*

Link: https://doi.org/10.1016/j.electacta.2015.03.123

[4] Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129.*

Link: https://doi.org/10.1016/j.electacta.2017.07.050

if you are using the pyDRTtools to compute the *Hilbert Transform*, you should cite:

[5] Liu, J., Wan, T. H., & Ciucci, F. (2020).A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores. Electrochimica Acta, 357, 136864.*

Link: https://doi.org/10.1016/j.electacta.2020.136864

**How to get support?**

Just write to francesco.ciucci@ust.hk or francesco.ciucci@uni-bayreuth.de
