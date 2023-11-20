# pyDRTtools

We are pleased to introduce the pyDRTtools, is the python version of DRTtools for computing distribution relaxation times (DRT) from electrochemical impedance spectroscopy (EIS) data. 

**What is the pyDRTtools? Why would I want it?**

pyDRTtools is a Python GUI that analyzes EIS data via the DRT model. pyDRTtools includes:

- an intuitive GUI for computing DRT based on Tikhonov regularization

- several options for optimizing the estimation of the DRT

- a sampler that allows you to determine the credible intervals of your DRT

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
pip install cvxopt jupyterlab PyQt5
pip install -i https://test.pypi.org/simple/ pyDRTtools==0.2.8.82
```
**ipython**
```
!launchGUI
```
Detailed installation instructions are available in the DRT toolbox user's guide (also included with the standard distribution).

**How to cite this work?**
Refer to the GitHub repository (https://github.com/ciuccislab/pyDRTtools) so that you can cite each function appropriately.
# **How to get support?**
Just write to francesco.ciucci@ust.hk or francesco.ciucci@uni-bayreuth.de
