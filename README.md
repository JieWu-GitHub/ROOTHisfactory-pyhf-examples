<!--
 * @Author       : Jie Wu j.wu@cern.ch
 * @Date         : 2025-02-12 02:06:51 +0100
 * @LastEditors  : Jie Wu j.wu@cern.ch
 * @LastEditTime : 2025-02-13 09:21:45 +0100
 * @FilePath     : README.md
 * @Description  : 
 * 
 * Copyright (c) 2025 by everyone, All Rights Reserved. 
-->
# Overview

This repository provides a self-contained example demonstrating how to perform a template fit using [pyhf](https://github.com/scikit-hep/pyhf). It adapts an online HistFactory tutorial and refactors it into a more realistic scenario. Specifically, we fit the observable \( R_{D^*} \) in the decay \( D^* \to \tau \nu_\tau \), where \( R_{D^*} \) is defined as:

\[
R_{D^*} = \frac{\mathcal{B}(D^* \to \tau \nu_\tau)}{\mathcal{B}(D^* \to \mu \nu_\mu)}.
\]

In this example, there is a single analysis region containing four samples: 
    - `sigmu`
    - `sigtau`
    - `D1`
    - `misID`
The template histograms are stored in `DemoHistos.root` as three-dimensional histograms, and the data histogram is named `h_data`.

---

## Scripts

This example includes the following scripts, which perform the **same fit** but use different tools or languages:

1. **`HistFactDstTauDemo_new.C`**  
   Uses **CERN ROOT HistFactory** in C++.
   Can be run by typing `root -l run_HistFactDstTauDemo_new.C` in the terminal.

2. **`HistFactDstTauDemo.py`**  
   Uses **CERN ROOT HistFactory** in Python, mirroring the structure of the C++ script.
   Can be run by typing `python run_HistFactDstTauDemo.py` in the terminal.

3. **`HistFactDstTauDemo_pyhf.py`**  
   Uses [pyhf](https://github.com/scikit-hep/pyhf) to flatten the 3D histograms into 1D, then performs the fit.  
   Also uses [cabinetry](https://github.com/scikit-hep/cabinetry) for visualization.
   Can be run by typing `python run_HistFactDstTauDemo_pyhf.py` in the terminal.

4. **`HistFactDstTauDemo_pyhf_new.ipynb`**
   In this jupyter notebook, we first flatten the 3D histograms into 1D, then perform the fit by using pyhf.
   The model is constructed by using 
      1. pyhf, and the fit is performed by using pyhf.
      2. cabinetry, where the 1D flattened histograms are collected and used for the model construction.
---

Feel free to explore these scripts as a reference for your own analyses or as a stepping stone for integrating ROOT-based workflows with pyhf.