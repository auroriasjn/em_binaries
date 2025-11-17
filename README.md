[![Python Version 3.9](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/github/license/auroriasjn/DRAGONWebInterface)](https://github.com/auroriasjn/DRAGONWebInterface/blob/master/LICENSE)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# EM Binaries Project
Unresolved binaries and multiple systems play a crucial and yet largely underrated role in a variety of astrophysical contexts. Since unresolved binaries typically fall above ~0.5-0.75 mag of a fitted isochrone in a CMD if their mass ratio $q = 1.0$ (in other words, they broaden the main-sequence in a cluster), unresolved binaries—specifically, contact binaries when looking at photometric data—can severely bias estimates of the age, metallicity, and stellar type classification. To attempt to correct for this, we design a isochrone mixture model that takes into account both the weight of a single-star isochrone and also model a companion isochrone with companion mass at a ratio $q$ to the primary. We show that this model is mildly successful but fails when the cluster has a variety of ages (synthetic data).

# Installation
Please ```git clone``` this repository by running the following command:
```angular2html
git clone https://github.com/auroriasjn/em_binaries.git
```
Upon installation, please **change directories** into the `em_binaries` directory:
```angular2html
cd em_binaries
```
Make sure that you have **Anaconda** installed. Run the following command
```angular2html
./install.sh
```
This will create
an entirely new Anaconda environment for you called ```iso_env```.

**Do note that this project only works for arm64 architectures due to how isochrones is compiled under the hood. The installation FAILS on x86_64 architectures. This is not a problem with this project's code but has in fact been raised multiple times by the creators of this package. Thus, in order to run this code, you must have built your Anaconda environment for the arm64 architecture. You can verify this by seeing that when the packages are being compiled, you see `osx-arm64` during the pre-compile stage.**

# Getting Help

If you have a question, please send an email to `jeremy.ng@yale.edu`.





