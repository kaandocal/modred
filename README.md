# Variational Model Reduction for the Chemical Master Equation

This repository contains code for the paper

K. Öcal, G. Sanguinetti and R. Grima, "Model Reduction for the Chemical Master Equation: an Information-Theoretic Approach", [arXiv:2210.05329](http://arxiv.org/abs/2210.05329) (2022)

The code in this repository is designed to integrate with [Catalyst.jl](https://github.com/SciML/Catalyst.jl). The principal part of this repository is contained in the file [likelihoods.jl](./likelihoods.jl), which allows the user to compute log-likelihoods of trajectories generated using the SSA (Eq. 4 in the paper), for an arbitrary stochastic reaction network. The file [hiddenlikelihoods.jl](./hiddenlikelihoods.jl) uses the filtering version of the CME to compute log-likelihoods of projected trajectories; this is only necessary to numerically evaluate KL divergences. A simple demonstration of the main approach can be found in the file [tg/demo.jl](tg/demo.jl). The rest of the files show how to implement various parts of the paper.

If you have any questions or comments, please feel free to contact the authors or open a pull request on GitHub.

### Models:
- Telegraph Model (folder `tg`)
- Autoregulatory Feedback Loop (folder `afl`)
- Michaelis-Menten System (folder `mm`)
- Genetic Oscillator (folder `osc`)
