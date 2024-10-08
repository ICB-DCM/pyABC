---
title: 'pyABC: Efficient and robust easy-to-use approximate Bayesian computation'
tags:
  - approximate Bayesian computation
  - ABC
  - likelihood-free inference
  - high-performance computing
  - parallel
  - sequential Monte Carlo
  - Python
authors:
  - name: Yannik Schälte
    orcid: 0000-0003-1293-820X
    affiliation: "1, 2, 3"
  - name: Emmanuel Klinger
    affiliation: "2, 3, 4"
  - name: Emad Alamoudi
    orcid: 0000-0002-9129-4635
    affiliation: "1"
  - name: Jan Hasenauer
    orcid: 0000-0002-4935-3312
    affiliation: "1, 2, 3"
affiliations:
  - name: Faculty of Mathematics and Natural Sciences, University of Bonn, Bonn, Germany
    index: 1
  - name: Institute of Computational Biology, Helmholtz Center Munich, Neuherberg, Germany
    index: 2
  - name: Center for Mathematics, Technical University Munich, Garching, Germany
    index: 3
  - name: Department of Connectomics, Max Planck Institute for Brain Research, Frankfurt, Germany
    index: 4
date: 26 February 2022
bibliography: paper.bib
---

<!--250-1000 words-->

# Summary

<!--describe high-level functionality and purpose of the software for a diverse, non-specialist audience-->

The Python package pyABC provides a framework for approximate Bayesian computation (ABC), a likelihood-free parameter inference method popular in many research areas.
At its core, it implements a sequential Monte-Carlo (SMC) scheme, with various algorithms to adapt to the problem structure and automatically tune hyperparameters.
To scale to computationally expensive problems, it provides efficient parallelization strategies for multi-core and distributed systems.
The package is highly modular and designed to be easily usable.
In this major update to pyABC, we implement several advanced algorithms that facilitate efficient and robust inference on a wide range of data and model types.
In particular, we implement algorithms to accurately account for measurement noise, to adaptively scale-normalize distance metrics, to robustly handle data outliers, to elucidate informative data points via regression models, to circumvent summary statistics via optimal transport based distances, and to avoid local optima in acceptance threshold sequences by predicting acceptance rate curves.
Further, we provide, besides previously existing support of Python and R, interfaces in particular to the Julia language, the COPASI simulator, and the PEtab standard.

# Statement of Need

<!--clearly illustrate the research purpose of the software-->

Mathematical models are important tools to describe and study real-world systems, allowing to understand underlying mechanisms [@Gershenfeld1999; @Kitano2002].
They are commonly subject to unknown parameters that need to be estimated using observed data [@Tarantola2005].
The Bayesian framework allows doing so by updating prior beliefs about parameters by the likelihood of data given parameters.
However, especially for complex stochastic models, evaluating the likelihood is often infeasible [@TavareBal1997; @Wilkinson2009; @JagiellaRic2017].
Thus, likelihood-free methods such as ABC have been developed [@AndrieuRob2009; @GourierouxMon1993; @PriceDro2018; @PritchardSei1999; @BeaumontZha2002].
ABC is widely applicable, as it only requires an executable forward process model, simulating data given model parameters.
In a nutshell, ABC circumvents likelihood evaluation by accepting parameters if a distance between corresponding simulated and observed data is below a threshold [@SissonFan2018Handbook] (\autoref{fig:concept}).
ABC is often combined with a sequential Monte-Carlo (ABC-SMC) scheme using importance sampling, which gradually reduces the acceptance threshold and thus improves the posterior approximation, while maintaining high acceptance rates [@SissonFan2007; @DelMoralDou2006].

![**Basic ABC algorithm.** Parameters $\theta\sim\pi(\theta)$ are sampled from the prior or a proposal distribution, and passed to a simulator model (here exemplarily biological systems) generating potentially stochastic simulated data according to the likelihood $y\sim\pi(y|\theta)$. These are optionally passed through a summary statistics function (here exemplarily a neural network model as employed in @SchaelteHas2022Pre) giving a low-dimensional representation $s(y)$. Summary statistics of simulated and observed data are compared via a distance metric $d$, and the underlying parameters accepted if the distance is below an acceptance threshold $\varepsilon$.\label{fig:concept}](concept.pdf){ width=99% }

While conceptually simple and widely applicable, ABC is computationally expensive, as it requires simulating the forward model many times for different parameters. Its practical performance relies on a number of factors.
pyABC implements at its core an ABC-SMC scheme based on @ToniStu2010 and facilicates robust and efficient inference for a broad spectrum of applications via robust methods and self-tuned choices of hyperparameters, reducing the need for manual tuning.
An article on core features of pyABC was previously published [@KlingerRic2018], discussing in particular adaptive transition kernels [@FilippiBar2013], population sizes [@KlingerHas2017], and wall-time efficient parallelization via dynamic scheduling.
pyABC is in use in a variety of fields, e.g. to model virus transmission on cellular [@ImleKum2019] and population level [@KerrStu2021], neuron circuits [@BittnerPal2021], cancer [@ColomHer2021], gene expression [@CoulierHel2021], axolotl regeneration [@CostaOts2021], universe expansion [@BernardoSai2021], cardiac electrophysiology [@CantwellYum2019], and bee colonies [@MinucciCur2021].

Besides pyABC, there exist several other software packages implementing different algorithms in different languages, each with their own strengths, including notably in Python ABCpy [@DuttaSch2017] and ELFI [@LintusaariVuo2018], in Julia GpABC [@TankhilevichIsh2020], and in R EasyABC [@JabotFau2013].
In particular, ABCpy, ELFI and GpABC, which are actively maintained at the time of writing, implement various likelihood-free methods, including and beyond ABC-SMC, such as Bayesian optimization, synthetic likelihoods, or Gaussian process emulation.
In contrast, pyABC focuses on providing a broadly applicable, efficient and robust implementation of ABC-SMC with various tailored algorithms.
Exclusive to pyABC are, at the time of writing and to the best of our knowledge, the following features:
While ABCpy and ELFI also allow distributed execution via static scheduling, only pyABC implements dynamic scheduling, further improving wall-time efficiency [@KlingerRic2018].
Further, most methods described in the below section are only implemented in pyABC:
While e.g. ELFI implements basic adaptive distances based on @Prangle2017, only pyABC implements the robuster approach @SchaelteAla2021.
While e.g. ABCpy implements basic regression-based summary statistics based on @FearnheadPra2012, only pyABC implements the approaches in @SchaelteHas2022Pre overcoming limitations in the method by @FearnheadPra2012, improving the posterior approximation and introducing sensitivity weights.
Further, only pyABC implements the exact method under measurement noise @SchaelteHas2020.

# New Features

The methods discussed in the following have been newly implemented in pyABC (version 0.12), with details in the API documentation and Jupyter example notebooks accessible via the online documentation:

*Variations in different data scales and robustness to outliers.* In ABC, a distance metric is used to quantify differences between simulated and observed data.
When simulations for different data points vary on different scales, highly variable ones dominate the acceptance decision.
@Prangle2017 introduces a method to, in an ABC-SMC framework, iteratively update distance weights to normalize contributions.
A further problem are outliers, i.e. errors in the measurement process that the model does not account for [@GhoshVog2012; @MotulskyChr2003].
@SchaelteAla2021 show that an approach adapted from @Prangle2017 can robustly identify outliers and reduce their impact.
The approaches by @Prangle2017 and @SchaelteAla2021 are now implemented in pyABC.

*Identify informative data.* Instead of operating on the full data, in ABC often summary statistics, i.e. low-dimensional data representations, are employed [@BlumNun2013].
A particular line of approaches uses as statistics the outputs of inverse regression models of parameters on data, e.g. via linear regression [@FearnheadPra2012], neural networks [@JiangWu2017], or Gaussian processes [@BorowskaGiu2021].
In @SchaelteHas2022Pre, such approaches are combined with adaptive scale-normalization, and extended to achieve a higher-order posterior approximation. Further, inverse regression models are used to, instead of constructing summary statistics, inform robust sensitivity weights accounting for informativeness.
All of the above approaches are now implemented in pyABC, with regression models interfaced via scikit-learn [@scikitlearn2011].

*Accurate handling of noise.* The approximation error of ABC methods is often unclear.
@Wilkinson2013 shows that ABC can be considered as giving exact inference under an implicit distance-induced noise model.
In @SchaelteHas2020, this insight is used to develop an efficient ABC-SMC based exact inference scheme in the presence of measurement noise. The framework is now integrated into pyABC.

*Optimal transport distances.* Besides the above-mentioned adaptive distances, pyABC now in particular implements Wasserstein distances, which consider an optimal transport problem between distributions and may allow to circumvent the use of summary statistics [@BerntonJac2019].

*Acceptance threshold selection.* Efficiency and convergence of ABC-SMC algorithms further depend on the acceptance threshold sequence. @SilkFil2013 discuss that common schemes based on quantiles of previous values [@DrovandiPet2011] can fail in the presence of local minima, and propose a method based on analyzing predicted acceptance rate curves.
pyABC now implements a modified version, using importance sampling instead of unscented transform to predict the acceptance rate as a function of the threshold.

*Interoperability.* Not only algorithms, but also accessibility and interoperability determine the usefulness of a tool.
Besides natural support of Python, and previously established support of R, pyABC now also provides an efficient interface to models written in Julia [@BezansonEde2017], to biochemical pathway models defined in SBML or COPASI format by using the COPASI toolbox [@HoopsSah2006], and supports the PEtab inference standard [@SchmiesterSch2021], currently only for the ODE simulator AMICI [@FroehlichWei2021].
Finally, it allows to connect to models written in arbitrary languages and frameworks via file exchange.

# Availability and Development

pyABC is being developed open-source under a 3-clause BSD license. The code, designed to be highly modular and extensible, is hosted on [GitHub](https://github.com/icb-dcm/pyabc) and continuously tested.
Extensive documentation is hosted on [Read the Docs](https://pyabc.rtfd.io), including API documentation and numerous Jupyter notebooks containing tutorials, outlining features, and showcasing applications.

# Acknowledgements

We thank many collaboration partners and pyABC users for valuable input, in particular Frank Bergmann for the COPASI wrapper, and Elba Raimúndez for fruitful discussions.
This work was supported by the German Federal Ministry of Education and Research (BMBF)
(FitMultiCell/031L0159 [Link to project](https://fitmulticell.gitlab.io/) and EMUNE/031L0293) and the German Research Foundation (DFG)
under Germany’s Excellence Strategy (EXC 2047 390873048 and EXC 2151 390685813).

# References
