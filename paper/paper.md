---
title: 'whampy: Python Package to Interact with, Visualize, and Analyze the Wisconsin H-Alpha Mapper - Sky Survey'
tags:
  - Python
  - astronomy
  - interstellar medium
  - emission lines
  - position-position-velocity
  - kinematics
authors:
  - name: Dhanesh Krishnarao
    orcid: 0000-0002-7955-7359
    affiliation: 1
affiliations:
 - name: Department of Astronomy, University of Wisconsin-Madison
   index: 1
date: 15 November 2019
bibliography: paper.bib
---

# Summary

The Wisconsin H-Alpha Mapper (WHAM) is the worlds most sensitive instrument observing diffuse optical emission lines in the Milky Way to study the warm ionized medium [@Haffner:2003]. The completed WHAM - Sky Survey (WHAM-SS) is an all sky 3D H-alpha emission map in position-position-velocity space. The survey is composed of over 50,000 spectra that each equally sample a 1-degree beam of the sky on an irregular grid. While regularly gridded data-cubes are available, they require an interpolation process that can sometimes leave artifacts. 

The ``whampy`` package provides an easy way to work directly with the individual observed spectra without a need for interpolation. This package uses ``astropy`` [@astropy] to load the entire survey and simplifies the visualization and data analysis methods commonly used with optical spectroscopy. The ``whampy.SkySurvey`` class contains all of this functionality along with specific routines to perform analysis done in WHAM science papers, such as @Krishnarao:2017. This package also contains methods primarily intended for the core WHAM team to work with preliminary data products at other wavelengths ([NII], [SII], H-beta) and for data reduction and calibration processes described in @Haffner:2003. Future survey releases and multiwavelength data products can be analyzed with the current functionality of this package as well. 

An example of the type of analysis and visualizations that can be made using ``whampy`` is shown below. In this example, emission from the far Carina spiral arm is kinematically isolated using the longitude-velocity track defined in @Reid:2016. The map integrates each H-alpha spectrum in a 16 km/s window centered around this track to create a zeroth order moment map of the spiral arm as done in the analysis of @Krishnarao:2017.

![Example spiral arm map of the far Carina arm using ``whampy``.](figure.pdf)

# References