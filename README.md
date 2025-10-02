# UrbanMaterialStock_Simulation
Probabilistic simulation framework for estimating urban building material stocks under typology and geometric uncertainty. Includes AutoML-based classification, calibrated typology sampling, and Monte Carlo modeling using RASMI material intensities.

This repository contains the implementation of a probabilistic simulation framework for estimating urban building material stocks under typology and geometric uncertainty. It supports reproducible Monte Carlo simulations, typology classification using AutoML, and analysis of sample size effects on material-specific uncertainty.


## 📦 Dataset Overview

- `RASMI_intensity.csv`: Contains p50 material intensity values (kg/m²) for different building categories based on height and typology. Materials include Concrete, Brick, Wood, Steel, Glass, Plastics, Aluminium, and Copper.
- `Typology_P_Final.csv`: Includes building-level attributes such as coordinates, area, height, true and predicted typology, and calibrated probabilities for Residential, Mixed-Use, Institutional, and Amenities classes.
- `AutoML_Sample_Dataset.csv`: Provides geometric and morphological features used for typology classification, including perimeter, aspect ratio, convexity, solidity, and form ratio.
- ⚠️ Note: The actual dataset for AutoML classificcation could not be shared due to licensing restrictions. A synthetic sample is provided instead for structural reference only. 
