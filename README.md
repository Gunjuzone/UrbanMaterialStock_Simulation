# UrbanMaterialStock_Simulation
Probabilistic simulation framework for estimating urban building material stocks under typology and geometric uncertainty. Includes AutoML-based classification, calibrated typology sampling, and Monte Carlo modeling using RASMI material intensities.

##  Overview
This repository contains the implementation of a probabilistic simulation framework for estimating urban building material stocks under typology and geometric uncertainty.
It accounts for:

- **Typology uncertainty** via probabilistic classification and calibrated sampling.
- **Geometric uncertainty** through randomized perturbations of building height and area.
- **Material intensity variability** using p50 values from the RASMI dataset


The framework supports:
- Monte Carlo simulations
- Typology classification using AutoML
- Uncertainty quantification
- Typology- and material-specific stock analysis

##  Dataset Overview

- `RASMI_intensity.csv`: Contains p50 material intensity values (kg/m²) for different building categories based on height and typology. Materials include Concrete, Brick, Wood, Steel, Glass, Plastics, Aluminium, and Copper.
- `MC_dataset.csv`: Includes building-level attributes such as coordinates, area, height, true and predicted typology, and calibrated probabilities for Residential, Mixed-Use, Institutional, and Amenities classes.
- `AutoML_Sample_Dataset.csv`: Provides geometric and morphological features used for typology classification, including perimeter, aspect ratio, convexity, solidity, and form ratio.
- ⚠️ Note: The actual dataset for AutoML classificcation could not be shared due to licensing restrictions. A synthetic sample is provided instead for structural reference only. 


##  Core Script

- `Monte_Carlo_Main.py`: Main simulation script implementing:
  - Typology sampling
  - Geometric perturbation
  - Material stock estimation
  - Uncertainty metrics
  - Building-level and district level material total stock analysis

---

##  How to Run

python Monte_Carlo_Main.py

 ## Requirements
- pandas
- numpy
- matplotlib
