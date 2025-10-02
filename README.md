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
- ⚠️ Note: The actual dataset for AutoML classification and Monte Carlo Simulation could not be shared due to licensing restrictions. A synthetic sample is provided instead for structural reference only. 

---

## 🗺️ Zoning Data Source

Typology classification was partially derived from zoning keyword analysis using municipal land-use records from Ajuntament de Barcelona. These records were merged with building geometry from the Mazar dataset using ArcGIS Pro.

The original zoning dataset is publicly available and referenced in the manuscript:
**Ajuntament de Barcelona.** (2025). *Barcelona municipal land-use records*. Urban Planning Information Portal. Retrieved July 15, 2025, from  https://w20.bcn.cat/CartoBCN/getFile.ashx?prod=107.BARCELONA.2

##  Core Script

- `Monte_Carlo_Main.py`: Main simulation script implementing:
  - Typology sampling
  - Geometric perturbation
  - Material stock estimation
  - Uncertainty metrics
  - Building-level and district level material total stock analysis

- `Sample_Size_Vs_Uncertainty.py`: Analyzes the effect of sample size on material-specific uncertainty

  - Runs Monte Carlo simulations across progressively larger building samples
  - Computes and plots coefficient of variation (CV) vs. sample size
  - Produces per-material uncertainty curves and scaling plots
  - Outputs summary statistics and reduction ratios showing how uncertainty decreases as sample size grows
---

##  How to Run
### Step 1: Run AutoML Typology Classification
python AutoML_Typology_Classifier.py

### Step 2: Run Monte Carlo Simulation
python Monte_Carlo_Main.py

 ## Requirements
- pandas
- numpy
- matplotlib
### Install via:
  pip install -r requirements.txt

