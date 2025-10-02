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
- `Typology_P_Final.csv`: Includes building-level attributes such as coordinates, area, height, true and predicted typology, and calibrated probabilities for Residential, Mixed-Use, Institutional, and Amenities classes.
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

## 🚀 How to Run

python Monte_Carlo_Main.py

Or use the main function directly:

from Monte_Carlo_Main import main_geometric_only

mc, building_stats, uncertainty_metrics, calibration_results = main_geometric_only(
    data_path="Typology_P_Final.csv",
    rasmi_intensity_path="RASMI_intensity.csv",
    n_iterations=3000,
    temperature=1.5,
    save_dir="geometric_mc_results",
    height_uncertainty=0.15,
    area_uncertainty=0.15,
    enable_typology_uncertainty=True
)
