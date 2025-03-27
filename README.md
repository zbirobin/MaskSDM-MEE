# MaskSDM

This repository contains the code to reproduce the results from the paper:
**MaskSDM with Shapley values to improve flexibility, robustness, and explainability in species distribution modeling.**

## Getting Started

### Requirements

Dependencies can be installed using:

```sh
pip install -r requirements.txt
```

### GPU Support

All training and experiments were conducted on an NVIDIA GeForce RTX 3090 with 24GB of memory. The model can be run on a CPU, but computational time will be significantly longer.

## Data Acquisition

### Processed Data

The processed datasets are available [here on Zenodo](https://zenodo.org/records/15096721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIyM2FjMjg1LWVlZmMtNDIwOS1iZGU3LTdjMzhlNDY3YjIwYyIsImRhdGEiOnt9LCJyYW5kb20iOiIyMWE2NWU2YmU2NGY1YzVhZmM4ZWI2MWY3ODEwNDM0OSJ9.TSW3U7BAincAtI-P_tmI_CpBFjUkxRdAo2DQ9wK5TUUBd7YzG1cYi8uPXN74vZlNvxFRAzQMiRhKI1d290kepQ).

All the data is available in the `data` zipped folder and should be copied into the corresponding `data` folder.

### Extracting Data

The following notebooks are used to process the original raw data:

- `data/predictors_extraction.ipynb`: Extracts environmental predictors for each sPlotOpen plot.
- `data/occurrences.ipynb`: Processes sPlotOpen plots to create a species occurrence matrix.
- `data/generate_map_data.ipynb`: Extracts environmental predictors for the area of interest, e.g., for making predictions.

These notebooks require downloading the original environmental rasters and species occurrence data and placing them in the `data` folder.

### Data Sources

- **Species Data:** Available from the [iDiv Repository](https://doi.org/10.25829/idiv.3474-bb7k72)
- **Environmental Predictors:**
  - [WorldClim](https://www.worldclim.org/data/worldclim21.html)
  - [SoilGrids](https://soilgrids.org/)
  - [Human Influence Data](https://doi.org/10.5061/dryad.052q5)
  - [SatCLIP Models](https://github.com/microsoft/satclip)

- **SatCLIP:** MaskSDM utilizes SatCLIP embeddings. To install SatCLIP in the `data` folder, run:

```sh
git clone https://github.com/microsoft/satclip.git data/satclip
```


### Trained Models

Trained checkpoints of the model are available [here on Zenodo](https://zenodo.org/records/15096721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIyM2FjMjg1LWVlZmMtNDIwOS1iZGU3LTdjMzhlNDY3YjIwYyIsImRhdGEiOnt9LCJyYW5kb20iOiIyMWE2NWU2YmU2NGY1YzVhZmM4ZWI2MWY3ODEwNDM0OSJ9.TSW3U7BAincAtI-P_tmI_CpBFjUkxRdAo2DQ9wK5TUUBd7YzG1cYi8uPXN74vZlNvxFRAzQMiRhKI1d290kepQ).

The associated files are in the `models` zipped folder and should be copied into the corresponding `models` folder while maintaining the same folder structure.

## Training MaskSDM

To train the MaskSDM model, run:

```sh
python train_model.py
```

## Reproducing Results and Figures

### Results and Tables

- **`results.ipynb`**: Reproduces the results in:
  - Table 2
  - Figure 3
  - Figure 9
  - Table 4, Table 5, Table 6

### Prediction Maps

- **`predictions_maps.ipynb`**: Generates the prediction maps for:
  - Figure 4
  - Figure 11
  - Figure 12
  - Figure 13

### Shapley Value Analysis

The Shapley values are precomputed and available for download [here on Zenodo](https://zenodo.org/records/15096721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIyM2FjMjg1LWVlZmMtNDIwOS1iZGU3LTdjMzhlNDY3YjIwYyIsImRhdGEiOnt9LCJyYW5kb20iOiIyMWE2NWU2YmU2NGY1YzVhZmM4ZWI2MWY3ODEwNDM0OSJ9.TSW3U7BAincAtI-P_tmI_CpBFjUkxRdAo2DQ9wK5TUUBd7YzG1cYi8uPXN74vZlNvxFRAzQMiRhKI1d290kepQ).

The associated files are in the `results` zipped folder and should be copied into the corresponding `results` folder.

- **`shapley_global.ipynb`**: Computes the global Shapley values for:
  - Table 1
  - Figure 5
  - Figure 7
  - Figure 8

- **`shapley_local.ipynb`**: Computes the local Shapley values for:
  - Figure 6


## Additional files

- **`data_helpers.py`**: Provides functions for loading and processing data.  
- **`losses.py`**: Defines the loss functions used to train the models.  
- **`modules.py`**: Implements the PyTorch modules required to build MaskSDM.  
- **`train_model.py`**: Runs the training process for MaskSDM and baseline models. Model parameters are defined here and can be adjusted as needed.  
- **`training_helpers.py`**: Includes utility functions to facilitate model training.  
