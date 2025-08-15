# MaskSDM

This repository contains the code to reproduce the results from the paper:
**MaskSDM with Shapley values to improve flexibility, robustness, and explainability in species distribution modeling.**

## 🚀 Getting Started

This `README.md` file provides the necessary information to reproduce all the experiments and results from the paper. Additionally, we offer a detailed Python notebook, `getting_started.ipynb`, designed for users who may be less familiar with Python and PyTorch.

### ⚙️ Requirements

This project is entirely written in Python (version 3.10.18), primarily using the PyTorch library. All required dependencies can be installed in a Python virtual environment using:

```sh
pip install -r requirements.txt
```

All dependencies, along with their specific versions, are listed in the `requirements.txt` file. You may need to adjust the NVIDIA-related packages to match your hardware setup.

For a fully reproducible setup, we also provide a Dockerfile. If you are familiar with Docker, you can use it to open notebooks or run Python code:

```sh
docker build -t masksdm .
docker run -it -v $(pwd):/app --rm -p 8888:8888 masksdm
```

This Dockerfile is configured for CPU usage by default. To run it on a GPU, replace the first line with: `FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`. Note that you may need to adjust the image to match the CUDA version installed on your GPU. A list of available PyTorch images can be found [here](https://hub.docker.com/r/pytorch/pytorch/tags). Make sure to use a PyTorch 2.6 image to ensure compatibility with our code.

### 🖥️ Hardware Support

All training and experiments were performed on an NVIDIA GeForce RTX 3090 with 24GB of memory. However, the results can be reproduced on any device (GPU, eGPU, or CPU), though the computational time will vary depending on the hardware's parallel processing capabilities. The code automatically detects and uses a GPU if available; otherwise, it defaults to running on the CPU. If you encounter memory issues, especially on lower-end devices, consider reducing the batch size in the training configuration to mitigate them. You may also need to install or update the appropriate NVIDIA drivers to work with PyTorch, depending on your specific setup.

## 📊 Data Acquisition

### 💾 Processed Data

To avoid downloading and preprocessing the complete original raw datasets, you can directly access the processed versions [here on Zenodo](https://zenodo.org/records/15096721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIyM2FjMjg1LWVlZmMtNDIwOS1iZGU3LTdjMzhlNDY3YjIwYyIsImRhdGEiOnt9LCJyYW5kb20iOiIyMWE2NWU2YmU2NGY1YzVhZmM4ZWI2MWY3ODEwNDM0OSJ9.TSW3U7BAincAtI-P_tmI_CpBFjUkxRdAo2DQ9wK5TUUBd7YzG1cYi8uPXN74vZlNvxFRAzQMiRhKI1d290kepQ). All the data is available in the `data` zipped folder and should be copied and unzipped into the corresponding `data` folder of this repository.

### 📁 Data Sources

If you wish to reprocess the data yourself, you must first download the required datasets from the following sources and place them in the `data` folder of this repository:

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

### 🗃️ Extracting Data

Then the following notebooks can be used to process the original raw data:

- `data/predictors_extraction.ipynb`: Extracts environmental predictors for each sPlotOpen plot.
- `data/occurrences.ipynb`: Processes sPlotOpen plots to create a species occurrence matrix.
- `data/generate_map_data.ipynb`: Extracts environmental predictors for the area of interest, e.g., for making predictions.

These notebooks require downloading the original environmental rasters and species occurrence data and placing them in the `data` folder of this repository.


### 🤖 Trained Models

Trained checkpoints of the model are available [here on Zenodo](https://zenodo.org/records/15096721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIyM2FjMjg1LWVlZmMtNDIwOS1iZGU3LTdjMzhlNDY3YjIwYyIsImRhdGEiOnt9LCJyYW5kb20iOiIyMWE2NWU2YmU2NGY1YzVhZmM4ZWI2MWY3ODEwNDM0OSJ9.TSW3U7BAincAtI-P_tmI_CpBFjUkxRdAo2DQ9wK5TUUBd7YzG1cYi8uPXN74vZlNvxFRAzQMiRhKI1d290kepQ), allowing you to skip the training step and use the model directly.

The associated files are in the `models` zipped folder and should be copied and unzipped into the corresponding `models` folder of this repository while maintaining the same folder structure.

## 🧠 Training MaskSDM

The MaskSDM model can be trained by running the following command:

```sh
python train_model.py
```

During training, the model’s AUC performance on the validation set will be displayed in the terminal, and model checkpoints are automatically saved to the `models` folder. All key model and training parameters are defined in the `train_model.py` file and can be modified directly to suit your needs. One notable parameter is `extra_masking` that enables the random masking of additional predictors during training, i.e., masked data modeling.

## 🔍 Reproducing Results and Figures

### 📈 Results and Tables

- **`results.ipynb`**: Reproduces the results in:
  - Table 2
  - Figure 3
  - Figure 9
  - Table 4
  - Table 5
  - Table 6
  - Table 7
  - Table 8

### 🗺️ Prediction Maps

- **`predictions_maps.ipynb`**: Generates the prediction maps for:
  - Figure 4
  - Figure 11
  - Figure 12
  - Figure 13

### 📊 Shapley Value Analysis

The Shapley values are precomputed and available for download [here on Zenodo](https://zenodo.org/records/15096721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIyM2FjMjg1LWVlZmMtNDIwOS1iZGU3LTdjMzhlNDY3YjIwYyIsImRhdGEiOnt9LCJyYW5kb20iOiIyMWE2NWU2YmU2NGY1YzVhZmM4ZWI2MWY3ODEwNDM0OSJ9.TSW3U7BAincAtI-P_tmI_CpBFjUkxRdAo2DQ9wK5TUUBd7YzG1cYi8uPXN74vZlNvxFRAzQMiRhKI1d290kepQ).

The associated files are in the `results` zipped folder and should be copied and unzipped into the corresponding `results` folder of this directory.

- **`shapley_global.ipynb`**: Computes the global Shapley values for:
  - Table 1
  - Figure 5
  - Figure 7
  - Figure 8

- **`shapley_local.ipynb`**: Computes the local Shapley values for:
  - Figure 6


## 📄 Additional files

- **`data_helpers.py`**: Provides functions for loading and processing data.  
- **`losses.py`**: Defines the loss functions used to train the models.  
- **`modules.py`**: Implements the PyTorch modules required to build MaskSDM.  
- **`train_model.py`**: Runs the training process for MaskSDM and baseline models. Training and model hyperparameters are defined here and can be adjusted as needed.  
- **`training_helpers.py`**: Includes utility functions to facilitate model training.  
