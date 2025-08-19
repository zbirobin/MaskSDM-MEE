import torch
import numpy as np

from training_helpers import seed_everything, train
from data_helpers import get_data, get_split_indices

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Set random seed for reproducibility
random_seed = 42
seed_everything(random_seed)

# Load data
dataset = "splot" # "geoplant" # "splot"
data = get_data(dataset)

 # Select species with more than min_num_obs observations
min_num_obs = 20
data["y"] = data["y"][:, data["y"].sum(axis=0) > min_num_obs]
n_features = data["tabular_x"].shape[1]
n_samples, n_species = data["y"].shape

# Split data into train, val, and test sets
split_mode = "extrapolate"
spacing = 5
train_indices, val_indices, test_indices = get_split_indices(data, mode=split_mode, spacing=spacing)
data["x_train"], data["y_train"] = data["tabular_x"][train_indices], data["y"][train_indices]
data["x_val"], data["y_val"] = data["tabular_x"][val_indices], data["y"][val_indices]
data["x_test"], data["y_test"] = data["tabular_x"][test_indices], data["y"][test_indices]
data["satclip_embeddings_train"] = data["satclip_embeddings"][train_indices]
data["satclip_embeddings_val"] = data["satclip_embeddings"][val_indices]
data["satclip_embeddings_test"] = data["satclip_embeddings"][test_indices]

# Normalize tabular data
train_mean = np.nanmean(data["x_train"], axis=0)
train_std = np.nanstd(data["x_train"], axis=0)
data["x_train"] = (data["x_train"] - train_mean)/(train_std + 0.0001)
data["x_val"] = (data["x_val"] - train_mean)/(train_std + 0.0001)
data["x_test"] = (data["x_test"] - train_mean)/(train_std + 0.0001)

# Enable masking of missing values using the Mask Token in MaskSDM
masking = True

# Enable masked data modeling during training with MaskSDM
extra_masking = True

# For experiments with Oracle: put NaN to excluded values
#data["x_train"][:, 27:29] = np.nan
#data["x_val"][:, 27:29] = np.nan
#data["x_train"][:, 32:52] = np.nan
#data["x_val"][:, 32:52] = np.nan
#data["satclip_embeddings_train"] = np.tile(data["satclip_embeddings_train"].mean(axis=0, keepdims=True), (len(data["satclip_embeddings_train"]), 1))
#data["satclip_embeddings_val"] = np.tile(data["satclip_embeddings_val"].mean(axis=0, keepdims=True), (len(data["satclip_embeddings_val"]), 1))
if not masking:
    data["x_train"] = np.nan_to_num(data["x_train"], nan=0) # The mean of each feature is 0
    data["x_val"]  = np.nan_to_num(data["x_val"] , nan=0)
    data["x_test"] = np.nan_to_num(data["x_test"], nan=0)
    # medians = np.nanmedian(data["x_train"], axis=0)
    # data["x_train"] = np.where(np.isnan(data["x_train"]), medians, data["x_train"])
    # data["x_val"] = np.where(np.isnan(data["x_val"]), medians, data["x_val"])
    # data["x_test"] = np.where(np.isnan(data["x_test"]), medians, data["x_test"])

# Set up the configuration dictionary
# This dictionary contains all the hyperparameters and settings for the model and training process
config = {}

config["device"] = device
config["seed"] = random_seed

config["dataset"] = dataset
config["n_features"] = n_features
config["n_species"] = n_species
config["n_samples_train"] = len(data["y_train"])
config["n_samples_val"] = len(data["y_val"])
config["n_samples_test"] = len(data["y_test"])
config["split_mode"] = split_mode
config["spacing"] = spacing
config["min_num_obs"] = min_num_obs
config["indices_evaluated_species"] = np.intersect1d(np.intersect1d(np.sum(data["y_train"], axis=0).nonzero()[0], np.sum(data["y_val"], axis=0).nonzero()[0]),
                                                            np.sum(data["y_test"], axis=0).nonzero()[0]).tolist()
config["n_evaluated_species"] = len(config["indices_evaluated_species"])
config["satclip"] = True

config["model"] = "FTTransformer" # Can be "FTTransformer", "MLP", "ResNet", "linear", "Maxent"
config["d_hidden"] = 192
config["n_heads"] = 8
config["n_blocks"] = 7 # For FTTransformer
config["n_layers"] = 7 # For MLP and ResNet baselines
config["dropout"] = 0.1
config["d_out"] = n_species

config["epochs"] = 1000
config["batch_size"] = 256
config["batch_size_eval"] = 4096
config["loss"] = "weighted"
config["species_weights"] = torch.tensor(len(data["y_train"])/(np.sum(data["y_train"], axis=0) + 1e-5), 
                                         dtype=torch.float32).to(config["device"])
config["optimizer"] = "AdamW"
config["scheduler_free"] = True
config["lr"] = 0.001
config["weight_decay"] = 0.01
config["warmup_steps"] = 1000

config["masking"] = masking
config["extra_masking"] = extra_masking

# Saving and logging
config["save_dir"] = "models/masksdm"
config["use_wandb"] = False
config["wandb_init"] = {
    "project": "MaskSDM_final",
    "entity": "TOFILL",
    "name": "MaskSDM"
}
    
train(config, data)
