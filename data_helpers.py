import pandas as pd
import numpy as np
import verde as vd

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


def get_data(dataset):
    """ Load data for the given dataset. """
    
    if dataset == "splot":        
        return get_splot_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_splot_data():
    """ Load the sPlot dataset. """
        
    worldclim_data = pd.read_csv("data/worldclim_data.csv", index_col=0).set_index("PlotObservationID")
    soilgrid_data = pd.read_csv("data/soilgrid_data.csv", index_col=0).set_index("PlotObservationID")
    location_data = pd.read_csv("data/location_data.csv", index_col=0).set_index("PlotObservationID")
    topographic_data = pd.read_csv("data/topographic_data.csv", index_col=0).set_index("PlotObservationID")
    metadata_data = pd.read_csv("data/metadata_data.csv", index_col=0).set_index("PlotObservationID")
    cover_data = pd.read_csv("data/cover_data.csv", index_col=0).set_index("PlotObservationID")
    height_data = pd.read_csv("data/height_data.csv", index_col=0).set_index("PlotObservationID")
    human_data = pd.read_csv("data/human_data.csv", index_col=0).set_index("PlotObservationID")
        
    satclip_embeddings = np.load("data/satclip_embeddings.npy").astype(np.float32)

    tabular_x = pd.concat([worldclim_data, soilgrid_data, location_data, topographic_data, metadata_data, cover_data, height_data, human_data], axis=1)    
    tabular_names = list(tabular_x.columns)
    tabular_x = tabular_x.to_numpy().astype(np.float32)
    
    species_occurrences = np.load("data/species_occurrences.npy")
    
    data = {}
    data["tabular_x"] = tabular_x
    data["satclip_embeddings"] = satclip_embeddings
    data["tabular_names"] = tabular_names
    data["y"] = species_occurrences
    
    return data


def get_torch_dataset(config, x, y, satclip_embeddings):
    """ Get a torch dataset for the given configuration."""
    
    if config["dataset"] == "splot":
        return SplotDataset(x, y, satclip_embeddings)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    
def get_split_indices(data, mode="extrapolate"):
    """ Get the indices for the train, validation and test sets."""
    
    test_size = 0.15
    val_size = 0.15
    
    spacing = 1
    
    split_seed = 42
    
    i_lon = data["tabular_names"].index("Longitude")
    i_lat = data["tabular_names"].index("Latitude")
    coordinates = data["tabular_x"][:, [i_lon, i_lat]]

    data_indices = np.arange(len(data["tabular_x"]))
    
    if mode == "interpolate":
        
        train_indices, test_indices = train_test_split(data_indices, test_size=test_size)
        train_indices, val_indices = train_test_split(train_indices, test_size=val_size/(1-test_size))
        
    elif mode == "extrapolate":
        
        train_block, test_block = vd.train_test_split(
            coordinates.transpose(),
            data_indices,
            spacing=spacing,
            test_size=test_size,
            random_state=split_seed,
        )
        train_indices, test_indices = train_block[1][0], test_block[1][0]

        train_block, val_block = vd.train_test_split(
            coordinates[train_indices].transpose(),
            train_indices,
            spacing=spacing,
            test_size=val_size/(1-test_size),
            random_state=split_seed,
        )
        train_indices, val_indices = train_block[1][0], val_block[1][0]
        
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    
    return train_indices, val_indices, test_indices
    

class SplotDataset(Dataset):

    def __init__(self, x, y, satclip_embeddings):
        self.x = x
        self.y = y
        self.satclip_embeddings = satclip_embeddings
        self.length = len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.satclip_embeddings[idx]
    
    def __len__(self):
        return self.length