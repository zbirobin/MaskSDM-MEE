import torch


def get_loss_fn(config):
    
    if config["loss"] == "BCE":
        return get_BCE(config)
    elif config["loss"] == "weighted":
        return weighted_loss(config["species_weights"])
    else:
        raise ValueError(f"Unknown loss: {config['loss']}")
    

def get_BCE(config):
    pos_weight = torch.tensor(config["pos_weight"])
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def weighted_loss(species_weights):
    """ Get the weighted loss function of Zbinden et al. (2024)."""
    
    def loss_fn(y_pred, y_true):
        batch_size = y_pred.size(0)
        
        y_pred = torch.sigmoid(y_pred)
        
        loss_pos = (log_loss(y_pred) * y_true * species_weights.unsqueeze(0).repeat(batch_size, 1)).mean()
        loss_neg = (log_loss(1 - y_pred) * (1 - y_true) * (species_weights/(species_weights - 1)).unsqueeze(0).repeat(batch_size, 1)).mean()
        return loss_pos + loss_neg
    
    return loss_fn
    
    
def log_loss(pred):
    """Helper function."""
    return -torch.log(pred + 1e-5)