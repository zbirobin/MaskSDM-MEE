import random
import os

import numpy as np
import torch
import wandb

from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_auroc
from schedulefree import AdamWScheduleFree 

from losses import get_loss_fn
from data_helpers import get_torch_dataset
from modules import get_model


def seed_everything(seed=42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # Numpy seed also uses by Scikit Learn
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train(config, data):
    
    train_loader = DataLoader(get_torch_dataset(config, data["x_train"], data["y_train"], data["satclip_embeddings_train"]), 
                              batch_size=config["batch_size"], shuffle=True, generator=torch.Generator(device=config["device"]))
    val_loader = DataLoader(get_torch_dataset(config, data["x_val"], data["y_val"], data["satclip_embeddings_val"]),
                            batch_size=4096, shuffle=False)
    test_loader = DataLoader(get_torch_dataset(config, data["x_test"], data["y_test"], data["satclip_embeddings_test"]),
                            batch_size=4096, shuffle=False)
    
    model = get_model(config).to(config["device"])
    
    loss_fn = get_loss_fn(config)
    optimizer = get_optimizer(config, model)
    model.train()
    
    if config["use_wandb"]:
        wandb.init(project="MaskSDM_final", entity="TOFILL", name=f"MaskSDM", config=config)

    best_val_auc = 0
    for epoch in range(1, config["epochs"]+1):
        
        model.train()
        optimizer.train()
        
        for batch in train_loader:
            
            optimizer.zero_grad()
            x_batch, y_batch, satclip_embeddings = batch
            x_batch = x_batch.to(config["device"])
            y_batch = y_batch.type(torch.FloatTensor).to(config["device"])
            satclip_embeddings = satclip_embeddings.to(config["device"])

            if config["masking"]:
                nan_mask = ~x_batch.isnan() # Mask nan values (mask = 0)
                if config["extra_masking"]:
                    mask_prob = np.random.uniform(0.0, 1.0)
                    random_mask = torch.bernoulli(torch.full(x_batch.shape, mask_prob)).to(config["device"])
                    x_mask = torch.logical_and(random_mask, nan_mask) # Masked values are equal to 0
                    satclip_embedding_mask = torch.bernoulli(torch.full((len(satclip_embeddings),), mask_prob)).to(config["device"])
                else:
                    x_mask = nan_mask
                    satclip_embedding_mask = torch.ones(len(satclip_embeddings), dtype=torch.bool).to(config["device"])
                if not config["satclip"]:
                    satclip_embedding_mask = torch.zeros(len(satclip_embeddings), dtype=torch.bool).to(config["device"])
                y_pred = model(x_batch, satclip_embeddings, x_mask.to(config["device"]), satclip_embedding_mask)
            else:
                y_pred = model(x_batch, satclip_embeddings)
            
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

        model.eval()
        optimizer.eval()
        
        with torch.no_grad():
            
            val_auc = evaluate(config, model, val_loader)
            print(f"Epoch {epoch}, val AUC: {val_auc:3f}")
            
            if best_val_auc < val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), f"models/MaskSDM_{epoch}.pt")
            
            if config["use_wandb"]:
                val_auc_temp = evaluate(config, model, val_loader, excluded_variables=range(1, config["n_features"]))
                val_auc_prec = evaluate(config, model, val_loader, excluded_variables=list(range(11)) + list(range(12, config["n_features"])))
                val_auc_wc = evaluate(config, model, val_loader, excluded_variables=range(19, config["n_features"]))
                val_auc_sg = evaluate(config, model, val_loader, excluded_variables=list(range(19)) + list(range(27, config["n_features"])))
                val_auc_loc = evaluate(config, model, val_loader, excluded_variables=list(range(27)) + list(range(29, config["n_features"])))
                val_auc_topo = evaluate(config, model, val_loader, excluded_variables=list(range(29)) + list(range(31, config["n_features"])))
                wandb.log({"val_auc": val_auc, "val_auc_temp": val_auc_temp, "val_auc_prec": val_auc_prec, "val_auc_sg": val_auc_sg,
                           "val_auc_wc": val_auc_wc, "val_auc_loc": val_auc_loc, "val_auc_topo": val_auc_topo})
                
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"models/MaskSDM_{epoch}.pt")
        
    if config["use_wandb"]:     
        wandb.finish()

    return model

   
def get_optimizer(config, model):
    
    if config["optimizer"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "AdamW":
        if config["scheduler_free"]:
            return AdamWScheduleFree(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], warmup_steps=config["warmup_steps"])
        else:
            return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def evaluate(config, model, dataloader, excluded_variables=None):
    
    preds = []
    y = []
    for batch in dataloader:
        
        x_batch, y_batch, satclip_embeddings = batch
        
        x_mask = (~torch.isnan(x_batch))
        if excluded_variables is not None:
            x_mask[:, excluded_variables] = 0
        x_mask = x_mask.to(config["device"])
        x_batch = x_batch.to(config["device"])
        y.append(y_batch)
        satclip_embeddings = satclip_embeddings.to(config["device"])

        if config["masking"]:
            if config["satclip"] and excluded_variables is None:
                satclip_embeddings_mask = torch.tensor(np.ones(len(satclip_embeddings))).to(config["device"])
            else:
                satclip_embeddings_mask = torch.tensor(np.zeros(len(satclip_embeddings))).to(config["device"])
            y_pred = model(x_batch, satclip_embeddings, x_mask, satclip_embeddings_mask)
        else:
                y_pred = model(x_batch, satclip_embeddings)
        
        preds.append(y_pred)
    
    preds = torch.concatenate(preds, axis=0).float()
    y = torch.concatenate(y, axis=0).to(config["device"]).int()
    model.train()
        
    considered_species = config["indices_evaluated_species"]
    auc = binary_auroc(preds[:, considered_species].T, y[:, considered_species].T, num_tasks=len(considered_species)).mean()
        
    return auc