import os
import json
import torch
import optuna
import random
import numpy as np

from torch import nn
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from GNN_Cross_Attention import GNNCrossAttentionModel
from SMILES_to_Graph import load_dataset

# ==============================
# CONFIG
# ==============================

CSV_PATH = "Data_scaled.csv"

N_TRIALS = 150
EPOCHS = 200
PATIENCE = 20

BEST_VAL_GLOBAL = float("inf")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# ==============================
# REPRODUCTIBILITÉ
# ==============================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ==============================
# DATASET
# ==============================

print("Loading dataset...")
# dataset = load_dataset(CSV_PATH)

# train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
# #train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_data = load_dataset("train_dataset.csv")
test_data = load_dataset("test_dataset.csv")

print("Train:", len(train_data))
#print("Val:", len(val_data))
print("Test:", len(test_data))


# ==============================
# COLLATE FUNCTION
# ==============================

def collate_fn(batch):

    graph_don_list = []
    graph_acc_list = []
    targets = []

    for sample in batch:
        graph_don_list.append(sample["graph_donor"])
        graph_acc_list.append(sample["graph_acceptor"])
        targets.append(sample["y"])

    graph_don_batch = Batch.from_data_list(graph_don_list)
    graph_acc_batch = Batch.from_data_list(graph_acc_list)

    targets = torch.stack(targets)

    return graph_don_batch, graph_acc_batch, targets


# ==============================
# MSE PONDÉRÉE
# ==============================

class WeightedMSELoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights).to(DEVICE)

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        loss = loss * self.weights
        return loss.mean()


# poids OSC typiques
LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]

criterion = WeightedMSELoss(LOSS_WEIGHTS)


# ==============================
# TRAIN
# ==============================

def train_epoch(model, loader, optimizer):

    model.train()
    total_loss = 0

    for batch in loader:
        graph_don, graph_acc, y = batch
        graph_don = graph_don.to(DEVICE)
        graph_acc = graph_acc.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(graph_don, graph_acc)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ==============================
# VALIDATION
# ==============================

def validate_epoch(model, loader):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        for graph_don, graph_acc, y in loader:

            graph_don = graph_don.to(DEVICE)
            graph_acc = graph_acc.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(graph_don, graph_acc)

            loss = criterion(pred, y)

            total_loss += loss.item()

    return total_loss / len(loader)


# ==============================
# OBJECTIVE OPTUNA
# ==============================

def objective(trial):
    global BEST_VAL_GLOBAL
    heads = trial.suggest_int("num_attn_heads", 2, 8)
    head_dim = trial.suggest_int("hidden_dim", 32, 96, step=8)
    
    hidden_dim = heads * head_dim
    
    embedding_dim = trial.suggest_categorical("embedding_dim",[32,64,128, 256])
    num_layers = trial.suggest_int("num_gnn_layers", 2, 5)
    
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    model = GNNCrossAttentionModel(
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_gnn_layers=num_layers,
        num_attn_heads=heads,
        dropout=dropout,
        num_outputs=6,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate_epoch(model, val_loader)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val:
            best_val = val_loss

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_val <BEST_VAL_GLOBAL:
        BEST_VAL_GLOBAL = best_val
        torch.save({
                "model_state_dict": model.state_dict(),
                "params": trial.params,
                "val_loss": val_loss,
                "epoch": epoch
            }, "best_GNN1.pt")
        print(f"\nNew best model saved with val_loss={best_val:.4f} at epoch {epoch} with params: {trial.params}")

    return best_val


# ==============================
# RUN OPTUNA
# ==============================

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=50,
        interval_steps=5
    )
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\nBest parameters")
print(study.best_params)

# sauvegarde best params
# with open("best_params.json", "w") as f:
    # json.dump(study.best_params, f, indent=4)
    
  
# test model

# dataset = load_dataset(CSV_PATH)

# train_size = int(0.8 * len(dataset))
# val_size = int(0.1 * len(dataset))
# test_size = len(dataset) - train_size - val_size

# train_data = dataset[:train_size]
# val_data = dataset[train_size:train_size+val_size]
# test_data = dataset[train_size+val_size:]



test_loader = DataLoader(
    test_data,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn
)

print("Test samples:", len(test_data))


# =============================
# LOAD MODEL
# =============================

checkpoint = torch.load("best_GNN1.pt", map_location=DEVICE)

params = checkpoint["params"]

print("\nBest parameters:")
print(params)

model = GNNCrossAttentionModel(
    hidden_dim= params["hidden_dim"] * params["num_attn_heads"],
    embedding_dim=params["embedding_dim"],
    num_gnn_layers=params["num_gnn_layers"],
    num_attn_heads=params["num_attn_heads"],
    dropout=params["dropout"],
    num_outputs=6,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("\nModel loaded successfully")


# =============================
# TEST
# =============================

all_preds = []
all_targets = []

with torch.no_grad():

    for graph_don, graph_acc, y in test_loader:

        graph_don = graph_don.to(DEVICE)
        graph_acc = graph_acc.to(DEVICE)
        y = y.to(DEVICE)

        preds = model(graph_don, graph_acc)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)


# =============================
# METRICS
# =============================

mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print("\n===== GLOBAL METRICS =====")
print("MSE:", mse)
print("MAE:", mae)
print("R2 :", r2)


# =============================
# METRICS PAR PROPRIÉTÉ
# =============================

names = ["PCE", "Voc", "Jsc", "FF", "dHOMO", "dLUMO"]

print("\n===== METRICS PER PROPERTY =====")

for i in range(6):

    mse_i = mean_squared_error(all_targets[:, i], all_preds[:, i])
    mae_i = mean_absolute_error(all_targets[:, i], all_preds[:, i])
    r2_i = r2_score(all_targets[:, i], all_preds[:, i])

    print(f"\n{names[i]}")
    print("MSE:", mse_i)
    print("MAE:", mae_i)
    print("R2 :", r2_i)
