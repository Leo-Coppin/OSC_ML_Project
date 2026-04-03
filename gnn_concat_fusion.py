import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Importation de tes modules locaux
from gnn_encoder import MolecularGNNEncoder, NODE_FEATURES, EDGE_FEATURES, EMBEDDING_DIM
from SMILES_to_Graph import load_dataset 

# =============================================================================
# 1. MODÈLE GNN CONCATENATION 
# =============================================================================

class GNNConcatFusion(nn.Module):
    def __init__(self, share_encoder=False, mlp_hidden=256, num_targets=6, dropout=0.2):
        super().__init__()

        # Encodeur GIN (Graphes)
        self.encoder_don = MolecularGNNEncoder(
            node_features=NODE_FEATURES,
            edge_features=EDGE_FEATURES,
            embedding_dim=EMBEDDING_DIM,
            dropout=dropout
        )
        
        # Si share_encoder est False, on crée un deuxième réseau pour l'accepteur
        self.encoder_acc = self.encoder_don if share_encoder else MolecularGNNEncoder(
            node_features=NODE_FEATURES,
            edge_features=EDGE_FEATURES,
            embedding_dim=EMBEDDING_DIM,
            dropout=dropout
        )

        fused_dim = 2 * EMBEDDING_DIM # Concaténation des deux vecteurs de 128

        # MLP de prédiction finale
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, num_targets)
        )

    def forward(self, don_data, acc_data):
        h_don = self.encoder_don(don_data) # Vecteur caractéristique donneur
        h_acc = self.encoder_acc(acc_data) # Vecteur caractéristique accepteur
        h_fused = torch.cat([h_don, h_acc], dim=1) # Fusion
        return self.mlp(h_fused)

# =============================================================================
# 2. DATASET & UTILS
# =============================================================================

class DonorAcceptorDataset(torch.utils.data.Dataset):
    def __init__(self, raw_subset):
        self.data = raw_subset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # On récupère les graphes et la cible normalisée
        return item['graph_donor'], item['graph_acceptor'], item['y_norm']

def collate_pairs(batch):
    donors, acceptors, targets = zip(*batch)
    return (
        Batch.from_data_list(list(donors)),
        Batch.from_data_list(list(acceptors)),
        torch.stack(targets)
    )

# =============================================================================
# 3. TRAINING & EVALUATION
# =============================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for don, acc, y in loader:
        don, acc, y = don.to(device), acc.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(don, acc)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for don, acc, y in loader:
        out = model(don.to(device), acc.to(device)).cpu()
        all_preds.append(out)
        all_targets.append(y)
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    mse = F.mse_loss(preds, targets).item()
    
    # Calcul du R2 par colonne (Target)
    ss_res = ((targets - preds) ** 2).sum(0)
    ss_tot = ((targets - targets.mean(0)) ** 2).sum(0)
    r2_scores = (1 - ss_res / (ss_tot + 1e-8)).numpy()
    
    return mse, r2_scores

# =============================================================================
# 4. MAIN EXECUTION 
# =============================================================================

if __name__ == "__main__":
    # Paramètres
    CSV_PATH = "Data.csv" 
    MASTER_SPLIT_PATH = "master_split.csv" # Généré par ton script de synchro
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_NAMES = ['Voc', 'Jsc', 'FF', 'PCE', 'delta_LUMO', 'delta_HOMO']

    # 1. Chargement des Graphes
    print("⏳ Étape 1 : Chargement et conversion des SMILES en graphes...")
    all_raw_data = load_dataset(CSV_PATH)
    
    # 2. Alignement sur le MASTER SPLIT (L'arbitre du duel)
    print(f"🎯 Étape 2 : Alignement sur {MASTER_SPLIT_PATH}...")
    master_split = pd.read_csv(MASTER_SPLIT_PATH, sep=";", index_col=0)
    
    # On ne garde que les indices validés par le script de synchro
    sync_data = [all_raw_data[i] for i in master_split.index]
    
    # 3. Normalisation des cibles (uniquement les 6 colonnes physiques)
    print("⚖️ Étape 3 : Normalisation des cibles...")
    all_y = np.array([d['y'].numpy() for d in sync_data])
    all_y = all_y[:, :6] # On ignore Target_CI (7ème colonne éventuelle)
    
    scaler = StandardScaler()
    all_y_norm = scaler.fit_transform(all_y)
    
    for i, data_dict in enumerate(sync_data):
        data_dict['y_norm'] = torch.tensor(all_y_norm[i], dtype=torch.float)
    
    joblib.dump(scaler, "scaler_gnn.pkl")

    # 4. Création des loaders (Train 70% / Test 30%)
    train_raw = [d for i, d in enumerate(sync_data) if master_split.iloc[i]['set'] == 'train']
    test_raw = [d for i, d in enumerate(sync_data) if master_split.iloc[i]['set'] == 'test']

    train_loader = DataLoader(DonorAcceptorDataset(train_raw), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairs)
    test_loader = DataLoader(DonorAcceptorDataset(test_raw), batch_size=BATCH_SIZE, collate_fn=collate_pairs)

    # 5. Initialisation et entraînement
    model = GNNConcatFusion(share_encoder=False, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print(f"🚀 Entraînement lancé sur {DEVICE}...")
    print(f"📊 Dataset : {len(train_raw)} Train | {len(test_raw)} Test")
    
    best_mse = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        test_mse, r2_list = evaluate(model, test_loader, DEVICE)
        scheduler.step(test_mse)
        
        if test_mse < best_mse:
            best_mse = test_mse
            torch.save(model.state_dict(), "best_gnn_model.pt")
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Test MSE: {test_mse:.4f} | R2 Moyen: {np.mean(r2_list):.3f}")

    # 6. Verdict Final
    print("\n" + "="*30)
    print("🏆 RÉSULTATS FINAUX DU GNN")
    print("="*30)
    model.load_state_dict(torch.load("best_gnn_model.pt"))
    _, final_r2 = evaluate(model, test_loader, DEVICE)
    
    for name, r2 in zip(TARGET_NAMES, final_r2):
        print(f"Propriété {name:<12} : R² = {r2:.3f}")
    
    print("-" * 30)
    print(f"Moyenne R² GNN : {np.mean(final_r2):.3f}")
    print(f"Baseline à battre (Mordred) : 0.387")
    print("="*30)