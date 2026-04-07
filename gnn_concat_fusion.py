import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from gnn_encoder import MolecularGNNEncoder, NODE_FEATURES, EDGE_FEATURES, EMBEDDING_DIM
from SMILES_to_Graph import load_dataset 

# =============================================================================
# 1. MODÈLE GNN CONCATENATION 
# =============================================================================

class GNNConcatFusion(nn.Module):
    def __init__(self, share_encoder=False, mlp_hidden=256, num_targets=6, dropout=0.3):
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
            
            nn.Linear(mlp_hidden, mlp_hidden), 
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
    # On utilise maintenant les fichiers générés par ton script de préparation
    TRAIN_CSV = "train_dataset.csv"
    TEST_CSV = "test_dataset.csv"
    
    BATCH_SIZE = 32
    LR = 5e-4
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_NAMES = ['Voc', 'Jsc', 'FF', 'PCE', 'delta_HOMO', 'delta_LUMO']

    # 1. Chargement et conversion des SMILES en graphes
    print("⏳ Étape 1 : Chargement des datasets séparés...")
    train_raw = load_dataset(TRAIN_CSV)
    test_raw = load_dataset(TEST_CSV)
    
    # 2. Normalisation des cibles
    # On concatène temporairement pour fitter le scaler sur l'ensemble du TRAIN
    print("⚖️ Étape 2 : Normalisation des cibles...")
    
    train_y = np.array([d['y'].numpy() for d in train_raw])
    test_y = np.array([d['y'].numpy() for d in test_raw])
    
    scaler = StandardScaler()
    # On fit le scaler UNIQUEMENT sur le train pour éviter le data leakage
    train_y_norm = scaler.fit_transform(train_y)
    test_y_norm = scaler.transform(test_y)
    
    # Ré-injection des cibles normalisées dans les dictionnaires
    for i, data_dict in enumerate(train_raw):
        data_dict['y_norm'] = torch.tensor(train_y_norm[i], dtype=torch.float)
    for i, data_dict in enumerate(test_raw):
        data_dict['y_norm'] = torch.tensor(test_y_norm[i], dtype=torch.float)
    
    # joblib.dump(scaler, "scaler_gnn.pkl")

    # 3. Création des loaders
    train_loader = DataLoader(DonorAcceptorDataset(train_raw), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairs)
    test_loader = DataLoader(DonorAcceptorDataset(test_raw), batch_size=BATCH_SIZE, collate_fn=collate_pairs)

    # 4. Initialisation du modèle 
    model = GNNConcatFusion(share_encoder=False, mlp_hidden=256, dropout=0.5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
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
            torch.save(model.state_dict(), "best_gnn_concat_model.pt")
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Test MSE: {test_mse:.4f} | R2 Moyen: {np.mean(r2_list):.3f}")

    # 5. Verdict Final
    print("\n" + "="*30)
    print("🏆 RÉSULTATS FINAUX DU GNN")
    print("="*30)
    # On recharge le meilleur modèle sauvegardé
    model.load_state_dict(torch.load("best_gnn_concat_model.pt"))
    _, final_r2 = evaluate(model, test_loader, DEVICE)
    
    for name, r2 in zip(TARGET_NAMES, final_r2):
        print(f"Propriété {name:<12} : R² = {r2:.3f}")
    
    print("-" * 30)
    print(f"Moyenne R² GNN : {np.mean(final_r2):.3f}")
    print("="*30)