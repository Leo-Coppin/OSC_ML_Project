import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. Chargement des sorties
# ─────────────────────────────────────────────
outputs = ["Voc", "Jsc", "FF", "PCE", "delta_HOMO", "delta_LUMO"]

y_train = pd.read_csv("train_dataset.csv", sep=';')[outputs]
y_test  = pd.read_csv("test_dataset.csv",  sep=';')[outputs]

# ─────────────────────────────────────────────
# 2. Définition des paires de fichiers
# ─────────────────────────────────────────────
train_test_pairs = [
    ("Data_RDKit_train.csv",   "Data_RDKit_test.csv",   "rdkit"),
    ("Data_Mordred_train.csv", "Data_Mordred_test.csv", "mordred"),
    ("Data_Morgan_train.csv",  "Data_Morgan_test.csv",  "morgan"),
    ("Data_MACCS_train.csv",   "Data_MACCS_test.csv",   "MACCS"),
    ("Data_PubChem_train.csv", "Data_PubChem_test.csv", "Pubchem"),
]

# ─────────────────────────────────────────────
# 3. Hyperparamètres ANN
# ─────────────────────────────────────────────
EPOCHS       = 200
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
PATIENCE     = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}\n")

# ─────────────────────────────────────────────
# 4. Architecture ANN
# ─────────────────────────────────────────────
class ANN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.network(x)

# ─────────────────────────────────────────────
# 5. Prétraitement
# ─────────────────────────────────────────────
def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame, name: str):
    float32_max = np.finfo(np.float32).max

    if name == "rdkit":
        X_train = X_train.clip(-float32_max, float32_max)
        X_test  = X_test.clip(-float32_max, float32_max)  # ✅ correction

    if name == "mordred":
        for df in [X_train, X_test]:
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            for col in non_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        all_features = sorted(set(X_train.columns) | set(X_test.columns))
        X_train = X_train.reindex(columns=all_features, fill_value=0)
        X_test  = X_test.reindex(columns=all_features, fill_value=0)

        nan_cols = X_train.columns[X_train.isna().any()].tolist()
        X_train  = X_train.drop(columns=nan_cols)
        X_test   = X_test.drop(columns=[c for c in nan_cols if c in X_test.columns])

    return X_train, X_test

# ─────────────────────────────────────────────
# 6. Entraînement & évaluation
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, name):

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # ✅ Sécurité NaN / inf
    X_tr_sc = np.nan_to_num(X_tr_sc, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_sc = np.nan_to_num(X_te_sc, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr = torch.tensor(X_tr_sc, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    model = ANN(X_tr.shape[1], len(outputs)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss  = float('inf')
    patience_count = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_te), y_te).item()

        scheduler.step(val_loss)

        if epoch % 20 == 0:
            print(f"  [{name}] Epoch {epoch:>3}/{EPOCHS}  |  Train: {train_loss:.4f}  |  Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  [{name}] Early stopping à l'époque {epoch}.")
                break

    # ✅ Protection finale
    if best_state is None:
        print(f"  [{name}] ⚠️ Aucun état valide trouvé (val_loss = NaN).")
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_pred = model(X_te).cpu().numpy()

    y_true = y_test.values
    r2   = r2_score(y_true, y_pred, multioutput='raw_values')
    mae  = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))

    torch.save(model.state_dict(), f"ANN_{name}_best.pt")

    return pd.DataFrame({'R2': r2, 'MAE': mae, 'RMSE': rmse}, index=outputs)

# ─────────────────────────────────────────────
# 7. Boucle principale
# ─────────────────────────────────────────────
all_results = {}

for train_file, test_file, name in train_test_pairs:
    print(f"\n{'='*55}")
    print(f"  Descripteur : {name}")
    print(f"{'='*55}")

    X_train = pd.read_csv(train_file, sep=';', low_memory=False)
    X_test  = pd.read_csv(test_file,  sep=';', low_memory=False)

    X_train, X_test = preprocess(X_train, X_test, name)

    results = train_and_evaluate(X_train, X_test, y_train, y_test, name)
    all_results[name] = results

    print(f"\n  Résultats [{name}] :")
    print(results.round(4))

# ─────────────────────────────────────────────
# 8. Tableaux récapitulatifs
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
print("  RÉCAPITULATIF R² par descripteur")
print(f"{'='*55}")
summary_r2 = pd.DataFrame(
    {name: res['R2'] for name, res in all_results.items()},
    index=outputs
)
print(summary_r2.round(4))

print(f"\n{'='*55}")
print("  RÉCAPITULATIF RMSE par descripteur")
print(f"{'='*55}")
summary_rmse = pd.DataFrame(
    {name: res['RMSE'] for name, res in all_results.items()},
    index=outputs
)
print(summary_rmse.round(4))
