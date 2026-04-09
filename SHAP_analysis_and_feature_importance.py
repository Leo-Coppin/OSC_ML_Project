import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import os

# ================================
# CONFIG
# ================================
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DISPLAY = 10

OUTPUT_NAMES = [
    "FF",
    "Jsc",
    "PCE",
    "Voc",
    "delta_HOMO",
    "delta_LUMO"
]
N_OUTPUTS = len(OUTPUT_NAMES)

DESCRIPTORS = {
    "RDKit":   "Data_RDKit",
    "Mordred": "Data_Mordred",
    "MACCS":   "Data_MACCS",
    "Morgan":  "Data_Morgan",
    "PubChem": "Data_PubChem",
}

os.makedirs("SHAP_RESULTS", exist_ok=True)

# ================================
# MODEL (IDENTIQUE À L’ENTRAÎNEMENT)
# ================================
class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
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

# ================================
# LOOP DATASETS
# ================================
for dataset_name, prefix in DESCRIPTORS.items():

    print(f"\n✅ SHAP analysis — {dataset_name}")

    # -------- LOAD DATA --------
    X_train = pd.read_csv(f"{prefix}_train.csv", sep=';')
    X_test  = pd.read_csv(f"{prefix}_test.csv",  sep=';')

    # -------- FEATURE ALIGNMENT --------
    all_features = sorted(set(X_train.columns) | set(X_test.columns))
    X_train = X_train.reindex(columns=all_features, fill_value=0)
    X_test  = X_test.reindex(columns=all_features, fill_value=0)

    # -------- SCALING --------
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    X_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_te, dtype=torch.float32, device=device)

    # -------- LOAD MODEL --------
    model = ANN(X_tr.shape[1], N_OUTPUTS)
    model.load_state_dict(
        torch.load(f"ANN_{dataset_name}_best.pt", map_location=device)
    )
    model.eval()

    # -------- SHAP --------
    background = X_tr[:20]
    explained  = X_te[:100]

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(explained)

    X_plot = explained.cpu().numpy()

    # ================================
    # NORMALISE SHAP FORMAT
    # ================================
    if isinstance(shap_values, list):
        shap_by_output = shap_values
    else:
        shap_values = np.array(shap_values)

        if shap_values.ndim == 3:
            shap_by_output = [
                shap_values[:, :, i] for i in range(shap_values.shape[2])
            ]

        elif shap_values.ndim == 2 and shap_values.shape[0] == X_plot.shape[1]:
            shap_by_output = [
                np.tile(shap_values[:, i], (X_plot.shape[0], 1))
                for i in range(shap_values.shape[1])
            ]
        else:
            raise ValueError(f"Unhandled SHAP shape {shap_values.shape}")

    # ================================
    # FEATURE IMPORTANCE (BAR PLOTS)
    # ================================
    for i, output_name in enumerate(OUTPUT_NAMES):

        sv = np.array(shap_by_output[i])

        if sv.shape != X_plot.shape:
            if sv.T.shape == X_plot.shape:
                sv = sv.T
            else:
                raise ValueError(
                    f"SHAP shape mismatch for {dataset_name} / {output_name}: "
                    f"SHAP={sv.shape}, X={X_plot.shape}"
                )

        mean_abs_shap = np.mean(np.abs(sv), axis=0)

        df_imp = (
            pd.DataFrame({
                "feature": all_features,
                "mean_abs_shap": mean_abs_shap
            })
            .sort_values("mean_abs_shap", ascending=False)
            .head(MAX_DISPLAY)
        )

        plt.figure(figsize=(6, 4))
        plt.barh(
            df_imp["feature"][::-1],
            df_imp["mean_abs_shap"][::-1]
        )
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"{dataset_name} – {output_name}")
        plt.tight_layout()

        plt.savefig(
            f"SHAP_RESULTS/FeatureImportance_{dataset_name}_{output_name}.png",
            dpi=300
        )
        plt.close()

    # ================================
    # BEESWARM FIGURE (ALL OUTPUTS)
    # ================================
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(11, 12),
        sharex=True
    )

    axes = axes.flatten()

    for i, output_name in enumerate(OUTPUT_NAMES):

        plt.sca(axes[i])

        shap.summary_plot(
            shap_by_output[i],
            X_plot,
            feature_names=all_features,
            max_display=MAX_DISPLAY,
            show=False
        )

        axes[i].set_title(output_name, fontsize=11)
        axes[i].set_xlim(-2, 2)
        plt.yticks(fontsize=8)

    fig.suptitle(
        f"SHAP Analysis – {dataset_name}",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(
        f"SHAP_RESULTS/SHAP_{dataset_name}_ALL_OUTPUTS.png",
        dpi=300
    )
    plt.close()

    print(f"✅ {dataset_name} terminé")