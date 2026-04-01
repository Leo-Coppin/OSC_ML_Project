"""
Encodeur GNN pour molécules donneur/accepteur
Architecture : GIN (Graph Isomorphism Network)
Entrée  : graphe moléculaire (torch_geometric.data.Data)
Sortie  : vecteur embedding de 128 dimensions par molécule

Dimensions issues de smiles_to_graph.py :
    - features atomes (x)      : 11 dimensions
    - features liaisons (edge_attr) : 6 dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from SMILES_to_Graph import load_dataset


# =============================================================================
# DIMENSIONS
# =============================================================================

NODE_FEATURES = 11   # taille du vecteur x par atome
EDGE_FEATURES  = 6   # taille du vecteur edge_attr par liaison
HIDDEN_DIM     = 64  # dimension des couches cachées du GNN
EMBEDDING_DIM  = 128 # dimension finale de l'embedding


# =============================================================================
# 1. ENCODEUR GNN (GIN)
# =============================================================================

class MolecularGNNEncoder(nn.Module):
    """
    Encodeur GNN basé sur GIN (Graph Isomorphism Network).

    Prend en entrée un graphe moléculaire et produit un vecteur
    embedding de taille EMBEDDING_DIM qui résume la molécule entière.

    Architecture :
        - 3 couches GINEConv (GIN avec features d'arêtes)
        - Batch Normalization après chaque couche
        - Global pooling (mean + add) pour agréger les atomes
        - Couche linéaire finale pour projeter vers EMBEDDING_DIM
    """

    def __init__(
        self,
        node_features: int = NODE_FEATURES,
        edge_features: int = EDGE_FEATURES,
        hidden_dim:    int = HIDDEN_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        num_layers:    int = 3,
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.num_layers    = num_layers
        self.dropout       = dropout
        self.embedding_dim = embedding_dim

        # --- Projection initiale des features d'arêtes ---
        # GINEConv exige que edge_attr ait la même dimension que les noeuds
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        # --- Couches GINEConv ---
        # Chaque couche GINEConv contient un petit réseau MLP interne
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = node_features if layer == 0 else hidden_dim

            # MLP interne de GIN : 2 couches linéaires avec activation
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # --- Projection finale vers l'embedding ---
        # On concatène mean pooling + add pooling → hidden_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Calcule l'embedding d'une molécule.

        """

        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = data.batch

        # Projection des features d'arêtes vers hidden_dim
        edge_attr = self.edge_encoder(edge_attr)

        # Passage dans les couches GINEConv
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling : agrège tous les atomes en un seul vecteur
        # On utilise mean + add pour capturer à la fois la moyenne et la somme
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        x_add  = global_add_pool(x, batch)   # [batch_size, hidden_dim]
        x_pool = torch.cat([x_mean, x_add], dim=1)  # [batch_size, hidden_dim * 2]

        # Projection finale
        embedding = self.projection(x_pool)  # [batch_size, embedding_dim]

        return embedding


# =============================================================================
# 2. VÉRIFICATION RAPIDE
# =============================================================================

def verify_encoder(dataset, index=0):
    """
    Teste l'encodeur sur un exemple du dataset.
    Affiche les dimensions de l'embedding produit.
    """

    print("\n=== Vérification de l'encodeur GNN ===")

    # Initialisation de l'encodeur
    encoder = MolecularGNNEncoder(
        node_features = NODE_FEATURES,
        edge_features = EDGE_FEATURES,
        hidden_dim    = HIDDEN_DIM,
        embedding_dim = EMBEDDING_DIM,
        num_layers    = 3,
        dropout       = 0.1,
    )

    # Passage en mode évaluation (désactive le dropout)
    encoder.eval()

    sample = dataset[index]

    # On ajoute le batch index manuellement (batch de taille 1)
    graph_don = sample['graph_donor']
    graph_acc = sample['graph_acceptor']
    graph_don.batch = torch.zeros(graph_don.x.shape[0], dtype=torch.long)
    graph_acc.batch = torch.zeros(graph_acc.x.shape[0], dtype=torch.long)

    with torch.no_grad():
        emb_don = encoder(graph_don)
        emb_acc = encoder(graph_acc)

    print(f"\nExemple n°{index}")
    print(f"  Graphe donneur   : {graph_don.x.shape[0]} atomes")
    print(f"  Graphe accepteur : {graph_acc.x.shape[0]} atomes")
    print(f"\n  Embedding donneur   : {emb_don.shape} → {emb_don.shape[1]} dimensions ✅")
    print(f"  Embedding accepteur : {emb_acc.shape} → {emb_acc.shape[1]} dimensions ✅")
    print(f"\n  Valeurs embedding donneur   (5 premières) : {emb_don[0, :5].tolist()}")
    print(f"  Valeurs embedding accepteur (5 premières) : {emb_acc[0, :5].tolist()}")
   
    print("\n✅ Encodeur fonctionnel !")
    print(f"   Chaque molécule produit un vecteur de {EMBEDDING_DIM} dimensions.")

    return encoder


def print_model_summary(encoder):
    """Affiche le nombre de paramètres de l'encodeur."""

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable    = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"\n=== Résumé du modèle ===")
    print(f"  Paramètres totaux     : {total_params:,}")
    print(f"  Paramètres entraînables : {trainable:,}")
    print(f"  Couches GINEConv      : {encoder.num_layers}")
    print(f"  Dimension cachée      : {HIDDEN_DIM}")
    print(f"  Dimension embedding   : {EMBEDDING_DIM}")


# =============================================================================
# 3. POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":

    CSV_PATH = "CSV_PATH"

    print("Chargement du dataset...")
    dataset = load_dataset(CSV_PATH)

    # Vérification de l'encodeur sur 1 exemple
    encoder = verify_encoder(dataset, index=0)


    # Résumé du modèle
    print_model_summary(encoder)
