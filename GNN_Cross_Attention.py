"""
GNN avec Cross-Attention pour molécules donneur/accepteur OSC
Architecture :
    1. GNN Encoder partagé → embeddings par atome pour don et acc
    2. Cross-Attention     → chaque atome don attend aux atomes acc (et vice versa)
    3. Global Pooling      → vecteur global par molécule
    4. Feed-Forward        → prédiction PCE, Voc, Jsc, FF, ΔHOMO, ΔLUMO
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch

from gnn_encoder import MolecularGNNEncoder, NODE_FEATURES, EDGE_FEATURES, HIDDEN_DIM, EMBEDDING_DIM
from SMILES_to_Graph import load_dataset

import optuna

# =============================================================================
# 1. MODULE CROSS-ATTENTION
# =============================================================================

class CrossAttention(nn.Module):
    """
    Cross-Attention entre les atomes du donneur et de l'accepteur.

    Pour chaque atome du donneur, calcule un score d'attention
    avec tous les atomes de l'accepteur, et vice versa.

    Paramètres :
        embed_dim  : dimension des embeddings d'atomes (= HIDDEN_DIM du GNN)
        num_heads  : nombre de têtes d'attention (multi-head attention)
        dropout    : taux de dropout sur les scores d'attention
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) doit être divisible par num_heads ({num_heads})"

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5   # 1/√d pour stabiliser les gradients

        # Projections linéaires pour Q, K, V
        # Don → Query, Acc → Key et Value (direction don→acc)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Projection de sortie
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Layer norm pour stabiliser l'entraînement
        self.norm_don = nn.LayerNorm(embed_dim)
        self.norm_acc = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x_don: torch.Tensor,    # [N_don, embed_dim] embeddings atomes donneur
        x_acc: torch.Tensor,    # [N_acc, embed_dim] embeddings atomes accepteur
        batch_don: torch.Tensor,  # [N_don] indices de batch pour le donneur
        batch_acc: torch.Tensor,  # [N_acc] indices de batch pour l'accepteur
    ):
        """
        Calcule le cross-attention moléculaire par paire dans le batch.

        Pour chaque paire (donneur_i, accepteur_i) dans le batch :
            - Les atomes du donneur attendant aux atomes de l'accepteur
            - Les atomes de l'accepteur attendant aux atomes du donneur

        Retourne les embeddings enrichis x_don et x_acc après attention.
        """

        batch_size = batch_don.max().item() + 1

        enriched_don_list = []
        enriched_acc_list = []

        # Traitement molécule par molécule dans le batch
        for b in range(batch_size):

            # Extraction des atomes de la molécule b
            mask_don = (batch_don == b)
            mask_acc = (batch_acc == b)

            h_don = x_don[mask_don]   # [N_don_b, embed_dim]
            h_acc = x_acc[mask_acc]   # [N_acc_b, embed_dim]

            # ---- Direction Don → Acc ----
            # Le donneur pose des questions sur l'accepteur
            enriched_don = self._single_cross_attention(
                query=h_don,   # atomes du donneur posent les questions
                key=h_acc,     # atomes de l'accepteur fournissent les clés
                value=h_acc,   # atomes de l'accepteur fournissent les valeurs
                residual=h_don
            )

            # ---- Direction Acc → Don ----
            # L'accepteur pose des questions sur le donneur
            enriched_acc = self._single_cross_attention(
                query=h_acc,
                key=h_don,
                value=h_don,
                residual=h_acc
            )

            enriched_don_list.append(enriched_don)
            enriched_acc_list.append(enriched_acc)

        # Reconstruction des tenseurs complets
        x_don_enriched = torch.cat(enriched_don_list, dim=0)
        x_acc_enriched = torch.cat(enriched_acc_list, dim=0)

        return x_don_enriched, x_acc_enriched

    def _single_cross_attention(
        self,
        query:    torch.Tensor,   # [N_q, embed_dim]
        key:      torch.Tensor,   # [N_k, embed_dim]
        value:    torch.Tensor,   # [N_k, embed_dim]
        residual: torch.Tensor,   # [N_q, embed_dim] pour la connexion résiduelle
    ) -> torch.Tensor:
        """
        Cross-attention simple entre query et key/value.
        Utilise une connexion résiduelle + LayerNorm (style Transformer).
        """

        N_q = query.shape[0]
        N_k = key.shape[0]

        # Projections Q, K, V
        Q = self.q_proj(query)    # [N_q, embed_dim]
        K = self.k_proj(key)      # [N_k, embed_dim]
        V = self.v_proj(value)    # [N_k, embed_dim]

        # Reshape pour multi-head : [N, embed_dim] → [num_heads, N, head_dim]
        Q = Q.view(N_q, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(N_k, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(N_k, self.num_heads, self.head_dim).transpose(0, 1)

        # Scores d'attention : [num_heads, N_q, N_k]
        # Chaque ligne = distribution d'attention d'un atome query sur tous les atomes key
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Softmax sur la dimension des clés → poids d'attention somment à 1
        attn_weights = F.softmax(scores, dim=-1)    # [num_heads, N_q, N_k]
        attn_weights = self.dropout(attn_weights)

        # Agrégation pondérée des valeurs
        attended = torch.matmul(attn_weights, V)    # [num_heads, N_q, head_dim]

        # Reshape inverse : [num_heads, N_q, head_dim] → [N_q, embed_dim]
        attended = attended.transpose(0, 1).contiguous().view(N_q, self.embed_dim)

        # Projection de sortie
        attended = self.out_proj(attended)

        # Connexion résiduelle + LayerNorm (style Post-Norm)
        # Permet un entraînement stable et évite la disparition du gradient
        output = self.norm_don(residual + attended)

        return output


# =============================================================================
# 2. MODÈLE COMPLET GNN + CROSS-ATTENTION
# =============================================================================

class GNNCrossAttentionModel(nn.Module):
    """
    Modèle complet pour prédiction multi-output sur paires donneur-accepteur.

    Pipeline :
        1. GNN partagé → embeddings par atome (don et acc utilisent le même encodeur)
        2. Cross-Attention → enrichissement mutuel des embeddings d'atomes
        3. Global Pooling → vecteur global par molécule (mean + add)
        4. Concaténation → vecteur de 256 dimensions (128 don + 128 acc)
        5. Feed-Forward → 6 prédictions (PCE, Voc, Jsc, FF, ΔHOMO, ΔLUMO)

    Note sur le GNN partagé :
        Donneur et accepteur utilisent le MÊME encodeur GNN (poids partagés).
        Cela force le modèle à apprendre une représentation universelle des
        molécules organiques plutôt que des représentations spécialisées.
        C'est la pratique standard dans la littérature sur les OSC.
    """

    def __init__(
        self,
        node_features:  int   = NODE_FEATURES,
        edge_features:  int   = EDGE_FEATURES,
        hidden_dim:     int   = HIDDEN_DIM,
        embedding_dim:  int   = EMBEDDING_DIM,
        num_gnn_layers: int   = 3,
        num_attn_heads: int   = 4,
        dropout:        float = 0.1,
        num_outputs:    int   = 6,     # PCE, Voc, Jsc, FF, ΔHOMO, ΔLUMO
    ):
        super().__init__()

        self.hidden_dim    = hidden_dim
        self.embedding_dim = embedding_dim

        # --- 1. Encodeur GNN partagé ---
        # On réutilise MolecularGNNEncoder mais on récupère les embeddings
        # par atome AVANT le pooling global
        self.gnn = _GNNBackbone(
            node_features = node_features,
            edge_features = edge_features,
            hidden_dim    = hidden_dim,
            num_layers    = num_gnn_layers,
            dropout       = dropout,
        )

        # --- 2. Cross-Attention ---
        self.cross_attention = CrossAttention(
            embed_dim  = hidden_dim,
            num_heads  = num_attn_heads,
            dropout    = dropout,
        )

        # --- 3. Projection finale après pooling ---
        # Pooling : mean + add → hidden_dim * 2 par molécule
        # Après concaténation don + acc : hidden_dim * 4
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
        )

        # --- 4. Tête de prédiction ---
        self.prediction_head = nn.Linear(embedding_dim // 2, num_outputs)

        # --- Couche d'embedding CI (pour extraction du score de compatibilité) ---
        # C'est la couche juste avant la prédiction finale
        # On peut l'extraire pour obtenir le CI comme dans l'ANN

    def forward(self, graph_don: Data, graph_acc: Data) -> torch.Tensor:
        """
        Calcule les prédictions pour un batch de paires donneur-accepteur.

        Entrée :
            graph_don : batch de graphes donneurs  (torch_geometric Batch)
            graph_acc : batch de graphes accepteurs (torch_geometric Batch)

        Sortie :
            predictions : [batch_size, 6]  (PCE, Voc, Jsc, FF, ΔHOMO, ΔLUMO)
        """

        # --- Étape 1 : GNN → embeddings par atome ---
        # Même encodeur pour don et acc (poids partagés)
        h_don = self.gnn(graph_don)   # [N_don_total, hidden_dim]
        h_acc = self.gnn(graph_acc)   # [N_acc_total, hidden_dim]

        # --- Étape 2 : Cross-Attention ---
        h_don_enriched, h_acc_enriched = self.cross_attention(
            x_don     = h_don,
            x_acc     = h_acc,
            batch_don = graph_don.batch,
            batch_acc = graph_acc.batch,
        )

        # --- Étape 3 : Global Pooling (mean + add) ---
        don_mean = global_mean_pool(h_don_enriched, graph_don.batch)  # [B, hidden_dim]
        don_add  = global_add_pool(h_don_enriched, graph_don.batch)   # [B, hidden_dim]
        acc_mean = global_mean_pool(h_acc_enriched, graph_acc.batch)  # [B, hidden_dim]
        acc_add  = global_add_pool(h_acc_enriched, graph_acc.batch)   # [B, hidden_dim]

        # Concaténation : [B, hidden_dim * 4]
        combined = torch.cat([don_mean, don_add, acc_mean, acc_add], dim=1)

        # --- Étape 4 : Projection + Prédiction ---
        # La sortie de projection est ton embedding CI pour ce modèle GNN
        embedding_CI = self.projection(combined)   # [B, embedding_dim // 2]
        predictions  = self.prediction_head(embedding_CI)  # [B, 6]

        return predictions

    def get_CI_embedding(self, graph_don: Data, graph_acc: Data) -> torch.Tensor:
        """
        Extrait l'embedding CI (couche avant la prédiction finale).
        Équivalent de extract_compatibility_index() pour l'ANN.
        """
        h_don = self.gnn(graph_don)
        h_acc = self.gnn(graph_acc)

        h_don_enriched, h_acc_enriched = self.cross_attention(
            h_don, h_acc, graph_don.batch, graph_acc.batch
        )

        don_mean = global_mean_pool(h_don_enriched, graph_don.batch)
        don_add  = global_add_pool(h_don_enriched, graph_don.batch)
        acc_mean = global_mean_pool(h_acc_enriched, graph_acc.batch)
        acc_add  = global_add_pool(h_acc_enriched, graph_acc.batch)

        combined     = torch.cat([don_mean, don_add, acc_mean, acc_add], dim=1)
        embedding_CI = self.projection(combined)

        return embedding_CI


# =============================================================================
# 3. BACKBONE GNN (embeddings par atome, sans pooling global)
# =============================================================================

class _GNNBackbone(nn.Module):
    """
    Backbone GNN qui retourne les embeddings PAR ATOME (avant pooling).
    Utilisé par GNNCrossAttentionModel pour alimenter le cross-attention.
    C'est essentiellement MolecularGNNEncoder sans la couche de projection finale.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim:    int,
        num_layers:    int   = 3,
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout    = dropout

        # Projection des features d'arêtes
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        # Couches GINEConv
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = node_features if layer == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data: Data) -> torch.Tensor:
        """
        Retourne les embeddings par atome : [N_atoms, hidden_dim]
        """
        x         = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr

        edge_attr = self.edge_encoder(edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x   # [N_atoms, hidden_dim] — PAS de pooling ici


# =============================================================================
# 4. VÉRIFICATION RAPIDE
# =============================================================================

def verify_model(dataset, index=0):
    """
    Teste le modèle complet sur un exemple du dataset.
    """

    print("\n=== Vérification GNN + Cross-Attention ===")

    model = GNNCrossAttentionModel(
        node_features  = NODE_FEATURES,
        edge_features  = EDGE_FEATURES,
        hidden_dim     = HIDDEN_DIM,
        embedding_dim  = EMBEDDING_DIM,
        num_gnn_layers = 3,
        num_attn_heads = 4,
        dropout        = 0.1,
        num_outputs    = 6,
    )
    model.eval()

    sample    = dataset[index]
    graph_don = sample['graph_donor']
    graph_acc = sample['graph_acceptor']

    # Ajout du batch index (batch de taille 1)
    graph_don.batch = torch.zeros(graph_don.x.shape[0], dtype=torch.long)
    graph_acc.batch = torch.zeros(graph_acc.x.shape[0], dtype=torch.long)

    with torch.no_grad():
        predictions  = model(graph_don, graph_acc)
        embedding_CI = model.get_CI_embedding(graph_don, graph_acc)

    print(f"\nExemple n°{index}")
    print(f"  Graphe donneur   : {graph_don.x.shape[0]} atomes")
    print(f"  Graphe accepteur : {graph_acc.x.shape[0]} atomes")
    print(f"\n  Prédictions : {predictions.shape} → 6 valeurs")
    print(f"  → PCE={predictions[0,0]:.3f}, Voc={predictions[0,1]:.3f}, Jsc={predictions[0,2]:.3f}")
    print(f"  → FF={predictions[0,3]:.3f},  ΔHOMO={predictions[0,4]:.3f}, ΔLUMO={predictions[0,5]:.3f}")
    print(f"\n  Embedding CI : {embedding_CI.shape} → {embedding_CI.shape[1]} dimensions")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Paramètres totaux : {total_params:,}")
    print("\n✅ Modèle GNN + Cross-Attention fonctionnel !")

    return model


# =============================================================================
# 5. POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":

    CSV_PATH = "Data.csv"

    print("Chargement du dataset...")
    dataset = load_dataset(CSV_PATH)

    model = verify_model(dataset, index=0)
    
    X = pd.DataFrame(
        dataset,
        columns=["graph_donor", "graph_acceptor", "y", "smiles_don", "smiles_acc"]
    )
    print(X.head())
    # print(dataset[0]['y'])
    # for samples in dataset:
        
