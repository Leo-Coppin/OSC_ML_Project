import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from PIL import Image
import io

from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from SMILES_to_Graph import load_dataset, smiles_to_graph
from gnn_encoder import NODE_FEATURES, EDGE_FEATURES
from gnn_concat_fusion import GNNConcatFusion, DonorAcceptorDataset, collate_pairs
from GNN_CrossAttention import GNNCrossAttentionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

outputs = ["PCE", "Voc", "Jsc", "FF", "delta_HOMO", "delta_LUMO"]

# Noms des features atomiques (dans le même ordre que get_atom_features)
Atom_Feature_Names = [
    "atomic_num",
    "degree",
    "num_implicit_hs",
    "is_aromatic",
    "is_in_ring",
    "formal_charge",
    "hyb_SP",
    "hyb_SP2",
    "hyb_SP3",
    "hyb_SP3D",
    "hyb_SP3D2",
]

Bond_Feature_Names = [
    "bond_single",
    "bond_double",
    "bond_triple",
    "bond_aromatic",
    "is_conjugated",
    "bond_in_ring",
]


# =============================================================================
# PARTIE A — GNNExplainer : importance des features et des atomes
# =============================================================================

class GNNExplainerAnalysis:
    """
    Applique GNNExplainer sur les deux GNN (Concat Fusion et Cross-Attention).

    Pour chaque paire (donneur, accepteur) :
      - GNNExplainer calcule un masque node_mask [N_atomes, N_features]
      - On moyenne sur les atomes → importance par feature [N_features]
      - On stocke aussi les valeurs brutes des features pour la couleur du beeswarm
      - On accumule sur toutes les paires pour avoir une vue globale

    Produit :
      - Un beeswarm plot style SHAP par modèle (donneur + accepteur séparés)
      - Un summary bar plot par modèle (donneur + accepteur côte à côte)
      - Un bar plot comparatif des deux modèles
    """

    def __init__(self, model, model_name: str, target_idx: int = 0):
        self.model       = model.to(DEVICE).eval()
        self.model_name  = model_name
        self.target_idx  = target_idx
        self.target_name = outputs[target_idx]

    # -------------------------------------------------------------------------
    # Wrapper : isole UN graphe pour GNNExplainer
    # -------------------------------------------------------------------------

    def _make_single_graph_wrapper(self, fixed_graph: Data, role: str):
        """
        Retourne un wrapper nn.Module qui accepte UN seul graphe
        (le donneur OU l'accepteur) et fixe l'autre.

        Args:
            fixed_graph : le graphe fixé (l'autre molécule de la paire)
            role        : "donor"    → on explique le donneur (fixed = accepteur)
                          "acceptor" → on explique l'accepteur (fixed = donneur)
        """
        model      = self.model
        target_idx = self.target_idx
        fixed      = fixed_graph.to(DEVICE)

        class _Wrapper(torch.nn.Module):
            def forward(self, x, edge_index, edge_attr=None, batch=None):
                n = x.shape[0]
                b = batch if batch is not None else torch.zeros(n, dtype=torch.long, device=x.device)

                explained_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=b)

                batch_size  = b.max().item() + 1
                fixed_batch = Batch.from_data_list([fixed] * batch_size)

                if role == "donor":
                    out = model(explained_graph, fixed_batch)
                else:
                    out = model(fixed_batch, explained_graph)

                return out[:, target_idx].unsqueeze(-1)

        return _Wrapper()

    # -------------------------------------------------------------------------
    # Explication d'une paire (donneur, accepteur)
    # -------------------------------------------------------------------------

    def explain_pair(self, graph_don: Data, graph_acc: Data):
        """
        Lance GNNExplainer sur le donneur ET l'accepteur d'une même paire.

        Retourne un dict avec :
          - node_mask : [N_atomes, N_features]  masque brut GNNExplainer
          - feat_imp  : [N_features]             importance par feature (moy. sur atomes)
          - node_imp  : [N_atomes]               importance par atome (moy. sur features)

        Note : edge_mask_type=None car les arêtes ne sont pas différentiables
               dans le message passing de ces modèles (évite ValueError).
        """
        results = {}

        for role, explained, fixed in [
            ("donor",    graph_don, graph_acc),
            ("acceptor", graph_acc, graph_don),
        ]:
            graph = explained.to(DEVICE)
            if not hasattr(graph, 'batch') or graph.batch is None:
                graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=DEVICE)

            wrapper = self._make_single_graph_wrapper(fixed, role)

            explainer = Explainer(
                model            = wrapper,
                algorithm        = GNNExplainer(epochs=300, lr=0.01),
                explanation_type = "model",
                node_mask_type   = "attributes",  # masque par feature atomique
                edge_mask_type   = None,           # désactivé : gradients d'arêtes non disponibles
                model_config     = dict(
                    mode        = "regression",
                    task_level  = "graph",
                    return_type = "raw",
                ),
            )

            explanation = explainer(
                x          = graph.x,
                edge_index = graph.edge_index,
                edge_attr  = graph.edge_attr,
                batch      = graph.batch,
            )

            node_mask = explanation.node_mask.cpu().numpy()   # [N_atomes, N_features]
            feat_imp  = node_mask.mean(axis=0)                # [N_features]
            node_imp  = node_mask.mean(axis=-1)               # [N_atomes]

            results[role] = {
                "node_mask" : node_mask,
                "feat_imp"  : feat_imp,
                "node_imp"  : node_imp,
            }

            print(f"  ✅ GNNExplainer [{role}] — {graph.x.shape[0]} atomes, "
                  f"max feat imp = {feat_imp.max():.4f}")

        return results

    # -------------------------------------------------------------------------
    # Run global : accumulation sur n_explain paires
    # -------------------------------------------------------------------------

    def run(self, dataset: list, n_explain: int = 30, seed: int = 42):
        """
        Lance GNNExplainer sur n_explain paires et agrège les importances.

        Retourne :
            don_feat_imp    : [N_features]         importance moyenne — donneur
            acc_feat_imp    : [N_features]         importance moyenne — accepteur
            don_feat_matrix : [n_explain, N_feat]  importances par paire — donneur
            acc_feat_matrix : [n_explain, N_feat]  importances par paire — accepteur
            don_feat_values : [n_explain, N_feat]  valeurs brutes des features — donneur
            acc_feat_values : [n_explain, N_feat]  valeurs brutes des features — accepteur
        """
        print(f"\n🔍 GNNExplainer — {self.model_name} — cible : {self.target_name}")

        np.random.seed(seed)
        indices = np.random.choice(len(dataset), size=min(n_explain, len(dataset)), replace=False)

        don_feat_matrix = []   # [n_explain, N_features] — importances GNNExplainer
        acc_feat_matrix = []
        don_feat_values = []   # [n_explain, N_features] — valeurs brutes (pour la couleur)
        acc_feat_values = []

        for k, idx in enumerate(indices):
            sample    = dataset[idx]
            graph_don = sample['graph_donor'].to(DEVICE)
            graph_acc = sample['graph_acceptor'].to(DEVICE)

            print(f"  Paire {k+1}/{len(indices)} (idx={idx})")
            pair_res = self.explain_pair(graph_don, graph_acc)

            don_feat_matrix.append(pair_res["donor"]["feat_imp"])
            acc_feat_matrix.append(pair_res["acceptor"]["feat_imp"])

            # Valeur moyenne de chaque feature atomique sur la molécule (couleur du beeswarm)
            don_feat_values.append(graph_don.x.mean(dim=0).cpu().numpy())
            acc_feat_values.append(graph_acc.x.mean(dim=0).cpu().numpy())

        don_feat_matrix = np.stack(don_feat_matrix)   # [n_explain, N_features]
        acc_feat_matrix = np.stack(acc_feat_matrix)
        don_feat_values = np.stack(don_feat_values)
        acc_feat_values = np.stack(acc_feat_values)

        don_feat_imp = don_feat_matrix.mean(axis=0)   # [N_features]
        acc_feat_imp = acc_feat_matrix.mean(axis=0)

        print(f"✅ Agrégation terminée sur {len(indices)} paires.")
        return (
            don_feat_imp, acc_feat_imp,
            don_feat_matrix, acc_feat_matrix,
            don_feat_values, acc_feat_values,
        )

    # -------------------------------------------------------------------------
    # Beeswarm plot style SHAP
    # -------------------------------------------------------------------------

    def plot_beeswarm(
        self,
        feat_matrix : np.ndarray,   # [n_explain, N_features] — importances par paire
        feat_values : np.ndarray,   # [n_explain, N_features] — valeurs brutes (couleur)
        role        : str,          # "Donneur" ou "Accepteur"
        top_k       : int  = 11,    # toutes les features par défaut (11 au total)
        save_path   : str  = None,
    ):
        """
        Beeswarm plot inspiré du SHAP summary plot :
          - axe X  = importance GNNExplainer (équivalent de la SHAP value)
          - axe Y  = features triées par importance moyenne décroissante
          - couleur = valeur de la feature (bleu=basse → rouge=haute)
          - jitter vertical pour éviter la superposition des points

        Note : contrairement à SHAP, les importances GNNExplainer sont toujours >= 0
               (masque d'attention), il n'y a donc pas de points à gauche de 0.
        """
        n_samples, n_feat = feat_matrix.shape
        k = min(top_k, n_feat)

        # Tri par importance moyenne décroissante
        mean_imp   = feat_matrix.mean(axis=0)
        order      = np.argsort(mean_imp)[::-1][:k]
        order_plot = order[::-1]   # du moins important (bas) au plus important (haut)

        feat_names_ordered = [Atom_Feature_Names[i] for i in order_plot]
        imp_ordered        = feat_matrix[:, order_plot]   # [n_samples, k]
        val_ordered        = feat_values[:, order_plot]   # [n_samples, k]

        # Normalisation [0, 1] des valeurs de features pour la colormap
        val_norm = val_ordered.copy()
        for j in range(val_norm.shape[1]):
            col  = val_norm[:, j]
            vmin, vmax = col.min(), col.max()
            if vmax > vmin:
                val_norm[:, j] = (col - vmin) / (vmax - vmin)
            else:
                val_norm[:, j] = 0.5

        cmap = plt.cm.coolwarm
        fig, ax = plt.subplots(figsize=(10, max(6, k * 0.52)))

        for feat_pos in range(k):
            x_vals = imp_ordered[:, feat_pos]
            colors = cmap(val_norm[:, feat_pos])

            # Jitter vertical (beeswarm simplifié)
            np.random.seed(feat_pos)
            y_jitter = feat_pos + np.random.uniform(-0.35, 0.35, size=n_samples)

            ax.scatter(
                x_vals, y_jitter,
                c=colors, s=20, alpha=0.85,
                linewidths=0, zorder=3,
            )

        # Ligne verticale à 0
        ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--', zorder=1)

        # Grille horizontale légère
        for j in range(k):
            ax.axhline(j, color='#dddddd', linewidth=0.5, linestyle=':', zorder=0)

        ax.set_yticks(range(k))
        ax.set_yticklabels(feat_names_ordered, fontsize=10)
        ax.set_xlabel("Importance GNNExplainer (impact sur la sortie du modèle)", fontsize=11)
        ax.set_title(
            f"GNNExplainer Summary — {self.model_name} — {self.target_name} — {role}",
            fontsize=13, fontweight='bold',
        )
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Colorbar verticale (Feature value : Low → High)
        sm = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
        cbar.set_label("Feature value", fontsize=10, rotation=270, labelpad=15)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Sauvegardé : {save_path}")
        plt.show()

    # -------------------------------------------------------------------------
    # Summary bar plot : donneur + accepteur côte à côte
    # -------------------------------------------------------------------------

    def plot_summary_bar(
        self,
        don_feat_imp : np.ndarray,
        acc_feat_imp : np.ndarray,
        top_k        : int  = 11,
        save_path    : str  = None,
    ):
        """
        Barplot horizontal des features les plus importantes (GNNExplainer).
        Panneau gauche = donneur (rouge), panneau droite = accepteur (bleu).
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"GNNExplainer — Feature Importance — {self.model_name} — {self.target_name}",
            fontsize=13, fontweight='bold',
        )

        for ax, imp, label, color in [
            (axes[0], don_feat_imp, "Donneur",   "#e74c3c"),
            (axes[1], acc_feat_imp, "Accepteur", "#3498db"),
        ]:
            k          = min(top_k, len(imp))
            sorted_idx = np.argsort(imp)[::-1][:k]

            ax.barh(
                [Atom_Feature_Names[i] for i in sorted_idx][::-1],
                imp[sorted_idx][::-1],
                color=color, alpha=0.85,
            )
            ax.set_xlabel("Importance moyenne (GNNExplainer)", fontsize=11)
            ax.set_title(label, fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Sauvegardé : {save_path}")
        plt.show()


# =============================================================================
# PARTIE B — Comparaison des deux modèles
# =============================================================================

def compare_models_gnnexplainer(
    results_concat : dict,
    results_cross  : dict,
    target_name    : str,
    save_dir       : str = ".",
):
    """
    Compare les importances GNNExplainer des deux modèles sur un seul graphique.

    results_* : dict avec clés "donor" et "acceptor", chacune contenant
                un tableau [N_features] d'importances moyennes.

    Produit un barplot à 2 panneaux (Donneur / Accepteur),
    avec les barres Concat et Cross-Attention côte à côte.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"GNNExplainer — Comparaison des modèles — {target_name}",
        fontsize=14, fontweight='bold',
    )

    x     = np.arange(len(Atom_Feature_Names))
    width = 0.35

    for ax, role, label in [
        (axes[0], "donor",    "Donneur"),
        (axes[1], "acceptor", "Accepteur"),
    ]:
        imp_concat = results_concat[role]
        imp_cross  = results_cross[role]

        # Tri par importance totale (union des deux modèles)
        order = np.argsort(imp_concat + imp_cross)[::-1]

        ax.bar(x - width/2, imp_concat[order], width, label="Concat Fusion",   color="#e74c3c", alpha=0.85)
        ax.bar(x + width/2, imp_cross[order],  width, label="Cross-Attention", color="#3498db", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([Atom_Feature_Names[i] for i in order], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Importance moyenne (GNNExplainer)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, f"gnnexplainer_comparison_{target_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾 Sauvegardé : {path}")
    plt.show()


# =============================================================================
# PARTIE C — Visualisation moléculaire (heatmap par atome)
# =============================================================================

def visualize_molecule_importance(
    smiles          : str,
    node_importance : np.ndarray,
    graph           : Data,
    title           : str = "",
    save_path       : str = None,
):
    """
    Dessine la molécule avec chaque atome coloré selon son importance.
    Rouge = très important, vert = peu important.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ⚠️  SMILES invalide : {smiles}")
        return

    imp = node_importance.copy()
    if imp.max() > imp.min():
        imp = (imp - imp.min()) / (imp.max() - imp.min())
    else:
        imp = np.zeros_like(imp)

    cmap     = cm.get_cmap("RdYlGn_r")
    atom_col = {i: cmap(score)[:3] for i, score in enumerate(imp)}

    bond_col = {}
    for bond_idx in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(bond_idx)
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_col[bond_idx] = cmap((imp[i] + imp[j]) / 2)[:3]

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 400)
    drawer.drawOptions().addAtomIndices = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms      = list(atom_col.keys()),
        highlightAtomColors = atom_col,
        highlightBonds      = list(bond_col.keys()),
        highlightBondColors = bond_col,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight='bold')

    try:
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg.encode())
        img = Image.open(io.BytesIO(png_data))
        axes[0].imshow(img)
    except ImportError:
        axes[0].text(0.5, 0.5,
                     "Installer cairosvg pour l'image\n(pip install cairosvg)",
                     ha='center', va='center', transform=axes[0].transAxes)
    axes[0].axis("off")

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm   = cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], orientation='vertical', fraction=0.8)
    cbar.set_label("Importance atomique (normalisée)", fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Sauvegardé : {save_path}")
    plt.show()


# =============================================================================
# PARTIE D — PIPELINE COMPLÈTE
# =============================================================================

def run_full_analysis(
    dataset_path_test : str,
    checkpoint_concat : str,
    checkpoint_cross  : str,
    target_idx        : int  = 0,
    n_explain         : int  = 30,
    save_dir          : str  = "gnnexplainer_results",
    cross_params      : dict = None,
):
    """
    Pipeline complète GNNExplainer pour les deux modèles GNN.

    Pour chaque modèle :
      1. Lance GNNExplainer sur n_explain paires
      2. Agrège les importances de features
      3. Produit un beeswarm plot style SHAP (donneur + accepteur séparés)
      4. Produit un summary bar plot (donneur + accepteur côte à côte)

    Puis compare les deux modèles sur un graphique de comparaison.

    Args:
        dataset_path_test : chemin vers test_dataset.csv
        checkpoint_concat : chemin vers best_gnn_concat_model.pt
        checkpoint_cross  : chemin vers best_GNN_CrossAttention.pt
        target_idx        : 0=PCE, 1=Voc, 2=Jsc, 3=FF, 4=delta_HOMO, 5=delta_LUMO
        n_explain         : nombre de paires à expliquer (30 = bon équilibre vitesse/stabilité)
        save_dir          : dossier de sauvegarde des figures
        cross_params      : hyperparamètres du modèle cross-attention (optionnel)
    """
    os.makedirs(save_dir, exist_ok=True)
    target_name = outputs[target_idx]

    print("=" * 60)
    print(f"  GNNExplainer Analysis — cible : {target_name}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Chargement du dataset
    # ------------------------------------------------------------------
    print("\nOuverture du dataset de test...")
    dataset = load_dataset(dataset_path_test)
    print(f"   {len(dataset)} paires chargées.")

    # ------------------------------------------------------------------
    # 2. Chargement des modèles
    # ------------------------------------------------------------------
    print("\nChargement des modèles...")

    # --- Concat Fusion ---
    model_concat = GNNConcatFusion(share_encoder=False, dropout=0.2)
    state = torch.load(checkpoint_concat, map_location=DEVICE)
    if "model_state_dict" in state:
        model_concat.load_state_dict(state["model_state_dict"])
    else:
        model_concat.load_state_dict(state)
    model_concat.to(DEVICE).eval()
    print("  Concat model chargé.")

    # --- Cross-Attention ---
    ckpt_cross = torch.load(checkpoint_cross, map_location=DEVICE)
    if cross_params is None:
        cross_params = ckpt_cross.get("params", {})

    heads      = cross_params.get("num_attn_heads", 4)
    head_dim   = cross_params.get("hidden_dim", 64)
    hidden_dim = heads * head_dim

    model_cross = GNNCrossAttentionModel(
        hidden_dim     = hidden_dim,
        embedding_dim  = cross_params.get("embedding_dim", 128),
        num_gnn_layers = cross_params.get("num_gnn_layers", 3),
        num_attn_heads = heads,
        dropout        = cross_params.get("dropout", 0.1),
        num_outputs    = 6,
    )
    model_cross.load_state_dict(ckpt_cross["model_state_dict"])
    model_cross.to(DEVICE).eval()
    print("  Cross-Attention model chargé.")

    # ------------------------------------------------------------------
    # 3. GNNExplainer — importance des features atomiques
    # ------------------------------------------------------------------
    print(f"\nGNNExplainer — importance des features atomiques ({n_explain} paires)...")

    all_results = {}

    for model, name in [
        (model_concat, "concat"),
        (model_cross,  "cross_attention"),
    ]:
        analyser = GNNExplainerAnalysis(model, name, target_idx)

        # Calcul des importances agrégées + matrices par paire
        (
            don_feat_imp, acc_feat_imp,
            don_feat_matrix, acc_feat_matrix,
            don_feat_values, acc_feat_values,
        ) = analyser.run(dataset, n_explain=n_explain)

        all_results[name] = {
            "donor"    : don_feat_imp,
            "acceptor" : acc_feat_imp,
        }

        # -- Beeswarm plots style SHAP (un par rôle) --------------------
        analyser.plot_beeswarm(
            feat_matrix = don_feat_matrix,
            feat_values = don_feat_values,
            role        = "Donneur",
            save_path   = os.path.join(
                save_dir,
                f"gnnexplainer_beeswarm_donor_{name}_{target_name}.png",
            ),
        )
        analyser.plot_beeswarm(
            feat_matrix = acc_feat_matrix,
            feat_values = acc_feat_values,
            role        = "Accepteur",
            save_path   = os.path.join(
                save_dir,
                f"gnnexplainer_beeswarm_acceptor_{name}_{target_name}.png",
            ),
        )

        # -- Summary bar plot classique (donneur + accepteur côte à côte)
        analyser.plot_summary_bar(
            don_feat_imp = don_feat_imp,
            acc_feat_imp = acc_feat_imp,
            save_path    = os.path.join(
                save_dir,
                f"gnnexplainer_summary_bar_{name}_{target_name}.png",
            ),
        )

    # ------------------------------------------------------------------
    # 4. Comparaison des deux modèles
    # ------------------------------------------------------------------
    print("\nComparaison des deux modèles...")
    compare_models_gnnexplainer(
        results_concat = all_results["concat"],
        results_cross  = all_results["cross_attention"],
        target_name    = target_name,
        save_dir       = save_dir,
    )

    print(f"\n✅ Analyse GNNExplainer terminée ! Figures sauvegardées dans '{save_dir}/'")
    return all_results


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":

    # --- Pour une seule cible (ex : PCE) ---
    # run_full_analysis(
    #     dataset_path_test = "test_dataset.csv",
    #     checkpoint_concat = "best_gnn_concat_model.pt",
    #     checkpoint_cross  = "best_GNN_CrossAttention.pt",
    #     target_idx        = 0,    # 0 = PCE
    #     n_explain         = 30,
    #     save_dir          = "gnnexplainer_results",
    # )

    # --- Pour toutes les cibles ---
    for i in range(6):
        print(f"\n{'='*60}")
        print(f"  Cible : {outputs[i]}")
        print(f"{'='*60}")
        run_full_analysis(
            dataset_path_test = "test_dataset.csv",
            checkpoint_concat = "best_gnn_concat_model.pt",
            checkpoint_cross  = "best_GNN_CrossAttention.pt",
            target_idx        = i,
            n_explain         = 30,
            save_dir          = f"gnnexplainer_results/target_{outputs[i]}",
        )
        print(f"Analyse terminée pour la cible : {outputs[i]}")