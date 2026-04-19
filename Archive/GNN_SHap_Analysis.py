import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import shap

from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
 
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
 
from SMILES_to_Graph import load_dataset, smiles_to_graph
from gnn_encoder import NODE_FEATURES, EDGE_FEATURES
from gnn_concat_fusion import GNNConcatFusion, DonorAcceptorDataset, collate_pairs
from GNN_CrossAttention import GNNCrossAttentionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

outputs = ["PCE", "Voc", "Jsc", "FF", "delta_HOMO", "delta_LUMO"]

# Atomic feature names (in the same ordred as get_atom_features)
Atom_Feature_Names = [
    "atomic_num",        # atomic number
    "degree",            # number of links
    "num_implicit_hs",   # implicites hydrogen
    "is_aromatic",       # Aromatic
    "is_in_ring",        # in a cycle
    "formal_charge",     # formal charge
    "hyb_SP",            # SP Hybridation 
    "hyb_SP2",           # SP2 Hybridation 
    "hyb_SP3",           # SP3Hybridation 
    "hyb_SP3D",          # SP3DHybridation
    "hyb_SP3D2",         # SP2D2 Hybridation
]

Bond_Feature_Names = [
    "bond_single",
    "bond_double",
    "bond_triple",
    "bond_aromatic",
    "is_conjugated",
    "bond_in_ring",
]

# identity most important atoms and bonds for each prediction
#   -> works for both models (concat and cross attention) 
class GNNExplainerAnalysis:

    def __init__(self, model, model_name: str, target_idx: int = 0):

        self.model      = model.to(DEVICE).eval()   #trained model
        self.model_name = model_name                #model name for saving results
        self.target_idx = target_idx                #target index for explanation (0=PCE, 1=Voc, …)
        self.target_name = outputs[target_idx]
        
    def _make_single_graph_wrapper(self, fixed_graph: Data, role: str):
        """
        Retourne un wrapper nn.Module qui accepte UN seul graphe
        (le donneur OU l'accepteur) et fixe l'autre.
 
        Cela permet à GNNExplainer d'explorer un graphe à la fois.
 
        Args:
            fixed_graph : le graphe fixé (l'autre molécule du binôme)
            role        : "donor" → on explique le donneur (fixed = accepteur)
                          "acceptor" → on explique l'accepteur (fixed = donneur)
        """
        model      = self.model
        target_idx = self.target_idx
        fixed      = fixed_graph.to(DEVICE)
 
        class _Wrapper(torch.nn.Module):
            def forward(self, x, edge_index, edge_attr=None, batch=None):
                # Reconstruit un Data à partir des arguments de GNNExplainer
                n = x.shape[0]
                b = batch if batch is not None else torch.zeros(n, dtype=torch.long, device=x.device)
 
                explained_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=b)
 
                # Duplique le graphe fixé pour matcher la taille du batch
                batch_size = b.max().item() + 1
                fixed_batch = Batch.from_data_list([fixed] * batch_size)
 
                if role == "donor":
                    out = model(explained_graph, fixed_batch)
                else:
                    out = model(fixed_batch, explained_graph)
 
                # GNNExplainer attend une sortie 2D [N, C] ou scalaire
                return out[:, target_idx].unsqueeze(-1)
 
        return _Wrapper()
    
    # -------------------------------------------------------------------------
    # Explication d'une paire (donneur, accepteur)
    # -------------------------------------------------------------------------
 
    def explain_pair(self, graph_don: Data, graph_acc: Data, smiles_don: str, smiles_acc: str):
        """
        Lance GNNExplainer sur le donneur ET l'accepteur d'une même paire.
 
        Retourne un dict avec les importances de nœuds et d'arêtes.
        """
        results = {}
 
        for role, explained, fixed, smiles in [
            ("donor",    graph_don, graph_acc, smiles_don),
            ("acceptor", graph_acc, graph_don, smiles_acc),
        ]:
            graph = explained.to(DEVICE)
            # S'assure que batch est défini
            if not hasattr(graph, 'batch') or graph.batch is None:
                graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=DEVICE)
 
            wrapper = self._make_single_graph_wrapper(fixed, role)
 
            explainer = Explainer(
                model          = wrapper,
                algorithm      = GNNExplainer(epochs=300, lr=0.01),
                explanation_type = "model",
                node_mask_type   = "attributes",   # masque par feature d'atome
                edge_mask_type   = "object",        # masque par arête
                model_config     = dict(
                    mode       = "regression",
                    task_level = "graph",
                    return_type = "raw",
                ),
            )
 
            explanation = explainer(
                x          = graph.x,
                edge_index = graph.edge_index,
                edge_attr  = graph.edge_attr,
                batch      = graph.batch,
            )
 
            # Importance par atome = moyenne sur les features du masque
            node_importance = explanation.node_mask.mean(dim=-1).cpu().numpy()  # [N_atoms]
 
            # Importance par arête
            edge_importance = explanation.edge_mask.cpu().numpy()               # [N_edges]
 
            results[role] = {
                "node_importance" : node_importance,
                "edge_importance" : edge_importance,
                "smiles"          : smiles,
                "graph"           : explained,
            }
 
            print(f"  ✅ GNNExplainer [{role}] — {graph.x.shape[0]} atomes, "
                  f"max node imp = {node_importance.max():.4f}")
 
        return results
 
 # -------------------------------------------------------------------------
    # Visualisation : coloration de la molécule par importance d'atomes
    # -------------------------------------------------------------------------
 
    @staticmethod
    def visualize_molecule_importance(
        smiles: str,
        node_importance: np.ndarray,
        edge_importance: np.ndarray,
        graph: Data,
        title: str = "",
        save_path: str = None,
    ):
        """
        Dessine la molécule avec chaque atome coloré selon son importance.
        Rouge = très important, blanc = neutre, bleu = peu important.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  ⚠️  SMILES invalide pour la visualisation : {smiles}")
            return
 
        # Normalisation [0, 1] pour la colormap
        imp = node_importance.copy()
        if imp.max() > imp.min():
            imp = (imp - imp.min()) / (imp.max() - imp.min())
        else:
            imp = np.zeros_like(imp)
 
        # Couleur par atome : colormap "RdYlGn_r" (rouge = important)
        cmap     = cm.get_cmap("RdYlGn_r")
        atom_col = {}
        for i, score in enumerate(imp):
            r, g, b, _ = cmap(score)
            atom_col[i] = (r, g, b)
 
        # Couleur par liaison : moyenne des deux atomes
        bond_col = {}
        edge_index = graph.edge_index.cpu().numpy()  # [2, N_edges]
        n_bonds = mol.GetNumBonds()
        for bond_idx in range(n_bonds):
            bond = mol.GetBondWithIdx(bond_idx)
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            score_bond = (imp[i] + imp[j]) / 2
            r, g, b, _ = cmap(score_bond)
            bond_col[bond_idx] = (r, g, b)
 
        # Dessin RDKit
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 400)
        drawer.drawOptions().addAtomIndices = False
 
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms   = list(atom_col.keys()),
            highlightAtomColors = atom_col,
            highlightBonds   = list(bond_col.keys()),
            highlightBondColors = bond_col,
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
 
        # Conversion SVG → image matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                 gridspec_kw={'width_ratios': [3, 1]})
        fig.suptitle(title, fontsize=13, fontweight='bold')
 
        # Affichage SVG via cairosvg si disponible, sinon texte simple
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
 
        # Colorbar verticale
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
 
    # -------------------------------------------------------------------------
    # Barplot : top-k atomes les plus importants
    # -------------------------------------------------------------------------
 
    @staticmethod
    def plot_top_atoms(smiles: str, node_importance: np.ndarray, top_k: int = 10,
                       title: str = "", save_path: str = None):
        """Barplot des top-k atomes les plus importants, avec leur symbole."""
 
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
 
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        labels  = [f"Atome {i} ({s})" for i, s in enumerate(symbols)]
 
        # Top-k
        k   = min(top_k, len(node_importance))
        idx = np.argsort(node_importance)[::-1][:k]
 
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(
            [labels[i] for i in idx][::-1],
            node_importance[idx][::-1],
            color=plt.cm.RdYlGn_r(node_importance[idx][::-1] /
                                   (node_importance.max() + 1e-8))
        )
        ax.set_xlabel("Importance (GNNExplainer)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
 
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
 
 
# =============================================================================
# PARTIE B — KernelSHAP : importance des FEATURES atomiques (global)
# =============================================================================
 
class KernelSHAPAnalysis:
    """
    Applique KernelSHAP sur une représentation vectorielle du graphe.
 
    Stratégie : on résume chaque graphe en un vecteur de features agrégées
    (moyenne sur les atomes → [NODE_FEATURES] pour chaque molécule).
    On concatène donneur + accepteur → vecteur de taille [2 * NODE_FEATURES].
 
    KernelSHAP calcule ensuite l'importance de chaque feature pour la prédiction.
 
    Avantage : fonctionne pour LES DEUX modèles avec le même code.
    Limite   : perd l'information spatiale (mais donne une vue globale).
    """
 
    def __init__(self, model, model_name: str, target_idx: int = 0):
        self.model       = model.to(DEVICE).eval()
        self.model_name  = model_name
        self.target_idx  = target_idx
        self.target_name = outputs[target_idx]
 
    # -------------------------------------------------------------------------
    # Conversion dataset → features agrégées (pour KernelSHAP)
    # -------------------------------------------------------------------------
 
    @staticmethod
    def dataset_to_feature_matrix(dataset: list) -> np.ndarray:
        """
        Transforme le dataset en matrice numpy [N, 2 * NODE_FEATURES].
 
        Chaque ligne = moyenne des features atomiques du donneur
                     + moyenne des features atomiques de l'accepteur.
 
        C'est le "background" que KernelSHAP va perturber.
        """
        rows = []
        for sample in dataset:
            feat_don = sample['graph_donor'].x.mean(dim=0).numpy()   # [11]
            feat_acc = sample['graph_acceptor'].x.mean(dim=0).numpy() # [11]
            rows.append(np.concatenate([feat_don, feat_acc]))          # [22]
        return np.array(rows, dtype=np.float32)
 
    # -------------------------------------------------------------------------
    # Wrapper : vecteur → prédiction scalaire (pour KernelSHAP)
    # -------------------------------------------------------------------------
 
    def _build_predict_fn(self, reference_dataset: list):
        """
        Construit la fonction de prédiction que KernelSHAP appellera.
 
        KernelSHAP passe un batch de vecteurs perturbés [N, 22].
        On reconstruit des graphes "perturbés" en remplaçant les features
        atomiques moyennes, puis on appelle le modèle.
 
        C'est une APPROXIMATION — les features sont appliquées uniformément
        sur tous les atomes de la molécule.
        """
        model      = self.model
        target_idx = self.target_idx
 
        # On garde un exemple de référence pour la structure des graphes
        ref_don = reference_dataset[0]['graph_donor']
        ref_acc = reference_dataset[0]['graph_acceptor']
 
        def predict_fn(X: np.ndarray) -> np.ndarray:
            """
            X : [N_samples, 2 * NODE_FEATURES]
            Retourne : [N_samples] — prédictions pour la cible choisie
            """
            preds = []
            n_feat = NODE_FEATURES  # 11
 
            for row in X:
                feat_don_vec = torch.tensor(row[:n_feat],  dtype=torch.float)  # [11]
                feat_acc_vec = torch.tensor(row[n_feat:],  dtype=torch.float)  # [11]
 
                # On remplace les features atomiques par le vecteur perturbé
                # (broadcast sur tous les atomes de la molécule de référence)
                n_don = ref_don.x.shape[0]
                n_acc = ref_acc.x.shape[0]
 
                x_don = feat_don_vec.unsqueeze(0).expand(n_don, -1)  # [N_don, 11]
                x_acc = feat_acc_vec.unsqueeze(0).expand(n_acc, -1)  # [N_acc, 11]
 
                g_don = Data(
                    x          = x_don,
                    edge_index = ref_don.edge_index,
                    edge_attr  = ref_don.edge_attr,
                    batch      = torch.zeros(n_don, dtype=torch.long),
                )
                g_acc = Data(
                    x          = x_acc,
                    edge_index = ref_acc.edge_index,
                    edge_attr  = ref_acc.edge_attr,
                    batch      = torch.zeros(n_acc, dtype=torch.long),
                )
 
                with torch.no_grad():
                    out = model(g_don.to(DEVICE), g_acc.to(DEVICE))
                preds.append(out[0, target_idx].item())
 
            return np.array(preds)
 
        return predict_fn
 
    # -------------------------------------------------------------------------
    # Lancement de KernelSHAP
    # -------------------------------------------------------------------------
 
    def run(self, dataset: list, n_background: int = 50, n_explain: int = 30):
        """
        Lance KernelSHAP sur n_explain échantillons du dataset.
 
        Args:
            dataset      : le dataset complet (liste de dicts)
            n_background : nombre d'exemples pour le background SHAP (50–100)
            n_explain    : nombre d'exemples à expliquer
 
        Retourne :
            shap_values  : np.array [n_explain, 2 * NODE_FEATURES]
            feature_names: liste des noms de features
        """
        print(f"\n🔍 KernelSHAP — {self.model_name} — cible : {self.target_name}")
 
        # Matrice de features
        X_all = self.dataset_to_feature_matrix(dataset)
 
        # Background : sous-ensemble aléatoire
        np.random.seed(42)
        bg_idx  = np.random.choice(len(dataset), size=min(n_background, len(dataset)), replace=False)
        X_bg    = X_all[bg_idx]
 
        # Exemples à expliquer
        ex_idx  = np.random.choice(len(dataset), size=min(n_explain, len(dataset)), replace=False)
        X_explain = X_all[ex_idx]
        ref_samples = [dataset[i] for i in ex_idx]
 
        # Fonction de prédiction
        predict_fn = self._build_predict_fn(ref_samples)
 
        # KernelSHAP
        explainer   = shap.KernelExplainer(predict_fn, X_bg)
        shap_values = explainer.shap_values(X_explain, nsamples=200, silent=True)
 
        # Noms des features : don_* + acc_*
        feature_names = (
            [f"don_{n}" for n in Atom_Feature_Names] +
            [f"acc_{n}" for n in Atom_Feature_Names]
        )
 
        print(f"SHAP calculated — shape : {shap_values.shape}")
        return shap_values, feature_names, X_explain
 
    # -------------------------------------------------------------------------
    # Visualisations SHAP
    # -------------------------------------------------------------------------
 
    def plot_summary(self, shap_values: np.ndarray, X_explain: np.ndarray,
                     feature_names: list, save_path: str = None):
        """
        Beeswarm plot SHAP : vue globale de l'importance des features.
        Chaque point = un exemple ; la couleur = la valeur de la feature.
        """
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_explain,
            feature_names = feature_names,
            show          = False,
            plot_type     = "dot",
            max_display   = 20,
        )
        plt.title(f"SHAP Summary — {self.model_name} — {self.target_name}",
                  fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Sauvegardé : {save_path}")
        plt.show()
 
    def plot_bar(self, shap_values: np.ndarray, feature_names: list,
                 top_k: int = 15, save_path: str = None):
        """
        Barplot des features les plus importantes (|SHAP| moyen).
        Séparé en features du donneur et de l'accepteur.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)  # [22]
        n        = NODE_FEATURES                      # 11
 
        # Séparation don / acc
        don_imp = mean_abs[:n]
        acc_imp = mean_abs[n:]
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Importance des features — {self.model_name} — {self.target_name}",
                     fontsize=13, fontweight='bold')
 
        for ax, imp, prefix, color in [
            (axes[0], don_imp, "Donneur",   "#e74c3c"),
            (axes[1], acc_imp, "Accepteur", "#3498db"),
        ]:
            sorted_idx = np.argsort(imp)[::-1][:top_k]
            ax.barh(
                [Atom_Feature_Names[i] for i in sorted_idx][::-1],
                imp[sorted_idx][::-1],
                color=color, alpha=0.8,
            )
            ax.set_xlabel("|SHAP| moyen", fontsize=11)
            ax.set_title(prefix, fontsize=12)
            ax.grid(axis='x', alpha=0.3)
 
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Sauvegardé : {save_path}")
        plt.show()
 
 
# =============================================================================
# PARTIE C — COMPARAISON DES DEUX MODÈLES
# =============================================================================
 
def compare_models_shap(
    model_concat,
    model_cross,
    dataset: list,
    target_idx: int = 0,
    n_background: int = 50,
    n_explain: int = 30,
    save_dir: str = ".",
    model_results : list = None,  # optionnel : résultats déjà calculés pour éviter de relancer SHAP
):
    """
    Lance KernelSHAP sur les deux modèles et compare les importances.
 
    Produit un barplot côte à côte : concat vs cross-attention.
    Très utile pour voir si les deux modèles "regardent" les mêmes features.
    """
    target_name = outputs[target_idx]
    print(f"\n📊 Comparaison SHAP — target : {target_name}")
 
    results = {}
    if model_results is None:
        for model, name in [(model_concat, "concat"), (model_cross, "cross_attention")]:
            analyser = KernelSHAPAnalysis(model, name, target_idx)
            sv, feat_names, X_ex = analyser.run(dataset, n_background, n_explain)
            results[name] = {"shap_values": sv, "feature_names": feat_names}
    else:
        for name, res in model_results:
            results[name] = res
 
    # Importance moyenne |SHAP| pour chaque modèle
    imp_concat = np.abs(results["concat"]["shap_values"]).mean(axis=0)
    imp_cross  = np.abs(results["cross_attention"]["shap_values"]).mean(axis=0)
    feat_names = results["concat"]["feature_names"]
 
    # Sélection des top-15 features (union des deux modèles)
    top_idx = np.argsort(imp_concat + imp_cross)[::-1][:15]
 
    fig, ax = plt.subplots(figsize=(12, 6))
    x     = np.arange(len(top_idx))
    width = 0.35
 
    ax.bar(x - width/2, imp_concat[top_idx], width, label="Concat Fusion",    color="#e74c3c", alpha=0.85)
    ax.bar(x + width/2, imp_cross[top_idx],  width, label="Cross-Attention",  color="#3498db", alpha=0.85)
 
    ax.set_xticks(x)
    ax.set_xticklabels([feat_names[i] for i in top_idx], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("|SHAP| moyen", fontsize=11)
    ax.set_title(f"feature importance comparasion  — target : {target_name}",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
 
    plt.tight_layout()
    path = f"{save_dir}/shap_comparison_{target_name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾 Sauvegardé : {path}")
    plt.show()
 
    return results
 
 
# =============================================================================
# PARTIE D — PIPELINE COMPLÈTE
# =============================================================================
 
def run_full_analysis(
    dataset_path_test: str,         #path to test_dataset.csv
    checkpoint_concat: str,         #path to best_gnn_concat_model.pt
    checkpoint_cross:  str,         #path to best_GNN_CrossAttention.pt
    target_idx:        int = 0,     # 0=PCE, 1=Voc, 2=Jsc, 3=FF, 4=delta_HOMO, 5=delta_LUMO
    n_background:      int = 50,    # number of samples for kernelSHAP
    n_explain:         int = 30,    # number of samples to explain
    save_dir:          str = "shap_GNN_results",
    concat_params:     dict = None, # paramètres du modèle concat si nécessaire
    cross_params:      dict = None, # paramètres du modèle cross-attention (chargés depuis checkpoint)
):
    #if it doesn't exist, create the directory to save the results
    import os
    os.makedirs(save_dir, exist_ok=True)
 
    print("=" * 60)
    print(f"  SHAP Analysis — cible : {outputs[target_idx]}")
    print("=" * 60)
 
    # charge Dataset
    print("\nOpening test dataset...")
    dataset = load_dataset(dataset_path_test)
    print(f"   {len(dataset)} pairs ready.")
 
    # Opening Models
    print("\nOpening models...")
 
    # --- Concat model ---
    model_concat = GNNConcatFusion(share_encoder=False, dropout=0.2)
    state = torch.load(checkpoint_concat, map_location=DEVICE)
    # Gère les checkpoints avec ou sans wrapper
    if "model_state_dict" in state:
        model_concat.load_state_dict(state["model_state_dict"])
    else:
        model_concat.load_state_dict(state)
    model_concat.to(DEVICE).eval()
    print("Concat model loaded.")
 
    # --- Cross-Attention model ---
    ckpt_cross = torch.load(checkpoint_cross, map_location=DEVICE)
    if cross_params is None:
        cross_params = ckpt_cross.get("params", {})
 
    # Reconstruction de hidden_dim à partir des hyperparamètres Optuna
    heads      = cross_params.get("num_attn_heads", 4)
    head_dim   = cross_params.get("hidden_dim", 64)
    hidden_dim = heads * head_dim   # comme dans ton objective Optuna
 
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
    print("  Cross-Attention model loaded.")
 
    # ------------------------------------------------------------------
    # 4. KernelSHAP — importance globale des features
    # ------------------------------------------------------------------
    print(f"\nKernelSHAP — Atomic features importance...")
 
    model_results = []
    for model, name in [(model_concat, "concat"), (model_cross, "cross_attention")]:
        analyser = KernelSHAPAnalysis(model, name, target_idx)
        sv, feat_names, X_ex = analyser.run(dataset, n_background, n_explain)
 
        analyser.plot_summary(sv, X_ex, feat_names,
                              save_path=f"{save_dir}/shap_summary_{name}_{outputs[target_idx]}.png")
 
        analyser.plot_bar(sv, feat_names,
                          save_path=f"{save_dir}/shap_bar_{name}_{outputs[target_idx]}.png")
        model_results.append((name, {"shap_values": sv, "feature_names": feat_names}))
        
    # Comparing the two models
    compare_models_shap(
        model_concat  = model_concat,
        model_cross   = model_cross,
        dataset       = dataset,
        target_idx    = target_idx,
        n_background  = n_background,
        n_explain     = n_explain,
        save_dir      = save_dir,
        model_results = model_results
    )
 
    print(f"\n✅ Shap Analysis finished ! Figures saved in '{save_dir}/'")
 
 
# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
 
if __name__ == "__main__":
 
    run_full_analysis(
        dataset_path_test = "test_dataset.csv",
        checkpoint_concat = "best_gnn_concat_model.pt",
        checkpoint_cross  = "best_GNN_CrossAttention.pt",
        target_idx        = 0,      # 0 = PCE (changer pour les autres cibles)
        n_background      = 70,     # augmenter pour plus de précision SHAP
        n_explain         = 50,     # idem
        save_dir          = "shap_GNN_results",
    )
 
    # --- Pour analyser TOUTES les cibles ---
    # for i in range(6):
    #     print(f"running analysis for target: {outputs[i]}")
    #     run_full_analysis(
    #         dataset_path_test = "test_dataset.csv",
    #     checkpoint_concat = "best_gnn_concat_model.pt",
    #         checkpoint_cross  = "best_GNN_CrossAttention.pt",
    #         target_idx        = i,      # 0 = PCE (changer pour les autres cibles)
    #         n_background      = 50,     # augmenter pour plus de précision SHAP
    #         n_explain         = 30,     # idem
    #         save_dir          = f"shap_GNN_results/target_{outputs[i]}"
    #     )
    #     print(f"Finished analysis for target: {outputs[i]}")