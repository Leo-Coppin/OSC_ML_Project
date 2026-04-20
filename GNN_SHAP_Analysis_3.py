import os
import io
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')   # ← backend sans GUI, à mettre AVANT import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from PIL import Image

from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer

from rdkit import Chem
from rdkit.Chem import Draw
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
# PARTIE A — GNNExplainer : importance des features, atomes et liaisons
# =============================================================================

class GNNExplainerAnalysis:
    """
    Applique GNNExplainer sur les deux GNN (Concat Fusion et Cross-Attention).

    Deux niveaux d'explication :

    1. NIVEAU FEATURE (beeswarm / barplot)
       node_mask_type = 'attributes' → masque [N_atomes, N_features]
       Moyenné sur les atomes → importance globale de chaque type de feature
       (atomic_num, hybridation, aromaticité, etc.)

    2. NIVEAU ATOME / LIAISON (visualisation sous-structures)
       node_mask_type = 'object'     → masque [N_atomes, 1]
       edge_mask_type = 'object'     → masque [N_edges]
       → indique QUELS atomes et QUELLES liaisons sont importants
       → répond à la question : "sur quel atome spécifique ?"
    """

    def __init__(self, model, model_name: str, target_idx: int = 0):
        self.model       = model.to(DEVICE).eval()
        self.model_name  = model_name
        self.target_idx  = target_idx
        self.target_name = outputs[target_idx]

    # -------------------------------------------------------------------------
    # Wrapper : isole UN graphe pour GNNExplainer
    # Note : on ne passe PAS edge_attr au wrapper pour permettre à PyG
    #        de calculer les gradients sur edge_index (edge_mask_type='object')
    # -------------------------------------------------------------------------
    def _make_single_graph_wrapper(self, fixed_graph: Data, role: str, detach_edges: bool = True):
        model      = self.model
        target_idx = self.target_idx
        fixed      = fixed_graph.to(DEVICE)

        class _Wrapper(torch.nn.Module):
            def forward(self, x, edge_index, edge_attr=None, batch=None):
                n = x.shape[0]
                b = batch if batch is not None else torch.zeros(
                    n, dtype=torch.long, device=x.device
                )

                # ✅ CORRECTION : toujours passer edge_attr s'il est disponible.
                # detach_edges=True  → on détache pour éviter les problèmes de gradient
                # detach_edges=False → on passe edge_attr tel quel (nécessaire pour edge_encoder)
                if edge_attr is not None:
                    ea = edge_attr.detach() if detach_edges else edge_attr
                else:
                    ea = fixed.edge_attr  # fallback sur les edge_attr du graphe fixed

                explained_graph = Data(
                    x          = x,
                    edge_index = edge_index,
                    edge_attr  = ea,
                    batch      = b,
                )

                batch_size  = b.max().item() + 1
                fixed_batch = Batch.from_data_list([fixed] * batch_size)

                if role == "donor":
                    out = model(explained_graph, fixed_batch)
                else:
                    out = model(fixed_batch, explained_graph)

                return out[:, target_idx].unsqueeze(-1)

        return _Wrapper()
    # -------------------------------------------------------------------------
    # Niveau 1 — Explication par FEATURE (pour beeswarm / barplot)
    # node_mask_type = 'attributes' → [N_atomes, N_features]
    # -------------------------------------------------------------------------

    def explain_pair_features(self, graph_don: Data, graph_acc: Data):
        results = {}

        for role, explained, fixed in [
            ("donor",    graph_don, graph_acc),
            ("acceptor", graph_acc, graph_don),
        ]:
            graph = explained.to(DEVICE)
            if not hasattr(graph, 'batch') or graph.batch is None:
                graph.batch = torch.zeros(
                    graph.x.shape[0], dtype=torch.long, device=DEVICE
                )

            # detach_edges=True → edge_attr passé mais détaché du graphe de calcul
            wrapper = self._make_single_graph_wrapper(fixed, role, detach_edges=True)

            explainer = Explainer(
                model            = wrapper,
                algorithm        = GNNExplainer(epochs=300, lr=0.01),
                explanation_type = "model",
                node_mask_type   = "attributes",
                edge_mask_type   = None,
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

            node_mask = explanation.node_mask.cpu().numpy()
            feat_imp  = node_mask.mean(axis=0)
            node_imp  = node_mask.mean(axis=-1)

            results[role] = {
                "node_mask" : node_mask,
                "feat_imp"  : feat_imp,
                "node_imp"  : node_imp,
            }

            print(f"  ✅ [features] [{role}] — {graph.x.shape[0]} atoms, "
                f"max feat imp = {feat_imp.max():.4f}")

        return results
    # -------------------------------------------------------------------------
    # Niveau 2 — Explication par ATOME et LIAISON (pour visualisation)
    # node_mask_type = 'object' → [N_atomes, 1]
    # edge_mask_type = 'object' → [N_edges]
    # -------------------------------------------------------------------------

    def explain_pair_substructures(self, graph_don: Data, graph_acc: Data):
        results = {}

        for role, explained, fixed in [
            ("donor",    graph_don, graph_acc),
            ("acceptor", graph_acc, graph_don),
        ]:
            graph = explained.to(DEVICE)
            if not hasattr(graph, 'batch') or graph.batch is None:
                graph.batch = torch.zeros(
                    graph.x.shape[0], dtype=torch.long, device=DEVICE
                )

            wrapper = self._make_single_graph_wrapper(fixed, role, detach_edges=True)

            explainer = Explainer(
                model            = wrapper,
                algorithm        = GNNExplainer(epochs=300, lr=0.01),
                explanation_type = "model",
                node_mask_type   = "object",
                edge_mask_type   = None,   # ← pas de gradient sur les arêtes
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

            node_imp = explanation.node_mask.squeeze(-1).cpu().numpy()

            # Edge importance dérivée depuis les atomes (moyenne source + destination)
            ei = graph.edge_index.cpu().numpy()
            edge_imp = (node_imp[ei[0]] + node_imp[ei[1]]) / 2.0

            results[role] = {
                "node_imp"   : node_imp,
                "edge_imp"   : edge_imp,
                "edge_index" : ei,
            }

            print(f"  ✅ [substructures] [{role}] — {graph.x.shape[0]} atoms, "
                f"{edge_imp.shape[0]} bonds, "
                f"max node imp = {node_imp.max():.4f}")

        return results
     # -------------------------------------------------------------------------
    # Run global — Niveau 1 : accumulation des importances de features
    # -------------------------------------------------------------------------

    def run(self, dataset: list, n_explain: int = 30, seed: int = 42):
        """
        Lance explain_pair_features() sur n_explain paires et agrège.

        Retourne :
            don_feat_imp    : [N_features]         importance moyenne — donneur
            acc_feat_imp    : [N_features]         importance moyenne — accepteur
            don_feat_matrix : [n_explain, N_feat]  importances par paire — donneur
            acc_feat_matrix : [n_explain, N_feat]  importances par paire — accepteur
            don_feat_values : [n_explain, N_feat]  valeurs brutes des features — donneur
            acc_feat_values : [n_explain, N_feat]  valeurs brutes des features — accepteur
        """
        print(f"\n🔍 GNNExplainer [features] — {self.model_name} — cible : {self.target_name}")

        np.random.seed(seed)
        indices = np.random.choice(
            len(dataset), size=min(n_explain, len(dataset)), replace=False
        )

        don_feat_matrix = []
        acc_feat_matrix = []
        don_feat_values = []
        acc_feat_values = []

        for k, idx in enumerate(indices):
            sample    = dataset[idx]
            graph_don = sample['graph_donor'].to(DEVICE)
            graph_acc = sample['graph_acceptor'].to(DEVICE)

            print(f"  Paire {k+1}/{len(indices)} (idx={idx})")
            pair_res = self.explain_pair_features(graph_don, graph_acc)

            don_feat_matrix.append(pair_res["donor"]["feat_imp"])
            acc_feat_matrix.append(pair_res["acceptor"]["feat_imp"])

            # Valeur moyenne de chaque feature atomique (pour la couleur du beeswarm)
            don_feat_values.append(graph_don.x.mean(dim=0).cpu().numpy())
            acc_feat_values.append(graph_acc.x.mean(dim=0).cpu().numpy())

        don_feat_matrix = np.stack(don_feat_matrix)
        acc_feat_matrix = np.stack(acc_feat_matrix)
        don_feat_values = np.stack(don_feat_values)
        acc_feat_values = np.stack(acc_feat_values)

        don_feat_imp = don_feat_matrix.mean(axis=0)
        acc_feat_imp = acc_feat_matrix.mean(axis=0)

        print(f"✅ Agrégation terminée sur {len(indices)} paires.")
        return (
            don_feat_imp, acc_feat_imp,
            don_feat_matrix, acc_feat_matrix,
            don_feat_values, acc_feat_values,
        )

    # -------------------------------------------------------------------------
    # Run sous-structures — Niveau 2 : visualisation atome/liaison
    # -------------------------------------------------------------------------

    def run_substructures(
        self,
        dataset    : list,
        n_explain  : int = 5,
        seed       : int = 42,
        save_dir   : str = ".",
    ):
        """
        Lance explain_pair_substructures() sur n_explain paires et produit
        une visualisation moléculaire pour chaque paire.

        n_explain = 5 suffit car chaque paire génère 2 figures (don + acc).
        """
        print(f"\n🔬 GNNExplainer [substructures] — {self.model_name} — cible : {self.target_name}")

        np.random.seed(seed)
        indices = np.random.choice(
            len(dataset), size=min(n_explain, len(dataset)), replace=False
        )

        for k, idx in enumerate(indices):
            sample    = dataset[idx]
            graph_don = sample['graph_donor'].to(DEVICE)
            graph_acc = sample['graph_acceptor'].to(DEVICE)

            smiles_don = sample.get('smiles_don',   None)
            smiles_acc = sample.get('smiles_acc', None)

            print(f"  Pair {k+1}/{len(indices)} (idx={idx})")
            pair_res = self.explain_pair_substructures(graph_don, graph_acc)

            # Visualisation donneur
            if smiles_don:
                visualize_substructures(
                    smiles    = smiles_don,
                    node_imp  = pair_res["donor"]["node_imp"],
                    edge_imp  = pair_res["donor"]["edge_imp"],
                    edge_index= pair_res["donor"]["edge_index"],
                    title     = (f"Sub Structures — {self.model_name} — {self.target_name}\n"
                                 f"Donor (pair {idx})"),
                    save_path = os.path.join(
                        save_dir,
                        f"substruct_donor_{self.model_name}_{self.target_name}_pair{idx}.png"
                    ),
                )

            # Visualisation accepteur
            if smiles_acc:
                visualize_substructures(
                    smiles    = smiles_acc,
                    node_imp  = pair_res["acceptor"]["node_imp"],
                    edge_imp  = pair_res["acceptor"]["edge_imp"],
                    edge_index= pair_res["acceptor"]["edge_index"],
                    title     = (f"Sub Structures — {self.model_name} — {self.target_name}\n"
                                 f"Acceptor (pair {idx})"),
                    save_path = os.path.join(
                        save_dir,
                        f"substruct_acceptor_{self.model_name}_{self.target_name}_pair{idx}.png"
                    ),
                )

    # -------------------------------------------------------------------------
    # Beeswarm plot style SHAP
    # -------------------------------------------------------------------------

    def plot_beeswarm(
        self,
        feat_matrix : np.ndarray,
        feat_values : np.ndarray,
        role        : str,
        top_k       : int = 11,
        save_path   : str = None,
    ):
        """
        Beeswarm plot inspiré du SHAP summary plot :
          - axe X  = importance GNNExplainer (équivalent SHAP value)
          - axe Y  = features triées par importance moyenne décroissante
          - couleur = valeur de la feature (bleu=basse → rouge=haute)
          - jitter vertical pour éviter la superposition des points

        Note : les importances GNNExplainer sont toujours >= 0 (masque d'attention),
               il n'y a donc pas de points à gauche de 0 contrairement à SHAP.
        """
        n_samples, n_feat = feat_matrix.shape
        k = min(top_k, n_feat)

        mean_imp   = feat_matrix.mean(axis=0)
        order      = np.argsort(mean_imp)[::-1][:k]
        order_plot = order[::-1]   # du moins important (bas) au plus important (haut)

        feat_names_ordered = [Atom_Feature_Names[i] for i in order_plot]
        imp_ordered        = feat_matrix[:, order_plot]
        val_ordered        = feat_values[:, order_plot]

        # Normalisation [0, 1] pour la colormap
        val_norm = val_ordered.copy()
        for j in range(val_norm.shape[1]):
            col        = val_norm[:, j]
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

            np.random.seed(feat_pos)
            y_jitter = feat_pos + np.random.uniform(-0.35, 0.35, size=n_samples)

            ax.scatter(
                x_vals, y_jitter,
                c=colors, s=20, alpha=0.85,
                linewidths=0, zorder=3,
            )

        ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--', zorder=1)
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
        top_k        : int = 11,
        save_path    : str = None,
    ):
        """
        Barplot horizontal des features les plus importantes.
        Panneau gauche = donneur (rouge), panneau droite = accepteur (bleu).
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"GNNExplainer — Feature Importance — {self.model_name} — {self.target_name}",
            fontsize=13, fontweight='bold',
        )

        for ax, imp, label, color in [
            (axes[0], don_feat_imp, "Donor",   "#e74c3c"),
            (axes[1], acc_feat_imp, "Acceptor", "#3498db"),
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
# PARTIE B — Visualisation des sous-structures importantes
# =============================================================================

def visualize_substructures(
    smiles     : str,
    node_imp   : np.ndarray,
    edge_imp   : np.ndarray,
    edge_index : np.ndarray,
    title      : str   = "",
    save_path  : str   = None,
    threshold  : float = 0.3,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ⚠️  SMILES invalide : {smiles}")
        return

    n_atoms = mol.GetNumAtoms()

    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.full_like(arr, 0.5, dtype=float)

    node_imp_norm = normalize(node_imp[:n_atoms])

    cmap_atoms = plt.colormaps["RdYlGn_r"]
    atom_colors = {i: tuple(float(c) for c in cmap_atoms(node_imp_norm[i])[:3]) for i in range(n_atoms)}
    atom_radii  = {i: float(0.3 + 0.4 * node_imp_norm[i]) for i in range(n_atoms)}
    
    bond_imp_map = {}
    for e_idx in range(edge_imp.shape[0]):
        i = int(edge_index[0, e_idx])
        j = int(edge_index[1, e_idx])
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is not None:
            b_idx = bond.GetIdx()
            bond_imp_map[b_idx] = max(bond_imp_map.get(b_idx, 0.0), float(edge_imp[e_idx]))

    if bond_imp_map:
        vals = np.array(list(bond_imp_map.values()))
        mn, mx = vals.min(), vals.max()
        bond_imp_norm = {
            k: (v - mn) / (mx - mn) if mx > mn else 0.5
            for k, v in bond_imp_map.items()
        }
    else:
        bond_imp_norm = {}

    cmap_bonds = plt.colormaps["YlOrRd"]
    bond_colors = {b_idx: tuple(float(c) for c in cmap_bonds(v)[:3]) for b_idx, v in bond_imp_norm.items()}
    highlight_bonds = [b for b, v in bond_imp_norm.items() if v >= threshold]
    highlight_atoms = list(range(n_atoms))

    drawer = rdMolDraw2D.MolDraw2DSVG(700, 500)
    drawer.drawOptions().addAtomIndices      = False
    drawer.drawOptions().addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms      = highlight_atoms,
        highlightAtomColors = atom_colors,
        highlightAtomRadii  = atom_radii,
        highlightBonds      = highlight_bonds,
        highlightBondColors = {b: bond_colors[b] for b in highlight_bonds},
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[6, 0.4, 0.4], wspace=0.05)
    ax_mol   = fig.add_subplot(gs[0])
    ax_cbar1 = fig.add_subplot(gs[1])
    ax_cbar2 = fig.add_subplot(gs[2])

    # ✅ Rendu SVG robuste : cairosvg en priorité, svglib en fallback
    img_ok = False
    try:
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg.encode())
        img = Image.open(io.BytesIO(png_data))
        ax_mol.imshow(img)
        img_ok = True
    except Exception:
        pass

    if not img_ok:
        # Rendu SVG → image
        try:
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=svg.encode())
            img = Image.open(io.BytesIO(png_data))
            ax_mol.imshow(img)
        except Exception as e:
            print(f"  ⚠️  Rendu SVG échoué ({e})")
            ax_mol.text(
                0.5, 0.5,
                "Installer cairosvg pour le rendu moléculaire\npip install cairosvg",
                ha='center', va='center', transform=ax_mol.transAxes,
                fontsize=11, color='red',
            )
    if not img_ok:
        ax_mol.text(
            0.5, 0.5,
            "Installer cairosvg ou svglib pour le rendu moléculaire\n"
            "pip install cairosvg   ou   pip install svglib",
            ha='center', va='center', transform=ax_mol.transAxes, fontsize=11,
            color='red',
        )

    ax_mol.axis("off")
    ax_mol.set_title(title, fontsize=12, fontweight='bold', pad=10)

    norm1 = mcolors.Normalize(vmin=0, vmax=1)
    sm1   = ScalarMappable(cmap="RdYlGn_r", norm=norm1)
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, cax=ax_cbar1)
    cb1.set_label("Atoms Importance ", fontsize=9, labelpad=5)
    cb1.set_ticks([0, 0.5, 1])
    cb1.set_ticklabels(['Low', 'Medium', 'High'], fontsize=8)

    norm2 = mcolors.Normalize(vmin=0, vmax=1)
    sm2   = ScalarMappable(cmap="YlOrRd", norm=norm2)
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, cax=ax_cbar2)
    cb2.set_label("Bonds Importance ", fontsize=9, labelpad=5)
    cb2.set_ticks([0, 0.5, 1])
    cb2.set_ticklabels(['Low', 'Medium', 'High'], fontsize=8)

    plt.tight_layout()

    # ✅ CORRECTION CLEF : sauvegarder AVANT plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Sauvegardé : {save_path}")
    else:
        print("  ⚠️  Aucun save_path fourni — figure non sauvegardée")

    plt.show()
    plt.close(fig)   # ✅ libère la mémoire
# =============================================================================
# PARTIE C — Comparaison des deux modèles
# =============================================================================

def compare_models_gnnexplainer(
    results_concat : dict,
    results_cross  : dict,
    target_name    : str,
    save_dir       : str = ".",
):
    """
    Compare les importances GNNExplainer des deux modèles sur un seul graphique.
    Barplot à 2 panneaux (Donneur / Accepteur), barres côte à côte.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"GNNExplainer — Comparaison des modèles — {target_name}",
        fontsize=14, fontweight='bold',
    )

    x     = np.arange(len(Atom_Feature_Names))
    width = 0.35

    for ax, role, label in [
        (axes[0], "donor",    "Donor"),
        (axes[1], "acceptor", "Acceptor"),
    ]:
        imp_concat = results_concat[role]
        imp_cross  = results_cross[role]

        order = np.argsort(imp_concat + imp_cross)[::-1]

        ax.bar(x - width/2, imp_concat[order], width,
               label="Concat Fusion",   color="#e74c3c", alpha=0.85)
        ax.bar(x + width/2, imp_cross[order],  width,
               label="Cross-Attention", color="#3498db", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [Atom_Feature_Names[i] for i in order],
            rotation=45, ha='right', fontsize=9,
        )
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
# PARTIE D — PIPELINE COMPLÈTE
# =============================================================================

def run_full_analysis(
    dataset_path_test  : str,
    checkpoint_concat  : str,
    checkpoint_cross   : str,
    target_idx         : int  = 0,
    n_explain          : int  = 30,   # paires pour beeswarm / barplot
    n_explain_substruct: int  = 5,    # paires pour visualisation sous-structures
    save_dir           : str  = "gnnexplainer_results",
    cross_params       : dict = None,
):
    """
    Pipeline complète GNNExplainer pour les deux modèles GNN.

    Pour chaque modèle, deux niveaux d'analyse :

    Niveau 1 — Features (n_explain paires) :
      → beeswarm plot style SHAP  (donneur + accepteur séparés)
      → summary bar plot          (donneur + accepteur côte à côte)

    Niveau 2 — Sous-structures (n_explain_substruct paires) :
      → visualisation moléculaire avec atomes et liaisons colorés
         selon leur importance GNNExplainer

    Puis comparaison des deux modèles.

    Args:
        dataset_path_test   : chemin vers test_dataset.csv
        checkpoint_concat   : chemin vers best_gnn_concat_model.pt
        checkpoint_cross    : chemin vers best_GNN_CrossAttention.pt
        target_idx          : 0=PCE, 1=Voc, 2=Jsc, 3=FF, 4=delta_HOMO, 5=delta_LUMO
        n_explain           : paires pour l'analyse features (30 recommandé)
        n_explain_substruct : paires pour la visualisation sous-structures (5 recommandé)
        save_dir            : dossier de sauvegarde des figures
        cross_params        : hyperparamètres du modèle cross-attention (optionnel)
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
    # 3. Analyse features (beeswarm + barplot)
    # ------------------------------------------------------------------
    print(f"\n📊 Niveau 1 — Features ({n_explain} paires)...")
    all_results = {}

    for model, name in [
        (model_concat, "concat"),
        (model_cross,  "cross_attention"),
    ]:
        analyser = GNNExplainerAnalysis(model, name, target_idx)

        (
            don_feat_imp, acc_feat_imp,
            don_feat_matrix, acc_feat_matrix,
            don_feat_values, acc_feat_values,
        ) = analyser.run(dataset, n_explain=n_explain)

        all_results[name] = {
            "donor"    : don_feat_imp,
            "acceptor" : acc_feat_imp,
        }

        # Beeswarm plots style SHAP
        analyser.plot_beeswarm(
            feat_matrix = don_feat_matrix,
            feat_values = don_feat_values,
            role        = "Donor",
            save_path   = os.path.join(
                save_dir,
                f"gnnexplainer_beeswarm_donor_{name}_{target_name}.png",
            ),
        )
        analyser.plot_beeswarm(
            feat_matrix = acc_feat_matrix,
            feat_values = acc_feat_values,
            role        = "Acceptor",
            save_path   = os.path.join(
                save_dir,
                f"gnnexplainer_beeswarm_acceptor_{name}_{target_name}.png",
            ),
        )

        # Summary bar plot
        analyser.plot_summary_bar(
            don_feat_imp = don_feat_imp,
            acc_feat_imp = acc_feat_imp,
            save_path    = os.path.join(
                save_dir,
                f"gnnexplainer_summary_bar_{name}_{target_name}.png",
            ),
        )

    # ------------------------------------------------------------------
    # 4. Visualisation sous-structures (atomes + liaisons)
    # ------------------------------------------------------------------
    print(f"\n🔬 Niveau 2 — Sous-structures ({n_explain_substruct} paires)...")

    substruct_dir = os.path.join(save_dir, "substructures")
    os.makedirs(substruct_dir, exist_ok=True)

    for model, name in [
        (model_concat, "concat"),
        (model_cross,  "cross_attention"),
    ]:
        analyser = GNNExplainerAnalysis(model, name, target_idx)
        analyser.run_substructures(
            dataset   = dataset,
            n_explain = n_explain_substruct,
            save_dir  = substruct_dir,
        )

    # ------------------------------------------------------------------
    # 5. Comparaison des deux modèles
    # ------------------------------------------------------------------
    print("\n⚖️  Comparaison des deux modèles...")
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
    run_full_analysis(
        dataset_path_test   = "test_dataset.csv",
        checkpoint_concat   = "best_gnn_concat_model.pt",
        checkpoint_cross    = "best_GNN_CrossAttention.pt",
        target_idx          = 0,    # 0 = PCE
        n_explain           = 1,
        n_explain_substruct = 5,
        save_dir            = "gnnexplainer_results",
    )

