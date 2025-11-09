#With saving Hyperparameter:
# ==========================================
# han_hdf_han_final_cv.py  (Final, inductive option) — Nested CV + Tuning CSV
# - Inductive training/eval pipeline [INDUCTIVE]
# - Enforce φ4 (H->V->V->H) usage
# - Uses virus features via token fusion
# - REAL nested CV for hyperparameter tuning (leakage-safe)
# - Writes per-fold tuning summary + consensus CSV for your paper tables
# ==========================================
import os
import copy
import random
from typing import Tuple, Optional, Dict, List, Tuple as Tup

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv

# -----------------------------
# 0) Config / Paths
# -----------------------------
PATH  = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/GNN/Zika/"
PATH2 = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/Zika/"

FILE_GV       = os.path.join(PATH,  "InteractionData_Zika_Human_For_GNN.xlsx")      # HV edges
FILE_GENE     = os.path.join(PATH2, "ZikaInputdataForDeepL500HDF1000Non39Features_WithClass.csv")  # gene feats + Class
FILE_PPI_H    = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/GNN/BiogridHuman-With_EID.csv"  # HH
FILE_VIRUS    = os.path.join(PATH,  "GeneVirus_Zika.xlsx")                           # virus feats
FILE_PPI_VV   = os.path.join(PATH,  "Zika_Zika_Pr_Pr_interaction.xlsx")              # VV
FILE_GENE_ALL = os.path.join(PATH2, "Zika_normalized_whole_data_withoutclass_500HDF1000Non39Features.csv")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------- reporting / virus name --------
VIRUS_NAME = "ZIKV"   # برای هر ویروس مقدار مناسب را بگذار
TUNING_SUMMARY_PATH = f"tuning_summary_{VIRUS_NAME}.csv"

# -----------------------------
# Training & model
# -----------------------------
HIDDEN  = 256
HEADS   = 4
DROPOUT = 0.30
WEIGHT_DECAY = 1e-4
LR = 1e-3
MAX_EPOCHS = 300
PATIENCE = 30
N_SPLITS = 10
VAL_SIZE = 0.2

# تعداد فولد برای نِستد (inner) CV
K_INNER = 5

# DropEdge روی یال‌های φ در زمان آموزش
DROPEDGE_PHI_P = 0.20

# Path-guided budgets (per-hop)
K_HH_PHI2 = 10      # H->H (φ2)
K_HH_PHI3 = 5       # H->H در گام اول φ3
K_HV_GENE_TO_V = None   # H->V (φ1/φ3): None -> همه
K_VH_VIRUS_TO_H = 15    # V->H (φ1/φ3)
K_VV_PER_VIRUS  = 10    # V->V (φ4)

# VV control
INCLUDE_VV_METAPATH   = True   # enforce building φ4
MIN_VV_EDGES_FOR_USE  = 1      # require even minimal VV to allow φ4

# Post-HAN MHA (Feature Fusion)
USE_POST_MHA = True

# -----------------------------
# Helpers
# -----------------------------
def make_binary_labels(class_series: pd.Series) -> torch.Tensor:
    s = class_series.astype(str).str.strip()
    labels_map = {'HDF': 1, 'nonHDF': 0, 'NonHDF': 0, 'nonHdf': 0, 'NONHDF': 0}
    if set(s.unique()).issubset(labels_map.keys()):
        y = s.map(labels_map).astype(int).values
        return torch.tensor(y, dtype=torch.long)
    s_num = pd.to_numeric(s, errors='coerce')
    if s_num.isna().any():
        bad = class_series[s_num.isna()].head(10)
        raise ValueError(f"Unknown labels in 'Class'. Examples:\n{bad}")
    un = set(s_num.astype(int).unique())
    if un.issubset({0,1}):
        return torch.tensor(s_num.astype(int).values, dtype=torch.long)
    if un == {1,2}:
        return torch.tensor(s_num.replace({1:1,2:0}).astype(int).values, dtype=torch.long)
    raise ValueError(f"Unsupported label set: {sorted(list(un))}")

def topk_per_src(edge_index: torch.Tensor, num_src: int, k: Optional[int], seed: int = SEED) -> torch.Tensor:
    if edge_index.numel() == 0 or k is None or k <= 0:
        return edge_index
    rng = np.random.default_rng(seed)
    row, col = edge_index.detach().cpu().numpy()
    nbrs: Dict[int, List[int]] = {}
    for u, v in zip(row, col):
        nbrs.setdefault(u, []).append(v)
    new_row, new_col = [], []
    for u in range(num_src):
        vs = nbrs.get(u, [])
        if len(vs) > k:
            vs = rng.choice(vs, size=k, replace=False)
        for v in vs:
            new_row.append(u); new_col.append(v)
    if len(new_row) == 0:
        return edge_index.new_zeros((2,0))
    return torch.tensor([new_row, new_col], dtype=torch.long, device=edge_index.device)

def unique_edges(e: torch.Tensor) -> torch.Tensor:
    if e.numel() == 0:
        return e
    u = torch.unique(e.t().contiguous(), dim=0)
    return u.t().contiguous()

def compose_edges(edge_ab: torch.Tensor, edge_bc: torch.Tensor,
                  num_a: int, num_b: int) -> torch.Tensor:
    if edge_ab.numel() == 0 or edge_bc.numel() == 0:
        return edge_ab.new_zeros((2,0), dtype=torch.long)
    b2c: Dict[int, List[int]] = {}
    b, c = edge_bc[0].cpu().numpy(), edge_bc[1].cpu().numpy()
    for bb, cc in zip(b, c):
        b2c.setdefault(int(bb), []).append(int(cc))
    a_list, c_list = [], []
    a_arr, b_arr = edge_ab[0].cpu().numpy(), edge_ab[1].cpu().numpy()
    for aa, bb in zip(a_arr, b_arr):
        outs = b2c.get(int(bb), [])
        for cc in outs:
            a_list.append(int(aa)); c_list.append(int(cc))
    if len(a_list) == 0:
        return edge_ab.new_zeros((2,0), dtype=torch.long)
    e = torch.tensor([a_list, c_list], dtype=torch.long, device=edge_ab.device)
    mask = (e[0] != e[1])
    e = e[:, mask]
    e = unique_edges(e)
    return e

# ---- Virus aggregation: mean of neighbor viruses per gene ----
def aggregate_virus_mean(x_virus: torch.Tensor, hv_edge: torch.Tensor, num_genes: int) -> torch.Tensor:
    if hv_edge.numel() == 0:
        return x_virus.new_zeros((num_genes, x_virus.size(1)))
    gene_idx = hv_edge[0]
    virus_idx = hv_edge[1]
    d = x_virus.size(1)
    agg = x_virus.new_zeros((num_genes, d))
    agg.index_add_(0, gene_idx, x_virus[virus_idx])
    deg = torch.bincount(gene_idx, minlength=num_genes).clamp(min=1).unsqueeze(1).to(agg.dtype)
    return agg / deg
# -------------------------------------------------------------------

# -----------------------------
# Build base hetero + budgets + compose φ-edges
# -----------------------------
def build_base_heterodata(apply_undirected: bool = True) -> Tuple[HeteroData, pd.DataFrame, pd.DataFrame]:
    interactions_df = pd.read_excel(FILE_GV)
    features_df     = pd.read_csv(FILE_GENE)
    biogrid_df      = pd.read_csv(FILE_PPI_H)
    virus_df        = pd.read_excel(FILE_VIRUS)
    vv_df           = pd.read_excel(FILE_PPI_VV)

    for df, req in [(interactions_df, ['Genes','GenesVirus']),
                    (biogrid_df, ['Ensembl_ID_A','Ensembl_ID_B']),
                    (virus_df, ['GenesSymbolVirus']),
                   ]:
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise KeyError(f"Missing columns {miss} in dataframe with columns {list(df.columns)}")

    y_gene = make_binary_labels(features_df['Class'])

    interactions_df = interactions_df[
        (interactions_df['Genes'].isin(features_df['Genes'])) &
        (interactions_df['GenesVirus'].isin(virus_df['GenesSymbolVirus']))
    ]
    biogrid_df = biogrid_df[
        (biogrid_df['Ensembl_ID_A'].isin(features_df['Genes'])) &
        (biogrid_df['Ensembl_ID_B'].isin(features_df['Genes']))
    ]

    genes_to_index  = {g: i for i, g in enumerate(features_df['Genes'])}
    # ---- FIXED LINE ----
    virus_to_index = {v: i for i, v in enumerate(virus_df['GenesSymbolVirus'])}

    X_gene  = torch.tensor(features_df.iloc[:, 1:-1].values, dtype=torch.float)
    X_virus = torch.tensor(virus_df.iloc[:, 1:].values,       dtype=torch.float)

    gv_edges = torch.tensor([
        [genes_to_index[row['Genes']], virus_to_index[row['GenesVirus']]]
        for _, row in interactions_df.iterrows()
    ], dtype=torch.long).t().contiguous()

    hh_edges = torch.tensor([
        [genes_to_index[row['Ensembl_ID_A']], genes_to_index[row['Ensembl_ID_B']]]
        for _, row in biogrid_df.iterrows()
    ], dtype=torch.long).t().contiguous()

    a_col = 'Official_Symbol_A_Zika'; b_col = 'Official_Symbol_B_Zika'
    if a_col in vv_df.columns and b_col in vv_df.columns:
        vv_edges = torch.tensor([
            [virus_to_index.get(row[a_col], -1), virus_to_index.get(row[b_col], -1)]
            for _, row in vv_df.iterrows()
        ], dtype=torch.long).t().contiguous()
        if vv_edges.numel() > 0:
            mask_valid = (vv_edges >= 0).all(dim=0)
            vv_edges = vv_edges[:, mask_valid]
    else:
        vv_edges = torch.empty((2,0), dtype=torch.long)

    data = HeteroData()
    data['gene'].x  = X_gene
    data['gene'].y  = y_gene
    data['virus'].x = X_virus
    data['gene','interacts','virus'].edge_index = gv_edges
    data['gene','interacts','gene' ].edge_index = hh_edges
    if vv_edges.numel() > 0:
        data['virus','interacts','virus'].edge_index = vv_edges

    if apply_undirected:
        data = T.ToUndirected()(data)  # creates 'rev_interacts'
    return data, features_df, virus_df

def add_budget_relations(data: HeteroData) -> HeteroData:
    Ng = data['gene'].x.size(0)
    Nv = data['virus'].x.size(0)

    hv_key = ('gene','interacts','virus')
    vh_key = ('virus','rev_interacts','gene') if ('virus','rev_interacts','gene') in data.edge_index_dict else ('virus','interacts','gene')
    hh_key = ('gene','interacts','gene')
    vv_key = ('virus','interacts','virus')

    if hh_key in data.edge_index_dict and data[hh_key].edge_index.numel() > 0:
        hh_e = data[hh_key].edge_index
        data['gene','hh2','gene'].edge_index = topk_per_src(hh_e, Ng, K_HH_PHI2, seed=SEED)
        data['gene','hh3','gene'].edge_index = topk_per_src(hh_e, Ng, K_HH_PHI3, seed=SEED+1)

    if hv_key in data.edge_index_dict and data[hv_key].edge_index.numel() > 0:
        hv_e = data[hv_key].edge_index
        data['gene','hv','virus'].edge_index = topk_per_src(hv_e, Ng, K_HV_GENE_TO_V, seed=SEED+2)

    if vh_key in data.edge_index_dict and data[vh_key].edge_index.numel() > 0:
        vh_e = data[vh_key].edge_index
        data['virus','vh','gene'].edge_index = topk_per_src(vh_e, Nv, K_VH_VIRUS_TO_H, seed=SEED+3)

    if vv_key in data.edge_index_dict and data[vv_key].edge_index.numel() > 0:
        vv_e = data[vv_key].edge_index
        data['virus','vv','virus'].edge_index = topk_per_src(vv_e, Nv, K_VV_PER_VIRUS, seed=SEED+4)

    return data

def add_metapath_phi_edges(data: HeteroData, include_vv: bool = INCLUDE_VV_METAPATH,
                           min_vv_edges: int = MIN_VV_EDGES_FOR_USE) -> HeteroData:
    Ng = data['gene'].x.size(0); Nv = data['virus'].x.size(0)
    device = data['gene'].x.device

    hv = ('gene','hv','virus'); vh = ('virus','vh','gene')
    hh2 = ('gene','hh2','gene'); hh3 = ('gene','hh3','gene'); vv = ('virus','vv','virus')

    def get_e(key):
        return data[key].edge_index if key in data.edge_index_dict else torch.empty((2,0), dtype=torch.long, device=device)

    hv_e  = get_e(hv)
    vh_e  = get_e(vh)
    hh2_e = get_e(hh2)
    hh3_e = get_e(hh3)
    vv_e  = get_e(vv)

    if hh2_e.numel() > 0:
        data['gene','phi2','gene'].edge_index = unique_edges(hh2_e)

    if hv_e.numel() > 0 and vh_e.numel() > 0:
        phi1 = compose_edges(hv_e, vh_e, Ng, Nv)
        if phi1.numel() > 0:
            data['gene','phi1','gene'].edge_index = phi1

    if hh3_e.numel() > 0 and hv_e.numel() > 0 and vh_e.numel() > 0:
        tmp = compose_edges(hh3_e, hv_e, Ng, Ng)
        phi3 = compose_edges(tmp, vh_e, Ng, Nv)
        if phi3.numel() > 0:
            data['gene','phi3','gene'].edge_index = phi3

    # φ4 (MANDATORY) = H->V->V->H
    if include_vv:
        if vv_e.numel() < min_vv_edges:
            raise RuntimeError("φ4 required: insufficient V-V edges for composing H→V→V→H.")
        if hv_e.numel() == 0 or vh_e.numel() == 0:
            raise RuntimeError("φ4 required: missing H→V or V→H edges for composing H→V→V→H.")
        tmp = compose_edges(hv_e, vv_e, Ng, Nv)
        phi4 = compose_edges(tmp, vh_e, Ng, Nv)
        if phi4.numel() == 0:
            raise RuntimeError("φ4 required: composition produced zero edges for H→V→V→H.")
        data['gene','phi4','gene'].edge_index = phi4

    # fallback
    if not any((('gene',f'phi{i}','gene') in data.edge_index_dict and data[('gene',f'phi{i}','gene')].edge_index.numel() > 0) for i in [1,2,3,4]):
        if ('gene','interacts','gene') in data.edge_index_dict:
            data['gene','phi2','gene'].edge_index = unique_edges(data['gene','interacts','gene'].edge_index)
        else:
            raise RuntimeError("No meta-path edges could be constructed.")
    return data

def build_hdf_heterodata_with_phi() -> Tuple[HeteroData, pd.DataFrame, pd.DataFrame]:
    data, features_df, virus_df = build_base_heterodata(apply_undirected=True)
    data = add_budget_relations(data)
    data = add_metapath_phi_edges(data, include_vv=INCLUDE_VV_METAPATH, min_vv_edges=MIN_VV_EDGES_FOR_USE)
    return data, features_df, virus_df

# -----------------------------
# [INDUCTIVE] - build induced graphs by allowed genes
# -----------------------------
def induce_graph_by_genes(data_base: HeteroData, allowed_gene_idx: np.ndarray) -> HeteroData:
    """
    Create a graph where only edges incident to allowed genes (for gene-gene and gene-virus directions) are kept.
    VV edges are kept as-is. Budgets and φ will be rebuilt on the filtered graph.
    """
    data = copy.deepcopy(data_base)
    Ng = data['gene'].x.size(0)
    allowed_mask = torch.zeros(Ng, dtype=torch.bool)
    allowed_mask[torch.as_tensor(allowed_gene_idx, dtype=torch.long)] = True

    # remove any previous budget/phi if present
    for k in list(data.edge_index_dict.keys()):
        if k[0]=='gene' and k[2]=='gene' and (k[1].startswith('phi') or k[1] in ['hh2','hh3']):
            del data[k]
        if k == ('gene','hv','virus') or k == ('virus','vh','gene') or (k[0]=='virus' and k[2]=='virus' and k[1]=='vv'):
            del data[k]

    # filter base HH
    hh_key = ('gene','interacts','gene')
    if hh_key in data.edge_index_dict:
        e = data[hh_key].edge_index
        keep = allowed_mask[e[0]] & allowed_mask[e[1]]
        data[hh_key].edge_index = e[:, keep]

    # filter HV and VH (both directions)
    hv_key = ('gene','interacts','virus')
    if hv_key in data.edge_index_dict:
        e = data[hv_key].edge_index
        keep = allowed_mask[e[0]]
        data[hv_key].edge_index = e[:, keep]

    vh_key = ('virus','rev_interacts','gene') if ('virus','rev_interacts','gene') in data.edge_index_dict else ('virus','interacts','gene')
    if vh_key in data.edge_index_dict:
        e = data[vh_key].edge_index
        keep = allowed_mask[e[1]]
        data[vh_key].edge_index = e[:, keep]

    # rebuild budgets + φ on filtered graph
    data = add_budget_relations(data)
    data = add_metapath_phi_edges(data, include_vv=INCLUDE_VV_METAPATH, min_vv_edges=MIN_VV_EDGES_FOR_USE)
    return data
# -----------------------------

# -----------------------------
# Semantic Attention (meta-path level)
# -----------------------------
class SemanticAttention(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.proj(X).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        alpha = torch.softmax(scores, dim=1)
        fused = torch.sum(alpha.unsqueeze(-1) * X, dim=1)
        return fused, alpha

# -----------------------------
# Post-HAN Feature Fusion (optional)
# -----------------------------
class FeatureFusionMHA(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        N = tokens.size(0)
        cls = self.cls.expand(N, 1, -1)
        z, _ = self.mha(cls, tokens, tokens)
        z = self.post(z)
        return z.squeeze(1)

# -----------------------------
# Model
# -----------------------------
class SimpleHAN(nn.Module):
    """
    - Per φ: GAT x2 (gene->gene)
    - Semantic attention across φ
    - Fusion via MHA over tokens: [proj_gene, han_fused, virus_token]
    """
    def __init__(self, in_dim_gene: int, in_dim_virus: int,
                 phi_names: List[str], hidden: int = HIDDEN, heads: int = HEADS, dropout: float = DROPOUT,
                 use_post_mha: bool = USE_POST_MHA, use_virus_token: bool = True):
        super().__init__()
        self.phi_names = phi_names
        self.use_post_mha = use_post_mha
        self.use_virus_token = use_virus_token
        self.hidden = hidden
        self.heads = heads

        self.proj_gene  = nn.Sequential(nn.Linear(in_dim_gene,  hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_virus = nn.Sequential(nn.Linear(in_dim_virus, hidden), nn.ReLU(), nn.Dropout(dropout))

        self.gat1 = nn.ModuleDict({phi: GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout) for phi in phi_names})
        self.gat2 = nn.ModuleDict({phi: GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout) for phi in phi_names})

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        self.semantic_att = SemanticAttention(hidden, hidden=128, dropout=dropout)

        if self.use_post_mha:
            self.post_mha = FeatureFusionMHA(hidden, n_heads=heads, dropout=dropout)

        self.cls = nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        xg = self.proj_gene(x_dict['gene'])
        xv = self.proj_virus(x_dict['virus'])

        phi_keys = [('gene', f'phi{i}', 'gene') for i in [1,2,3,4]]
        phi_keys = [k for k in phi_keys if (k in edge_index_dict and edge_index_dict[k].numel() > 0)]
        if not phi_keys:
            raise RuntimeError("No φ-edges in edge_index_dict")
        # Enforce that φ4 really exists
        if ('gene','phi4','gene') not in edge_index_dict or edge_index_dict[('gene','phi4','gene')].numel() == 0:
            raise RuntimeError("φ4 required but missing in edge_index_dict during forward.")

        per_phi = []
        for k in phi_keys:
            phi = k[1]
            e = edge_index_dict[k]
            h1 = self.gat1[phi](xg, e)
            h1 = self.norm1(F.relu(h1) + xg)
            h1 = self.dropout(h1)
            h2 = self.gat2[phi](h1, e)
            h2 = self.norm2(F.relu(h2) + h1)
            per_phi.append(h2)

        H = torch.stack(per_phi, dim=1)
        fused, alpha = self.semantic_att(H)

        if self.use_post_mha:
            hv_key = ('gene','hv','virus') if ('gene','hv','virus') in edge_index_dict else ('gene','interacts','virus')
            if hv_key in edge_index_dict and edge_index_dict[hv_key].numel() > 0:
                zv = aggregate_virus_mean(xv, edge_index_dict[hv_key], xg.size(0))
            else:
                zv = xv.new_zeros(xg.size())
            tokens = torch.stack([xg, fused, zv], dim=1)
            fused = self.post_mha(tokens)

        logit = self.cls(fused).squeeze(-1)
        return {'gene': logit}

# -----------------------------
# Train/Eval Utils
# -----------------------------
def scale_features_in_fold(data: HeteroData, train_idx: np.ndarray):
    Xg = data['gene'].x.detach().cpu().numpy()
    sc_gene = StandardScaler().fit(Xg[train_idx])
    data['gene'].x = torch.tensor(sc_gene.transform(Xg), dtype=torch.float, device=data['gene'].x.device)

    Xv = data['virus'].x.detach().cpu().numpy()
    sc_v = StandardScaler().fit(Xv)
    data['virus'].x = torch.tensor(sc_v.transform(Xv), dtype=torch.float, device=data['virus'].x.device)
    return data, sc_gene, sc_v

def apply_scalers_to_data(sc_gene: StandardScaler, sc_virus: StandardScaler, data: HeteroData):
    data['gene'].x  = torch.tensor(sc_gene.transform(data['gene'].x.detach().cpu().numpy()),
                                   dtype=torch.float, device=data['gene'].x.device)
    data['virus'].x = torch.tensor(sc_virus.transform(data['virus'].x.detach().cpu().numpy()),
                                   dtype=torch.float, device=data['virus'].x.device)
    return data

def build_train_edges_with_dropedge_phi(data: HeteroData, p_phi=DROPEDGE_PHI_P) -> Dict[Tup[str,str,str], torch.Tensor]:
    cur = dict(data.edge_index_dict)
    Ng = data['gene'].x.size(0)
    for name in ['phi1','phi2','phi3','phi4']:
        k = ('gene', name, 'gene')
        if k in cur and cur[k].numel() > 0 and p_phi > 0:
            e = cur[k]
            e, _ = dropout_adj(e, p=p_phi, force_undirected=True, num_nodes=Ng)
            cur[k] = e
    return cur

# ---- Youden’s J = TPR + TNR − 1 ----
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: int = 201, metric: str = 'f1') -> float:
    thrs = np.linspace(0.0, 1.0, grid)
    best_t, best_s = 0.5, -1.0
    y_true = y_true.astype(int)

    for t in thrs:
        pred = (y_prob >= t).astype(int)
        if metric == 'f1':
            s = f1_score(y_true, pred, zero_division=0)
        elif metric == 'youden':
            tp = np.sum((y_true == 1) & (pred == 1))
            tn = np.sum((y_true == 0) & (pred == 0))
            fp = np.sum((y_true == 0) & (pred == 1))
            fn = np.sum((y_true == 1) & (pred == 0))
            tpr = tp / (tp + fn + 1e-12)
            tnr = tn / (tn + fp + 1e-12)
            s = tpr + tnr - 1.0
        else:
            s = f1_score(y_true, pred, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)

# [INDUCTIVE] train on train-graph, validate on train∪valid-graph
def train_one_fold_inductive(model: SimpleHAN,
                             data_train_graph: HeteroData,
                             data_val_graph: HeteroData,
                             train_idx, val_idx,
                             lr=LR, max_epochs=MAX_EPOCHS, weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    model = model.to(DEVICE)
    data_train_graph = data_train_graph.to(DEVICE)
    data_val_graph   = data_val_graph.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=DEVICE)
    val_idx_t   = torch.as_tensor(val_idx,   dtype=torch.long, device=DEVICE)
    targets = data_train_graph['gene'].y.float()  # same y across graphs

    best_state = None
    best_metric  = -1.0
    bad = 0

    for _ in range(max_epochs):
        model.train()
        cur_edges = build_train_edges_with_dropedge_phi(data_train_graph)
        logits = model(data_train_graph.x_dict, cur_edges)['gene']
        loss = criterion(logits[train_idx_t], targets[train_idx_t])

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # اختیاری
        optimizer.step()

        # Validation on train∪valid graph (no DropEdge)
        model.eval()
        with torch.no_grad():
            val_logits = model(data_val_graph.x_dict, data_val_graph.edge_index_dict)['gene'][val_idx_t]
            val_prob   = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_true   = targets[val_idx_t].detach().cpu().numpy()
            val_auc    = roc_auc_score(val_true, val_prob)

        if val_auc > best_metric:
            best_metric = val_auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_split(model: SimpleHAN, data: HeteroData, idx, thr: Optional[float] = None):
    model.eval()
    data = data.to(DEVICE)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)['gene'][idx_t]
        prob   = torch.sigmoid(logits).cpu().numpy()
        true   = data['gene'].y[idx_t].cpu().numpy()
    use_thr = 0.5 if thr is None else float(thr)
    pred = (prob >= use_thr).astype(int)
    brier = float(np.mean((prob - true)**2))
    res = dict(
        precision = precision_score(true, pred, zero_division=0),
        recall    = recall_score(true, pred, zero_division=0),
        f1        = f1_score(true, pred, zero_division=0),
        accuracy  = accuracy_score(true, pred),
        auc_roc   = roc_auc_score(true, prob),
        auc_pr    = average_precision_score(true, prob),
        brier     = brier,
        threshold = use_thr
    )
    return res, prob, pred

# -----------------------------
# (NEW) Search space & inner-CV tuning (Nested CV) — returns stats for table
# -----------------------------
def get_search_space():
    """
    Minimal, fast search space for nested CV (expand if you have budget).
    """
    space = []
    for h in [128, 256]:
        for heads in [2, 4]:
            for dr in [0.2, 0.3]:
                for lr in [5e-4, 1e-3]:
                    for wd in [1e-5, 1e-4]:
                        space.append(dict(HIDDEN=h, HEADS=heads, DROPOUT=dr, LR=lr, WEIGHT_DECAY=wd))
    return space

def inner_cv_tune(data_base: HeteroData,
                  y: np.ndarray,
                  outer_train_idx: np.ndarray,
                  in_gene: int,
                  in_virus: int):
    """
    Nested CV: choose best hyperparameters using ONLY outer-train.
    Returns:
      best_cfg, inner_stats (dict with mean/sd and per-fold lists for AUROC/AUPRC)
    """
    inner_skf = StratifiedKFold(n_splits=K_INNER, shuffle=True, random_state=SEED)
    search_space = get_search_space()

    best_cfg = None
    best_auc = -1.0
    best_aupr = -1.0
    best_lists = None  # (aucs, auprs)

    for cfg in search_space:
        aucs, auprs = [], []

        for inner_train_rel, inner_val_rel in inner_skf.split(np.zeros_like(y[outer_train_idx]), y[outer_train_idx]):
            inner_train_idx = outer_train_idx[inner_train_rel]
            inner_val_idx   = outer_train_idx[inner_val_rel]

            data_inner_train    = induce_graph_by_genes(data_base, inner_train_idx)
            data_inner_trainval = induce_graph_by_genes(data_base, np.concatenate([inner_train_idx, inner_val_idx]))

            data_inner_train, sc_gene, sc_v = scale_features_in_fold(copy.deepcopy(data_inner_train), inner_train_idx)
            data_inner_trainval = apply_scalers_to_data(sc_gene, sc_v, data_inner_trainval)

            phi_names_train = [k[1] for k in data_inner_train.edge_index_dict.keys()
                               if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_inner_train[k].edge_index.numel()>0)]
            if 'phi4' not in phi_names_train:
                continue

            model = SimpleHAN(in_dim_gene=in_gene, in_dim_virus=in_virus,
                              phi_names=phi_names_train,
                              hidden=cfg['HIDDEN'], heads=cfg['HEADS'], dropout=cfg['DROPOUT'],
                              use_post_mha=USE_POST_MHA, use_virus_token=True)

            model = train_one_fold_inductive(model,
                                             data_inner_train,
                                             data_inner_trainval,
                                             inner_train_idx,
                                             inner_val_idx,
                                             lr=cfg['LR'],
                                             max_epochs=MAX_EPOCHS,
                                             weight_decay=cfg['WEIGHT_DECAY'],
                                             patience=PATIENCE)

            model.eval()
            with torch.no_grad():
                val_logits = model(data_inner_trainval.x_dict, data_inner_trainval.edge_index_dict)['gene'][torch.as_tensor(inner_val_idx, device=DEVICE)]
                val_prob   = torch.sigmoid(val_logits).cpu().numpy()
                val_true   = y[inner_val_idx]
            auc  = roc_auc_score(val_true, val_prob)
            aupr = average_precision_score(val_true, val_prob)
            aucs.append(auc); auprs.append(aupr)

        if len(aucs) == 0:
            continue

        mean_auc  = float(np.mean(aucs))
        mean_aupr = float(np.mean(auprs))

        if (mean_auc > best_auc) or (np.isclose(mean_auc, best_auc) and mean_aupr > best_aupr):
            best_auc, best_aupr = mean_auc, mean_aupr
            best_cfg = cfg
            best_lists = (aucs, auprs)

    if best_cfg is None:
        best_cfg = dict(HIDDEN=256, HEADS=4, DROPOUT=0.3, LR=1e-3, WEIGHT_DECAY=1e-4)
        inner_stats = dict(auc_mean=np.nan, auc_sd=np.nan, aupr_mean=np.nan, aupr_sd=np.nan,
                           auc_list=[], aupr_list=[])
        return best_cfg, inner_stats

    aucs, auprs = best_lists
    inner_stats = dict(
        auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
        aupr_mean=float(np.mean(auprs)), aupr_sd=float(np.std(auprs)),
        auc_list=aucs, aupr_list=auprs
    )
    return best_cfg, inner_stats

# -----------------------------
# Run: Stratified KFold CV (INDUCTIVE) — NESTED TUNING + CSV
# -----------------------------
def run_kfold_han_style():
    # base (no φ)
    data_base, genes_df, virus_df = build_base_heterodata(apply_undirected=True)
    # full graph with φ (for final outer test)
    data_full = copy.deepcopy(data_base)
    data_full = add_budget_relations(data_full)
    data_full = add_metapath_phi_edges(data_full, include_vv=INCLUDE_VV_METAPATH, min_vv_edges=MIN_VV_EDGES_FOR_USE)

    in_gene  = data_base['gene'].x.size(1)
    in_virus = data_base['virus'].x.size(1)
    y = data_base['gene'].y.cpu().numpy()

    # sanity: φ4 exists on full graph
    phi_names_full = [k[1] for k in data_full.edge_index_dict.keys()
                      if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_full[k].edge_index.numel()>0)]
    if 'phi4' not in phi_names_full:
        raise RuntimeError("φ4 is required but missing in constructed meta-paths on full graph.")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    metrics_list_best = []
    metrics_list_05   = []
    tuning_rows = []   # per-outer-fold tuning summaries

    for fold, (outer_train_idx, outer_test_idx) in enumerate(skf.split(np.zeros_like(y), y), 1):
        print(f"\n====[Outer Fold {fold}]====")
        print(f"train dist -> {dict(zip(*np.unique(y[outer_train_idx], return_counts=True)))}")
        print(f"test  dist -> {dict(zip(*np.unique(y[outer_test_idx],  return_counts=True)))}")

        # (A) tune hyperparameters via inner-CV on outer-train only
        best_cfg, inner_stats = inner_cv_tune(data_base, y, outer_train_idx, in_gene, in_virus)
        print(f"[Outer {fold}] best cfg: {best_cfg} | inner AUROC={inner_stats['auc_mean']:.3f}±{inner_stats['auc_sd']:.3f}, AUPRC={inner_stats['aupr_mean']:.3f}±{inner_stats['aupr_sd']:.3f}")

        # (B) build outer-valid (20% of outer-train) — for threshold selection & early stop target
        inner_train_idx, outer_valid_idx = train_test_split(
            outer_train_idx, test_size=0.2, random_state=SEED, stratify=y[outer_train_idx]
        )

        # induce graphs for outer-train and (train∪valid)
        data_train    = induce_graph_by_genes(data_base, inner_train_idx)
        data_trainval = induce_graph_by_genes(data_base, np.concatenate([inner_train_idx, outer_valid_idx]))

        # scale from inner_train only; apply to trainval and full
        data_train, sc_gene, sc_v = scale_features_in_fold(copy.deepcopy(data_train), inner_train_idx)
        data_trainval    = apply_scalers_to_data(sc_gene, sc_v, data_trainval)
        data_full_scaled = apply_scalers_to_data(sc_gene, sc_v, copy.deepcopy(data_full))

        # ensure φ4 exists on train graph
        phi_names_train = [k[1] for k in data_train.edge_index_dict.keys()
                           if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_train[k].edge_index.numel()>0)]
        if 'phi4' not in phi_names_train:
            raise RuntimeError("φ4 is required but missing in TRAIN meta-paths (inductive).")

        # (C) train final model on outer-train using best cfg
        model = SimpleHAN(in_dim_gene=in_gene, in_dim_virus=in_virus,
                          phi_names=phi_names_train,
                          hidden=best_cfg['HIDDEN'], heads=best_cfg['HEADS'], dropout=best_cfg['DROPOUT'],
                          use_post_mha=USE_POST_MHA, use_virus_token=True)

        model = train_one_fold_inductive(model,
                                         data_train,
                                         data_trainval,
                                         inner_train_idx,
                                         outer_valid_idx,
                                         lr=best_cfg['LR'],
                                         max_epochs=MAX_EPOCHS,
                                         weight_decay=best_cfg['WEIGHT_DECAY'],
                                         patience=PATIENCE)

        # (D) choose threshold on outer-valid
        model.eval()
        with torch.no_grad():
            val_logits = model(data_trainval.x_dict, data_trainval.edge_index_dict)['gene'][torch.as_tensor(outer_valid_idx, device=DEVICE)]
            val_prob   = torch.sigmoid(val_logits).cpu().numpy()
            val_true   = y[outer_valid_idx]
        best_thr = find_best_threshold(val_true, val_prob, grid=201, metric='f1')

        # (E) evaluate on outer-test (independent)
        test_res_best, _, _ = evaluate_split(model, data_full_scaled, outer_test_idx, thr=best_thr)
        test_res_05,   _, _ = evaluate_split(model, data_full_scaled, outer_test_idx, thr=0.5)

        print(f"[Outer {fold}] (best-F1 Thr={test_res_best['threshold']:.2f}) "
              f"AUC={test_res_best['auc_roc']:.3f}  AUPR={test_res_best['auc_pr']:.3f}  "
              f"F1={test_res_best['f1']:.3f}  Acc={test_res_best['accuracy']:.3f}  "
              f"Prec={test_res_best['precision']:.3f}  Rec={test_res_best['recall']:.3f}  "
              f"Brier={test_res_best['brier']:.4f}")

        print(f"[Outer {fold}] (thr=0.50) "
              f"AUC={test_res_05['auc_roc']:.3f}  AUPR={test_res_05['auc_pr']:.3f}  "
              f"F1={test_res_05['f1']:.3f}  Acc={test_res_05['accuracy']:.3f}  "
              f"Prec={test_res_05['precision']:.3f}  Rec={test_res_05['recall']:.3f}  "
              f"Brier={test_res_05['brier']:.4f}")

        metrics_list_best.append(test_res_best)
        metrics_list_05.append(test_res_05)

        # جمع آوری اطلاعات تیونینگ برای جدول
        tuning_rows.append(dict(
            virus = VIRUS_NAME,
            outer_fold = fold,
            HIDDEN = best_cfg['HIDDEN'],
            HEADS  = best_cfg['HEADS'],
            DROPOUT = best_cfg['DROPOUT'],
            LR = best_cfg['LR'],
            WEIGHT_DECAY = best_cfg['WEIGHT_DECAY'],
            Post_MHA = int(USE_POST_MHA),
            K_VH = K_VH_VIRUS_TO_H,
            K_VV = K_VV_PER_VIRUS,
            K_HH_phi2 = K_HH_PHI2,
            K_HH_phi3 = K_HH_PHI3,
            inner_auc_mean = inner_stats['auc_mean'],
            inner_auc_sd   = inner_stats['auc_sd'],
            inner_aupr_mean= inner_stats['aupr_mean'],
            inner_aupr_sd  = inner_stats['aupr_sd'],
            outer_test_auc = test_res_best['auc_roc'],
            outer_test_aupr= test_res_best['auc_pr'],
            outer_test_f1  = test_res_best['f1']
        ))

    keys = ['precision','recall','f1','accuracy','auc_roc','auc_pr','brier']
    print("\n=== Stratified KFold (Test @ best-F1 threshold from outer-valid) ===")
    for k in keys:
        m = float(np.mean([d[k] for d in metrics_list_best]))
        s = float(np.std([d[k] for d in metrics_list_best]))
        print(f"{k:>10s}: {m:.4f} ± {s:.4f}")

    print("\n=== Stratified KFold (Test @ fixed threshold = 0.50) ===")
    for k in keys:
        m = float(np.mean([d[k] for d in metrics_list_05]))
        s = float(np.std([d[k] for d in metrics_list_05]))
        print(f"{k:>10s}: {m:.4f} ± {s:.4f}")

    # نوشتن CSV تیونینگ + ردیف اجماعی
    df_tune = pd.DataFrame(tuning_rows)
    df_tune.to_csv(TUNING_SUMMARY_PATH, index=False)
    print(f"\nSaved per-fold tuning summary -> {TUNING_SUMMARY_PATH}")

    if not df_tune.empty:
        cfg_cols = ['HIDDEN','HEADS','DROPOUT','LR','WEIGHT_DECAY','Post_MHA','K_VH','K_VV','K_HH_phi2','K_HH_phi3']
        consensus = df_tune[cfg_cols].mode().iloc[0].to_dict()

        cons_inner_auc_mean = df_tune['inner_auc_mean'].mean()
        cons_inner_auc_sd   = df_tune['inner_auc_mean'].std()
        cons_inner_aupr_mean= df_tune['inner_aupr_mean'].mean()
        cons_inner_aupr_sd  = df_tune['inner_aupr_mean'].std()

        cons_row = {
            'virus': VIRUS_NAME, **consensus,
            'inner_AUROC_mean±SD': f"{cons_inner_auc_mean:.3f} ± {cons_inner_auc_sd:.3f}",
            'inner_AUPRC_mean±SD': f"{cons_inner_aupr_mean:.3f} ± {cons_inner_aupr_sd:.3f}"
        }
        print("\nConsensus (table-ready):")
        print(cons_row)
        pd.DataFrame([cons_row]).to_csv(f"consensus_hparams_{VIRUS_NAME}.csv", index=False)

# -----------------------------
# (Optional) Final training + Predict all genes
# -----------------------------
def run_fit_and_predict_all():
    # base + full
    data_base, genes_df, virus_df = build_base_heterodata(apply_undirected=True)
    data_full = copy.deepcopy(data_base)
    data_full = add_budget_relations(data_full)
    data_full = add_metapath_phi_edges(data_full, include_vv=INCLUDE_VV_METAPATH, min_vv_edges=MIN_VV_EDGES_FOR_USE)

    in_gene  = data_base['gene'].x.size(1)
    in_virus = data_base['virus'].x.size(1)
    y        = data_base['gene'].y.cpu().numpy()
    all_idx  = np.arange(len(y))

    phi_names = [k[1] for k in data_full.edge_index_dict.keys()
                 if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_full[k].edge_index.numel()>0)]
    if 'phi4' not in phi_names:
        raise RuntimeError("φ4 is required but missing in constructed meta-paths (final inference).")

    tr_full, val_hold = train_test_split(all_idx, test_size=VAL_SIZE, random_state=SEED, stratify=y)

    # [INDUCTIVE] graphs for training and validation
    data_train_full = induce_graph_by_genes(data_base, tr_full)
    data_trainval   = induce_graph_by_genes(data_base, np.concatenate([tr_full, val_hold]))

    # scale from train only, apply to both graphs and to full
    data_train_full, sc_gene_full, sc_virus_full = scale_features_in_fold(copy.deepcopy(data_train_full), tr_full)
    data_trainval = apply_scalers_to_data(sc_gene_full, sc_virus_full, data_trainval)
    data_full     = apply_scalers_to_data(sc_gene_full, sc_virus_full, data_full)

    model_full = SimpleHAN(in_dim_gene=in_gene, in_dim_virus=in_virus,
                           phi_names=[k[1] for k in data_train_full.edge_index_dict.keys() if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_train_full[k].edge_index.numel()>0)],
                           hidden=HIDDEN, heads=HEADS, dropout=DROPOUT,
                           use_post_mha=USE_POST_MHA, use_virus_token=True).to(DEVICE)

    model_full = train_one_fold_inductive(model_full, data_train_full, data_trainval,
                                          tr_full, val_hold,
                                          lr=LR, max_epochs=MAX_EPOCHS, weight_decay=WEIGHT_DECAY,
                                          patience=PATIENCE)

    # choose threshold on val (inductive val graph)
    model_full.eval()
    with torch.no_grad():
        val_logits = model_full(data_trainval.x_dict, data_trainval.edge_index_dict)['gene'][torch.as_tensor(val_hold, device=DEVICE)]
        val_prob   = torch.sigmoid(val_logits).cpu().numpy()
        val_true   = y[val_hold]
    best_thr = find_best_threshold(val_true, val_prob, grid=201, metric='f1')
    print(f"[Final Fit] chosen threshold (best F1 on held-out val): {best_thr:.3f}")

    # ----- Predict ALL genes on full graph -----
    df_all = pd.read_csv(FILE_GENE_ALL)
    interactions_df2 = pd.read_excel(FILE_GV)
    biogrid_df2      = pd.read_csv(FILE_PPI_H)
    virus_df2        = pd.read_excel(FILE_VIRUS)
    vv_df2           = pd.read_excel(FILE_PPI_VV)

    genes_to_index_all = {g: i for i, g in enumerate(df_all['Genes'])}
    virus_to_index2    = {v: i for i, v in enumerate(virus_df2['GenesSymbolVirus'])}

    interactions_df2 = interactions_df2[
        (interactions_df2['Genes'].isin(df_all['Genes'])) &
        (interactions_df2['GenesVirus'].isin(virus_df2['GenesSymbolVirus']))
    ]
    biogrid_df2 = biogrid_df2[
        (biogrid_df2['Ensembl_ID_A'].isin(df_all['Genes'])) &
        (biogrid_df2['Ensembl_ID_B'].isin(df_all['Genes']))
    ]

    X_gene_all  = torch.tensor(df_all.iloc[:, 1:].values, dtype=torch.float)
    X_virus_all = torch.tensor(virus_df2.iloc[:, 1:].values, dtype=torch.float)

    gv2 = torch.tensor([
        [genes_to_index_all[row['Genes']], virus_to_index2[row['GenesVirus']]]
        for _, row in interactions_df2.iterrows()
    ], dtype=torch.long).t().contiguous()

    hh2 = torch.tensor([
        [genes_to_index_all[row['Ensembl_ID_A']], genes_to_index_all[row['Ensembl_ID_B']]]
        for _, row in biogrid_df2.iterrows()
    ], dtype=torch.long).t().contiguous()

    a_col = 'Official_Symbol_A_Zika'; b_col = 'Official_Symbol_B_Zika'
    if a_col in vv_df2.columns and b_col in vv_df2.columns:
        vv2 = torch.tensor([
            [virus_to_index2.get(row[a_col], -1), virus_to_index2.get(row[b_col], -1)]
            for _, row in vv_df2.iterrows()
        ], dtype=torch.long).t().contiguous()
        if vv2.numel() > 0:
            mask_valid = (vv2 >= 0).all(dim=0)
            vv2 = vv2[:, mask_valid]
    else:
        vv2 = torch.empty((2,0), dtype=torch.long)

    data_all = HeteroData()
    data_all['gene'].x  = X_gene_all
    data_all['virus'].x = X_virus_all
    data_all['gene','interacts','virus'].edge_index = gv2
    data_all['gene','interacts','gene' ].edge_index = hh2
    if vv2.numel() > 0:
        data_all['virus','interacts','virus'].edge_index = vv2
    data_all = T.ToUndirected()(data_all)

    data_all = add_budget_relations(data_all)
    data_all = add_metapath_phi_edges(data_all, include_vv=INCLUDE_VV_METAPATH, min_vv_edges=MIN_VV_EDGES_FOR_USE)

    # apply scalers from inductive training
    data_all = apply_scalers_to_data(sc_gene_full, sc_virus_full, data_all)

    model_full.eval()
    data_all = data_all.to(DEVICE)
    with torch.no_grad():
        logits_all = model_full(data_all.x_dict, data_all.edge_index_dict)['gene']
        probs_all  = torch.sigmoid(logits_all).cpu().numpy()

    out_df = pd.DataFrame({
        'Gene_Name': df_all['Genes'],
        'Predicted_Probability': probs_all,
        'Predicted_Label_bestF1': (probs_all >= best_thr).astype(int),
        'Predicted_Label_thr0.5': (probs_all >= 0.5).astype(int)
    })
    out_path = 'Wholedata-HAN-predictions-Zika-final.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions -> {out_path}")
    print(f"Used thresholds: best-F1={best_thr:.3f} and fixed=0.50")

if __name__ == "__main__":
    # 1) Cross-validated evaluation (INDUCTIVE, Nested Tuning) + CSV summary
    run_kfold_han_style()

    # 2) (Optional) Final fit + predict-all (INDUCTIVE training, full-graph inference)
    run_fit_and_predict_all()
