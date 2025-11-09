# ablation.py

import os, copy, random, re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv

# ---- bring in core utilities / constants from final CV module ----
from .han_final_cv import (
    # model + training utils
    SimpleHAN, scale_features_in_fold, apply_scalers_to_data,
    train_one_fold_inductive, evaluate_split, find_best_threshold,
    add_budget_relations, add_metapath_phi_edges, build_base_heterodata,

    # graph helpers
    topk_per_src, unique_edges, compose_edges, aggregate_virus_mean,
    induce_graph_by_genes, make_binary_labels, get_search_space,

    # paths + constants
    FILE_GENE, FILE_GV, FILE_PPI_H, FILE_VIRUS, FILE_PPI_VV,
    DEVICE, VIRUS_NAME, SEED, N_SPLITS, K_INNER,
    K_HH_PHI2, K_HH_PHI3, K_HV_GENE_TO_V, K_VH_VIRUS_TO_H, K_VV_PER_VIRUS,
    HIDDEN, HEADS, DROPOUT, LR, MAX_EPOCHS, WEIGHT_DECAY, PATIENCE,
    INCLUDE_VV_METAPATH, MIN_VV_EDGES_FOR_USE
)

# -------------------------------
# 1) Feature-group patterns (regex prefixes)
# -------------------------------
FEATURE_GROUP_PATTERNS = {
    # 1) Sequence descriptors (DNA/Protein)
    "sequence": r"^(seq\.|aa_|dna_|kmer_|pseaa_|psedna_|conjoint_|qsorder_)",
    # 2) GO/Pathway enrichment + KEGG/Reactome counts
    "go_pathway": r"^(go_|kegg_|reactome_)",
    # 3) Protein domains & motifs & UTRs/PTMs
    "domains_motifs": r"^(pfam_|domain_|coil_|tmhmm_|ptm_|signalpeptide_|utr_)",
    # 4) Conservation / Homology / Orthologs
    "conservation": r"^(psiblast_|homolog_|ortholog_|ka_ks_|evalue_)",
    # 5) PPI topology & roles
    "ppi_topology": r"^(degree_|betweenness_|closeness_|pagerank_|refex_|rolx_|egonet_)",
    # 6) Embeddings + Localization
    "emb_loc": r"^(n2v_|node2vec_|deeploc_)"
}

def drop_feature_groups(features_df: pd.DataFrame, groups_to_drop: List[str]) -> pd.DataFrame:
    """
    حذف گروه‌های فیچر از DataFrame (ستون‌های 'Genes' و 'Class' حفظ می‌شوند).
    گروه‌ها را با کلیدهای FEATURE_GROUP_PATTERNS بده.
    """
    keep_cols = ['Genes', 'Class']
    patterns = [re.compile(FEATURE_GROUP_PATTERNS[g]) for g in groups_to_drop]
    drop_cols = []
    for c in features_df.columns:
        if c in keep_cols:
            continue
        if any(p.match(c) for p in patterns):
            drop_cols.append(c)
    out = features_df.drop(columns=drop_cols, errors='ignore')
    print(f"[Ablation] Dropped feature groups={groups_to_drop} -> removed {len(drop_cols)} columns; kept {out.shape[1]-2} features.")
    return out

# -------------------------------
# 2) Edge budgets with toggles (HH/HV/VV)
# -------------------------------
def add_budget_relations_with_toggles(data: HeteroData,
                                      use_hh: bool = True,
                                      use_hv: bool = True,
                                      use_vv: bool = True) -> HeteroData:
    Ng = data['gene'].x.size(0)
    Nv = data['virus'].x.size(0)

    if use_hh and ('gene','interacts','gene') in data.edge_index_dict:
        hh_e = data[('gene','interacts','gene')].edge_index
        data['gene','hh2','gene'].edge_index = topk_per_src(hh_e, Ng, K_HH_PHI2, seed=SEED)
        data['gene','hh3','gene'].edge_index = topk_per_src(hh_e, Ng, K_HH_PHI3, seed=SEED+1)

    if use_hv and ('gene','interacts','virus') in data.edge_index_dict:
        hv_e = data[('gene','interacts','virus')].edge_index
        data['gene','hv','virus'].edge_index = topk_per_src(hv_e, Ng, K_HV_GENE_TO_V, seed=SEED+2)

    if use_hv and ('virus','rev_interacts','gene') in data.edge_index_dict:
        vh_e = data[('virus','rev_interacts','gene')].edge_index
        data['virus','vh','gene'].edge_index = topk_per_src(vh_e, Nv, K_VH_VIRUS_TO_H, seed=SEED+3)

    if use_vv and ('virus','interacts','virus') in data.edge_index_dict:
        vv_e = data[('virus','interacts','virus')].edge_index
        data['virus','vv','virus'].edge_index = topk_per_src(vv_e, Nv, K_VV_PER_VIRUS, seed=SEED+4)

    return data

def add_metapath_phi_edges_allowed(data: HeteroData,
                                   allow_phi: List[int],
                                   enforce_phi4: bool) -> HeteroData:
    """
    ساخت متاپث‌های مجاز (phi1..phi4) با توجه به allow_phi.
    enforce_phi4=True: اگر φ4 خواسته شده ولی امکان ساختش نیست، ارور بده.
    """
    Ng = data['gene'].x.size(0); Nv = data['virus'].x.size(0)
    device = data['gene'].x.device

    def get_e(key):
        return data[key].edge_index if key in data.edge_index_dict else torch.empty((2,0), dtype=torch.long, device=device)

    hv  = get_e(('gene','hv','virus'))
    vh  = get_e(('virus','vh','gene')) if ('virus','vh','gene') in data.edge_index_dict else get_e(('virus','interacts','gene'))
    hh2 = get_e(('gene','hh2','gene'))
    hh3 = get_e(('gene','hh3','gene'))
    vv  = get_e(('virus','vv','virus'))

    if 2 in allow_phi and hh2.numel() > 0:
        data['gene','phi2','gene'].edge_index = unique_edges(hh2)

    if 1 in allow_phi and hv.numel() > 0 and vh.numel() > 0:
        phi1 = compose_edges(hv, vh, Ng, Nv)
        if phi1.numel() > 0:
            data['gene','phi1','gene'].edge_index = phi1

    if 3 in allow_phi and hh3.numel() > 0 and hv.numel() > 0 and vh.numel() > 0:
        tmp = compose_edges(hh3, hv, Ng, Ng)
        phi3 = compose_edges(tmp, vh, Ng, Nv)
        if phi3.numel() > 0:
            data['gene','phi3','gene'].edge_index = phi3

    if 4 in allow_phi:
        if vv.numel() > 0 and hv.numel() > 0 and vh.numel() > 0:
            tmp = compose_edges(hv, vv, Ng, Nv)
            phi4 = compose_edges(tmp, vh, Ng, Nv)
            if phi4.numel() > 0:
                data['gene','phi4','gene'].edge_index = phi4
        elif enforce_phi4:
            raise RuntimeError("φ4 required but insufficient VV/HV/VH to compose.")

    # fallback اگر هیچ φ ساخته نشد:
    has_any_phi = any(k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data[k].edge_index.numel()>0
                      for k in data.edge_index_dict.keys())
    if not has_any_phi:
        if ('gene','interacts','gene') in data.edge_index_dict:
            data['gene','phi2','gene'].edge_index = unique_edges(data[('gene','interacts','gene')].edge_index)
        else:
            raise RuntimeError("No meta-path edges could be constructed.")
    return data

# -------------------------------
# 3) Flexible model for ablation (toggle φ4 requirement / semantic-attn / virus-token)
# -------------------------------
class SimpleHANAblation(SimpleHAN):
    def __init__(self, *args,
                 require_phi4: bool = True,
                 use_semantic_attention: bool = True,
                 use_virus_token: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.require_phi4 = require_phi4
        self.use_semantic_attention = use_semantic_attention
        self.use_virus_token = use_virus_token

    def forward(self, x_dict, edge_index_dict):
        xg = self.proj_gene(x_dict['gene'])
        xv = self.proj_virus(x_dict['virus'])

        phi_keys = [('gene', f'phi{i}', 'gene') for i in [1,2,3,4]]
        phi_keys = [k for k in phi_keys if (k in edge_index_dict and edge_index_dict[k].numel() > 0)]
        if not phi_keys:
            raise RuntimeError("No φ-edges in edge_index_dict")
        if self.require_phi4:
            if ('gene','phi4','gene') not in edge_index_dict or edge_index_dict[('gene','phi4','gene')].numel() == 0:
                raise RuntimeError("φ4 required but missing (ablation model).")

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

        if self.use_semantic_attention:
            fused, _alpha = self.semantic_att(H)
        else:
            fused = H.mean(dim=1)

        if self.use_post_mha:
            hv_key = ('gene','hv','virus') if ('gene','hv','virus') in edge_index_dict else ('gene','interacts','virus')
            if self.use_virus_token and hv_key in edge_index_dict and edge_index_dict[hv_key].numel() > 0:
                zv = aggregate_virus_mean(xv, edge_index_dict[hv_key], xg.size(0))
                tokens = torch.stack([xg, fused, zv], dim=1)
            else:
                tokens = torch.stack([xg, fused], dim=1)
            fused = self.post_mha(tokens)

        logit = self.cls(fused).squeeze(-1)
        return {'gene': logit}

# -------------------------------
# 4) Base heterodata with custom features (after dropping groups)
# -------------------------------
def build_base_heterodata_custom_features(features_df: pd.DataFrame,
                                          apply_undirected: bool = True):
    interactions_df = pd.read_excel(FILE_GV)
    biogrid_df      = pd.read_csv(FILE_PPI_H)
    virus_df        = pd.read_excel(FILE_VIRUS)
    vv_df           = pd.read_excel(FILE_PPI_VV)

    for df, req in [(interactions_df, ['Genes','GenesVirus']),
                    (biogrid_df, ['Ensembl_ID_A','Ensembl_ID_B']),
                    (virus_df, ['GenesSymbolVirus'])]:
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
    virus_to_index  = {v: i for i, v in enumerate(virus_df['GenesSymbolVirus'])}

    feat_cols = [c for c in features_df.columns if c not in ['Genes','Class']]
    X_gene  = torch.tensor(features_df[feat_cols].values, dtype=torch.float)
    X_virus = torch.tensor(virus_df.iloc[:, 1:].values, dtype=torch.float)

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
        data = T.ToUndirected()(data)
    return data, features_df, virus_df

# -------------------------------
# 5) Inner-CV tuning specialized for ablation
# -------------------------------
def inner_cv_tune_ablation(data_base: HeteroData,
                           y: np.ndarray,
                           outer_train_idx: np.ndarray,
                           in_gene: int,
                           in_virus: int,
                           model_kwargs: dict,
                           allow_phi: List[int],
                           use_hh: bool = True,
                           use_hv: bool = True,
                           use_vv: bool = True,
                           enforce_phi4: bool = False):
    inner_skf = StratifiedKFold(n_splits=K_INNER, shuffle=True, random_state=SEED)
    search_space = get_search_space()
    best_cfg, best_auc, best_aupr, best_lists = None, -1.0, -1.0, None

    for cfg in search_space:
        aucs, auprs = [], []
        for tr_rel, va_rel in inner_skf.split(np.zeros_like(y[outer_train_idx]), y[outer_train_idx]):
            tr_idx = outer_train_idx[tr_rel]; va_idx = outer_train_idx[va_rel]

            data_tr = induce_graph_by_genes(data_base, tr_idx)
            data_tv = induce_graph_by_genes(data_base, np.concatenate([tr_idx, va_idx]))

            data_tr = add_budget_relations_with_toggles(data_tr, use_hh, use_hv, use_vv)
            data_tv = add_budget_relations_with_toggles(data_tv, use_hh, use_hv, use_vv)

            data_tr = add_metapath_phi_edges_allowed(data_tr, allow_phi, enforce_phi4=enforce_phi4)
            data_tv = add_metapath_phi_edges_allowed(data_tv, allow_phi, enforce_phi4=enforce_phi4)

            data_tr, scg, scv = scale_features_in_fold(copy.deepcopy(data_tr), tr_idx)
            data_tv = apply_scalers_to_data(scg, scv, data_tv)

            phi_names_train = [k[1] for k in data_tr.edge_index_dict.keys()
                               if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_tr[k].edge_index.numel()>0)]

            model = SimpleHANAblation(in_dim_gene=in_gene, in_dim_virus=in_virus,
                                      phi_names=phi_names_train,
                                      hidden=cfg['HIDDEN'], heads=cfg['HEADS'], dropout=cfg['DROPOUT'],
                                      use_post_mha=model_kwargs.get('use_post_mha', True),
                                      use_virus_token=model_kwargs.get('use_virus_token', True),
                                      require_phi4=model_kwargs.get('require_phi4', False),
                                      use_semantic_attention=model_kwargs.get('use_semantic_attention', True))

            model = train_one_fold_inductive(model, data_tr, data_tv, tr_idx, va_idx,
                                             lr=cfg['LR'], max_epochs=MAX_EPOCHS,
                                             weight_decay=cfg['WEIGHT_DECAY'], patience=PATIENCE)

            model.eval()
            with torch.no_grad():
                v_logits = model(data_tv.x_dict, data_tv.edge_index_dict)['gene'][torch.as_tensor(va_idx, device=DEVICE)]
                v_prob   = torch.sigmoid(v_logits).cpu().numpy()
                v_true   = y[va_idx]
            aucs.append(roc_auc_score(v_true, v_prob))
            auprs.append(average_precision_score(v_true, v_prob))

        if len(aucs) == 0:
            continue
        m_auc, m_aupr = float(np.mean(aucs)), float(np.mean(auprs))
        if (m_auc > best_auc) or (np.isclose(m_auc, best_auc) and m_aupr > best_aupr):
            best_auc, best_aupr = m_auc, m_aupr
            best_cfg, best_lists = cfg, (aucs, auprs)

    if best_cfg is None:
        best_cfg = dict(HIDDEN=256, HEADS=4, DROPOUT=0.3, LR=1e-3, WEIGHT_DECAY=1e-4)
        return best_cfg, dict(auc_mean=np.nan, auc_sd=np.nan, aupr_mean=np.nan, aupr_sd=np.nan)

    aucs, auprs = best_lists
    return best_cfg, dict(auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                          aupr_mean=float(np.mean(auprs)), aupr_sd=float(np.std(auprs)))

# -------------------------------
# 6) One ablation runner
# -------------------------------
@dataclass
class AblationCfg:
    name: str
    allow_phi: List[int]         # e.g., [1,2,3,4] or [1,2,3]
    use_hh: bool = True
    use_hv: bool = True
    use_vv: bool = True
    require_phi4: bool = True
    use_semantic_attention: bool = True
    use_post_mha: bool = True
    use_virus_token: bool = True
    drop_feature_groups: Optional[List[str]] = None  # e.g., ['ppi_topology']

def run_single_ablation(cfg: AblationCfg, out_csv_path: str):
    print(f"\n===== Ablation: {cfg.name} =====")

    # 1) load + optionally drop feature groups
    feat_df = pd.read_csv(FILE_GENE)
    if cfg.drop_feature_groups:
        feat_df = drop_feature_groups(feat_df, cfg.drop_feature_groups)
    data_base, genes_df, virus_df = build_base_heterodata_custom_features(feat_df, apply_undirected=True)

    # 2) build full-graph budgets + allowed meta-paths (for outer-test)
    data_full = copy.deepcopy(data_base)
    data_full = add_budget_relations_with_toggles(data_full, use_hh=cfg.use_hh, use_hv=cfg.use_hv, use_vv=cfg.use_vv)
    data_full = add_metapath_phi_edges_allowed(data_full, cfg.allow_phi, enforce_phi4=cfg.require_phi4)

    in_gene  = data_base['gene'].x.size(1)
    in_virus = data_base['virus'].x.size(1)
    y = data_base['gene'].y.cpu().numpy()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for fold, (outer_train_idx, outer_test_idx) in enumerate(skf.split(np.zeros_like(y), y), 1):
        # inner-CV tuning on outer-train
        best_cfg, inner_stats = inner_cv_tune_ablation(
            data_base, y, outer_train_idx, in_gene, in_virus,
            model_kwargs=dict(require_phi4=cfg.require_phi4,
                              use_semantic_attention=cfg.use_semantic_attention,
                              use_post_mha=cfg.use_post_mha,
                              use_virus_token=cfg.use_virus_token),
            allow_phi=cfg.allow_phi,
            use_hh=cfg.use_hh, use_hv=cfg.use_hv, use_vv=cfg.use_vv,
            enforce_phi4=cfg.require_phi4
        )

        # split outer-train -> inner_train + outer-valid
        inner_train_idx, outer_valid_idx = train_test_split(
            outer_train_idx, test_size=0.2, random_state=SEED, stratify=y[outer_train_idx]
        )

        # inductive graphs + budgets/metapaths
        data_train = induce_graph_by_genes(data_base, inner_train_idx)
        data_tv    = induce_graph_by_genes(data_base, np.concatenate([inner_train_idx, outer_valid_idx]))
        data_train = add_budget_relations_with_toggles(data_train, cfg.use_hh, cfg.use_hv, cfg.use_vv)
        data_tv    = add_budget_relations_with_toggles(data_tv,    cfg.use_hh, cfg.use_hv, cfg.use_vv)
        data_train = add_metapath_phi_edges_allowed(data_train, cfg.allow_phi, enforce_phi4=cfg.require_phi4)
        data_tv    = add_metapath_phi_edges_allowed(data_tv,    cfg.allow_phi, enforce_phi4=cfg.require_phi4)

        # scale by inner-train; apply to tv and full
        data_train, scg, scv = scale_features_in_fold(copy.deepcopy(data_train), inner_train_idx)
        data_tv     = apply_scalers_to_data(scg, scv, data_tv)
        data_full_  = apply_scalers_to_data(scg, scv, copy.deepcopy(data_full))

        phi_names_train = [k[1] for k in data_train.edge_index_dict.keys()
                           if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_train[k].edge_index.numel()>0)]

        model = SimpleHANAblation(in_dim_gene=in_gene, in_dim_virus=in_virus,
                                  phi_names=phi_names_train,
                                  hidden=best_cfg['HIDDEN'], heads=best_cfg['HEADS'], dropout=best_cfg['DROPOUT'],
                                  use_post_mha=cfg.use_post_mha, use_virus_token=cfg.use_virus_token,
                                  require_phi4=cfg.require_phi4,
                                  use_semantic_attention=cfg.use_semantic_attention)

        model = train_one_fold_inductive(model, data_train, data_tv,
                                         inner_train_idx, outer_valid_idx,
                                         lr=best_cfg['LR'], max_epochs=MAX_EPOCHS,
                                         weight_decay=best_cfg['WEIGHT_DECAY'], patience=PATIENCE)

        # threshold from outer-valid
        model.eval()
        with torch.no_grad():
            v_logits = model(data_tv.x_dict, data_tv.edge_index_dict)['gene'][torch.as_tensor(outer_valid_idx, device=DEVICE)]
            v_prob   = torch.sigmoid(v_logits).cpu().numpy()
            v_true   = y[outer_valid_idx]
        thr = find_best_threshold(v_true, v_prob, grid=201, metric='f1')

        # evaluate on outer-test (full graph scaled)
        test_res_best, _, _ = evaluate_split(model, data_full_, outer_test_idx, thr=thr)
        rows.append(dict(
            virus=VIRUS_NAME, ablation=cfg.name, outer_fold=fold,
            AUROC=test_res_best['auc_roc'],
            AUPRC=test_res_best['auc_pr'],
            F1=test_res_best['f1'],
            Acc=test_res_best['accuracy'],
            Prec=test_res_best['precision'],
            Rec=test_res_best['recall'],
            Brier=test_res_best['brier'],
            thr=thr,
            HIDDEN=best_cfg['HIDDEN'], HEADS=best_cfg['HEADS'],
            DROPOUT=best_cfg['DROPOUT'], LR=best_cfg['LR'],
            WD=best_cfg['WEIGHT_DECAY'],
            inner_auc_mean=inner_stats.get('auc_mean', np.nan),
            inner_auc_sd=inner_stats.get('auc_sd', np.nan),
            inner_aupr_mean=inner_stats.get('aupr_mean', np.nan),
            inner_aupr_sd=inner_stats.get('aupr_sd', np.nan)
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    print(f"[Ablation:{cfg.name}] saved -> {out_csv_path}")
    # paper-style summary
    for m in ['AUROC','AUPRC','F1','Acc','Prec','Rec','Brier']:
        print(f"{m}: {df[m].mean():.3f} ± {df[m].std():.3f}")
    return df

# -------------------------------
# 7) Ablation list + launcher for (e.g.) Table 3
# -------------------------------
def run_ablation_table3():
    out_path = f"ablation_table3_{VIRUS_NAME}.csv"
    ablations = [
        AblationCfg(name="Full",      allow_phi=[1,2,3,4], require_phi4=True,
                    use_semantic_attention=True, use_post_mha=True, use_virus_token=True,
                    drop_feature_groups=[]),
        AblationCfg(name="-phi4",     allow_phi=[1,2,3],   require_phi4=False,
                    use_semantic_attention=True, use_post_mha=True, use_virus_token=True),
        AblationCfg(name="NoSemAttn", allow_phi=[1,2,3,4], require_phi4=True,
                    use_semantic_attention=False, use_post_mha=True, use_virus_token=True),

        AblationCfg(name="-PostMHA",  allow_phi=[1,2,3,4], require_phi4=True,
                    use_semantic_attention=True, use_post_mha=False, use_virus_token=True),
        AblationCfg(name="-VirusTok", allow_phi=[1,2,3,4], require_phi4=True,
                    use_semantic_attention=True, use_post_mha=True, use_virus_token=False),

        # Feature ablations (ASCII-safe names for files)
        AblationCfg(name="-Topology", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['ppi_topology']),
        AblationCfg(name="-Node2Vec_DeepLoc", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['emb_loc']),
        AblationCfg(name="-GO_Pathway", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['go_pathway']),
        AblationCfg(name="-Sequence",  allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['sequence']),
        AblationCfg(name="-Conservation", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['conservation']),
        AblationCfg(name="-Domains", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['domains_motifs']),
    ]

    all_rows = []
    for cfg in ablations:
        df = run_single_ablation(cfg, out_csv_path=f"ablt_{cfg.name}_{VIRUS_NAME}.csv")
        all_rows.append(df)
    big = pd.concat(all_rows, axis=0).reset_index(drop=True)
    big.to_csv(out_path, index=False)
    print(f"\n=== TABLE 3 (outer-test, all ablations) -> {out_path} ===")

# Optional direct run:
# if __name__ == "__main__":
#     run_ablation_table3()

