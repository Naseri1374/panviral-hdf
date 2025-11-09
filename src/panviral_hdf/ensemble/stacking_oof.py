"""
Ensemble (Stacking) without leakage for RF + HAN.

"""

from __future__ import annotations
import os, json, math, warnings, re, glob, gzip
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# -------------------------
# Config (adjust paths)
# -------------------------
SEED = 42
N_SPLITS = 10
SPLIT_SCHEME = "8:1:1"   # "8:1:1" (paper) or "han_5x5" (mirror HAN code)
DEVICE_INDEX = 0         # GPU index for HAN

# Labeled gene table (features + Class)
FILE_GENE = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/Zika/ZikaInputdataForDeepL500HDF1000Non39Features_WithClass.csv"

# Final full predictions (for applying ensemble to ALL genes)
HAN_PRED_CSV = "Wholedata-HAN-predictions-Zika-final.csv"  # from run_fit_and_predict_all()
# RF predictions over whole genome: folder with many *.tsv.gz (prob per gene); we’ll average them
RF_PRED_DIR  = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/HRF/Publications/_PREDICTIONS"
RF_FILENAME_FILTER_KEYWORD = "Zika"  # filter filenames by keyword; set None to take all

OUT_DIR = "_ENSEMBLE_OUT"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Imports from your HAN code
# -------------------------
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score)
from sklearn.linear_model import LogisticRegression

from han_hdf_han_final_cv import (
    build_base_heterodata, add_budget_relations, add_metapath_phi_edges,
    induce_graph_by_genes, scale_features_in_fold, apply_scalers_to_data,
    SimpleHAN, train_one_fold_inductive,
    INCLUDE_VV_METAPATH, MIN_VV_EDGES_FOR_USE, HIDDEN, HEADS, DROPOUT, LR, MAX_EPOCHS, WEIGHT_DECAY, PATIENCE
)

DEVICE = torch.device(f"cuda:{DEVICE_INDEX}" if torch.cuda.is_available() else "cpu")

# -------------------------
# Small utils
# -------------------------
def set_seeds(seed: int = SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_metrics_bin(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        AUC   = roc_auc_score(y_true, y_prob),
        AUPR  = average_precision_score(y_true, y_prob),
        F1    = f1_score(y_true, y_pred, zero_division=0),
        Precision = precision_score(y_true, y_pred, zero_division=0),
        Recall    = recall_score(y_true, y_pred, zero_division=0),
        Accuracy  = accuracy_score(y_true, y_pred),
    )

def load_labeled_df(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if 'Genes' not in df.columns or 'Class' not in df.columns:
        raise RuntimeError("Expected columns 'Genes' and 'Class' in labeled file.")
    s = df['Class'].astype(str).str.replace('-', '').str.lower()
    y = np.where(s.eq('hdf'), 1, 0).astype(int)
    return pd.DataFrame({'Ensembl_ID': df['Genes'].astype(str), 'y': y}), df

# -------------------------
# RF — per-fold training producing OOF (val) and test preds (no leakage)
# -------------------------
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def near_zero_var_mask(X: pd.DataFrame, unique_cut: float = 10.0, freq_cut: float = 19.0) -> pd.Series:
    n = X.shape[0]
    mask = pd.Series(False, index=X.columns)
    for c in X.columns:
        v = X[c].astype(str).fillna('NA')
        counts = v.value_counts(dropna=False)
        if len(counts) <= 1:
            mask[c] = True; continue
        percent_unique = 100.0 * len(counts) / n
        most = counts.iloc[0]
        second = counts.iloc[1] if len(counts) > 1 else 0.5
        freq_ratio = (most / max(second, 1e-9)) if second > 0 else np.inf
        if percent_unique <= unique_cut and freq_ratio >= freq_cut:
            mask[c] = True
    return mask

def drop_high_correlation(X: pd.DataFrame, cutoff: float = 0.70) -> List[str]:
    if X.shape[1] <= 1:
        return list(X.columns)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    keep = [c for c in X.columns if c not in to_drop]
    return keep or list(X.columns)

def train_rf_fold_return_probs(X: pd.DataFrame, y: np.ndarray,
                               train_idx, val_idx, test_idx,
                               seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    X_train = X.iloc[train_idx].copy()
    X_val   = X.iloc[val_idx].copy()
    X_test  = X.iloc[test_idx].copy()

    nzv = near_zero_var_mask(X_train)
    cols = [c for c in X_train.columns if not nzv[c]]
    X_train, X_val, X_test = X_train[cols], X_val[cols], X_test[cols]

    pp = Pipeline([('imp', SimpleImputer(strategy='median')),
                   ('sc', StandardScaler())])
    Xtr = pd.DataFrame(pp.fit_transform(X_train), index=X_train.index, columns=cols)
    Xva = pd.DataFrame(pp.transform(X_val),   index=X_val.index,   columns=cols)
    Xte = pd.DataFrame(pp.transform(X_test),  index=X_test.index,  columns=cols)

    sel = drop_high_correlation(Xtr, cutoff=0.70)
    Xtr, Xva, Xte = Xtr[sel], Xva[sel], Xte[sel]

    p = Xtr.shape[1]
    mtry = max(1, int(round(math.sqrt(p))))
    rf = RandomForestClassifier(
        n_estimators=800, max_features=mtry, random_state=seed, n_jobs=-1, class_weight=None
    )
    rf.fit(Xtr, y[train_idx])

    prob_val  = rf.predict_proba(Xva)[:, 1]
    prob_test = rf.predict_proba(Xte)[:, 1]
    return prob_val, prob_test

# -------------------------
# HAN — per-fold OOF (val) and test probs via inductive pipeline (no leakage)
# -------------------------
def build_full_graph_scaled_later(data_base):
    data_full = add_budget_relations(copy_like(data_base))
    data_full = add_metapath_phi_edges(data_full, include_vv=INCLUDE_VV_METAPATH, min_vv_edges=MIN_VV_EDGES_FOR_USE)
    return data_full

def copy_like(data):
    import copy as _copy
    return _copy.deepcopy(data)

def han_fold_return_probs(data_base,
                          in_gene: int, in_virus: int, y_all: np.ndarray,
                          train_idx, val_idx, test_idx,
                          seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    data_train    = induce_graph_by_genes(data_base, np.asarray(train_idx))
    data_trainval = induce_graph_by_genes(data_base, np.asarray(list(train_idx) + list(val_idx)))
    data_full     = build_full_graph_scaled_later(data_base)

    data_train, sc_gene, sc_v = scale_features_in_fold(data_train, np.asarray(train_idx))
    data_trainval = apply_scalers_to_data(sc_gene, sc_v, data_trainval)
    data_full     = apply_scalers_to_data(sc_gene, sc_v, data_full)

    phi_names_train = [k[1] for k in data_train.edge_index_dict.keys()
                       if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_train[k].edge_index.numel()>0)]
    if 'phi4' not in phi_names_train:
        raise RuntimeError("φ4 required but missing in TRAIN graph.")

    model = SimpleHAN(in_dim_gene=in_gene, in_dim_virus=in_virus,
                      phi_names=phi_names_train,
                      hidden=HIDDEN, heads=HEADS, dropout=DROPOUT,
                      use_post_mha=True, use_virus_token=True).to(DEVICE)

    model = train_one_fold_inductive(model, data_train, data_trainval,
                                     np.asarray(train_idx), np.asarray(val_idx),
                                     lr=LR, max_epochs=MAX_EPOCHS, weight_decay=WEIGHT_DECAY,
                                     patience=PATIENCE)

    model.eval()
    with torch.no_grad():
        logits_val  = model(data_trainval.x_dict, data_trainval.edge_index_dict)['gene'][torch.as_tensor(val_idx,  device=DEVICE)]
        prob_val    = torch.sigmoid(logits_val).detach().cpu().numpy()

        logits_test = model(data_full.x_dict,     data_full.edge_index_dict)['gene'][torch.as_tensor(test_idx, device=DEVICE)]
        prob_test   = torch.sigmoid(logits_test).detach().cpu().numpy()

    return prob_val, prob_test

# -------------------------
# RF + HAN → OOF VAL + TEST per fold
# -------------------------
def make_splits(y: np.ndarray, scheme: str = SPLIT_SCHEME, seed: int = SEED):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (train_pool, test_idx) in enumerate(skf.split(np.zeros_like(y), y), 1):
        if scheme == "han_5x5":
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
            te_labels = y[test_idx]
            val_rel, test_rel = next(sss.split(np.zeros_like(te_labels), te_labels))
            val_idx = test_idx[val_rel]
            test2_idx = test_idx[test_rel]
            train_idx = train_pool
        else:
            train_full = train_pool
            val_frac_in_pool = 0.1 / 0.9  # so that val is 10% of ALL
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_in_pool, random_state=seed)
            tr_labels = y[train_full]
            tr_rel, va_rel = next(sss.split(np.zeros_like(tr_labels), tr_labels))
            train_idx = train_full[tr_rel]
            val_idx   = train_full[va_rel]
            test2_idx = test_idx
        yield fold, train_idx, val_idx, test2_idx

# -------------------------
# Load final full predictions for ALL genes (for deployment)
# -------------------------
def load_han_all_preds(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    gene_col = None
    for cand in ['Gene_Name','Genes','Gene','Ensembl_ID']:
        if cand in df.columns: gene_col = cand; break
    pcol = None
    for cand in ['Predicted_Probability','Prob','Probability','Prob_HDF','Score']:
        if cand in df.columns: pcol = cand; break
    if gene_col is None or pcol is None:
        raise RuntimeError("Columns for 'gene id' and 'prob' not found in HAN final CSV.")
    out = df[[gene_col, pcol]].copy()
    out.columns = ['Ensembl_ID','Prob_HAN']
    return out

def load_rf_all_preds(pred_dir: str, keyword: Optional[str] = None) -> pd.DataFrame:
    pats = glob.glob(os.path.join(pred_dir, "*.tsv.gz"))
    if keyword:
        pats = [p for p in pats if re.search(keyword, os.path.basename(p), flags=re.IGNORECASE)]
    if not pats:
        raise RuntimeError(f"No RF prediction files in {pred_dir}.")
    dfs = []
    for p in pats:
        with gzip.open(p, 'rt') as f:
            d = pd.read_csv(f, sep='\t')
        need = ['Ensembl_ID','Prob_HDF']
        if not set(need).issubset(d.columns): 
            continue
        d = d[need].rename(columns={'Prob_HDF': f"Prob_RF_{os.path.splitext(os.path.basename(p))[0]}"})
        dfs.append(d)
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on='Ensembl_ID', how='outer')
    rf_cols = [c for c in base.columns if c.startswith("Prob_RF_")]
    base['Prob_RF'] = base[rf_cols].mean(axis=1)
    return base[['Ensembl_ID','Prob_RF']].copy()

# -------------------------
# Main
# -------------------------
def main():
    set_seeds(SEED)

    # 0) Labeled data (for RF features + labels)
    lab_short, lab_full = load_labeled_df(FILE_GENE)
    gene_ids = lab_short['Ensembl_ID'].values.astype(str)
    y = lab_short['y'].values.astype(int)

    # RF feature matrix from FILE_GENE (all non-id/non-class columns)
    X_rf = lab_full.drop(columns=['Genes','Class'])
    X_rf.columns = [str(c) for c in X_rf.columns]
    X_rf = X_rf.apply(pd.to_numeric, errors='coerce')
    if X_rf.isna().any().any():
        X_rf = X_rf.fillna(X_rf.median())

    # 1) Build base hetero graph for HAN (once)
    data_base, genes_df, virus_df = build_base_heterodata(apply_undirected=True)
    in_gene  = data_base['gene'].x.size(1)
    in_virus = data_base['virus'].x.size(1)

    # 2) OOF containers (row-wise to keep Fold)
    val_rows = []
    test_rows = []
    splits_log = []

    # 3) Iterate folds (shared splits)
    for fold, train_idx, val_idx, test_idx in make_splits(y, scheme=SPLIT_SCHEME, seed=SEED):
        print(f"[Fold {fold}] sizes -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        splits_log.append({
            'fold': int(fold),
            'train_idx': list(map(int, np.asarray(train_idx).tolist())),
            'val_idx':   list(map(int, np.asarray(val_idx).tolist())),
            'test_idx':  list(map(int, np.asarray(test_idx).tolist())),
        })

        # RF per fold
        prob_rf_val,  prob_rf_test  = train_rf_fold_return_probs(X_rf, y, train_idx, val_idx, test_idx)

        # HAN per fold
        prob_han_val, prob_han_test = han_fold_return_probs(
            data_base, in_gene, in_virus, y,
            train_idx, val_idx, test_idx, seed=SEED
        )

        # Append OOF val rows
        for idx, p_rf, p_han in zip(val_idx, prob_rf_val, prob_han_val):
            val_rows.append({
                'Index': int(idx),
                'Fold': int(fold),
                'Ensembl_ID': gene_ids[idx],
                'y': int(y[idx]),
                'Prob_RF': float(p_rf),
                'Prob_HAN': float(p_han),
            })

        # Collect test rows
        for i, idx in enumerate(test_idx):
            test_rows.append({
                'Index': int(idx),
                'Fold': int(fold),
                'Ensembl_ID': gene_ids[idx],
                'y': int(y[idx]),
                'Prob_RF': float(prob_rf_test[i]),
                'Prob_HAN': float(prob_han_test[i]),
            })

    # 4) Save OOF val table & test table
    oof_val_df = pd.DataFrame(val_rows).sort_values(['Fold','Index']).reset_index(drop=True)
    oof_val_df.to_csv(os.path.join(OUT_DIR, "oof_val_table.csv"), index=False)

    test_df = pd.DataFrame(test_rows).sort_values(['Fold','Index']).reset_index(drop=True)
    test_df.to_csv(os.path.join(OUT_DIR, "perfold_test_table.csv"), index=False)

    # 5) Fit stacking meta-learner on OOF(val), evaluate on test
    X_val = oof_val_df[['Prob_RF','Prob_HAN']].values
    y_val = oof_val_df['y'].values.astype(int)

    Cs = [0.01, 0.1, 1, 3, 10, 30]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_auc, best_C = -1.0, 1.0
    for C in Cs:
        aucs=[]
        for tr, va in skf.split(X_val, y_val):
            lr = LogisticRegression(C=C, solver='liblinear')
            lr.fit(X_val[tr], y_val[tr])
            p = lr.predict_proba(X_val[va])[:,1]
            aucs.append(roc_auc_score(y_val[va], p))
        m = float(np.mean(aucs))
        if m > best_auc:
            best_auc, best_C = m, C
    meta = LogisticRegression(C=best_C, solver='liblinear')
    meta.fit(X_val, y_val)

    # Evaluate on test (concatenate all folds)
    test_mat = test_df[['Prob_RF','Prob_HAN']].values
    test_prob_ens = meta.predict_proba(test_mat)[:,1]

    m_base_rf  = evaluate_metrics_bin(test_df['y'].values, test_df['Prob_RF'].values)
    m_base_han = evaluate_metrics_bin(test_df['y'].values, test_df['Prob_HAN'].values)
    m_ens      = evaluate_metrics_bin(test_df['y'].values, test_prob_ens)

    met_tbl = pd.DataFrame([
        {'Model':'RF',  **m_base_rf},
        {'Model':'HAN', **m_base_han},
        {'Model':'Stacking (LR, OOF→meta)', **m_ens},
    ])
    met_tbl.to_csv(os.path.join(OUT_DIR, "stacking_metrics_test.csv"), index=False)

    info = {
        'scheme': SPLIT_SCHEME,
        'n_splits': N_SPLITS,
        'seed': SEED,
        'meta_best_C': float(best_C),
        'meta_cv_auc_on_OOF_val': float(best_auc),
        'splits': splits_log
    }
    with open(os.path.join(OUT_DIR, "stacking_meta_info.json"), 'w') as f:
        json.dump(info, f, indent=2)

    print("\n=== Test metrics (aggregated across folds) ===")
    print(met_tbl.to_string(index=False))

    # 6) Apply ensemble to ALL genes (final HAN+RF global preds)
    print("\nApplying ensemble to ALL genes...")
    HAN_all = load_han_all_preds(HAN_PRED_CSV)
    RF_all  = load_rf_all_preds(RF_PRED_DIR, keyword=RF_FILENAME_FILTER_KEYWORD)
    ALL = pd.merge(HAN_all, RF_all, on='Ensembl_ID', how='inner').dropna()
    if ALL.empty:
        warnings.warn("No overlap between HAN and RF all-gene predictions.")
    X_all = ALL[['Prob_RF','Prob_HAN']].values
    ALL['Prob_Ensemble'] = meta.predict_proba(X_all)[:,1]
    ALL = ALL.sort_values('Prob_Ensemble', ascending=False).reset_index(drop=True)
    ALL.to_csv(os.path.join(OUT_DIR, "ensemble_predictions_all_genes.csv"), index=False)
    print(f"Saved: {os.path.join(OUT_DIR, 'ensemble_predictions_all_genes.csv')}")

if __name__ == "__main__":
    main()
