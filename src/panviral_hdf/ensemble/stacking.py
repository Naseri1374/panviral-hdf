# src/panviral_hdf/ensemble/stacking.py
# ------------------------------------------------------------------
# Ensemble (final predictions): HAN + RF
# - Reads final prediction files (HAN CSV, multiple RF tsv.gz)
# - Compares Weighted Average vs Logistic Regression stacking via CV
# - Saves chosen method metadata + ensemble predictions (labeled & all)
# ------------------------------------------------------------------

from __future__ import annotations
import os
import re
import json
import gzip
import glob
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)

# اختیاری برای نگاشت (اگر لازم شد)
try:
    import mygene  # pip install mygene
except Exception:
    mygene = None
    warnings.warn("mygene نصب نیست؛ اگر نگاشت لازم شود، ممکن است join کامل نشود.")

# -----------------------------
# تنظیمات مسیرها
# -----------------------------
# پوشهٔ RF که قبلاً در کد RF تعریف شده بود (همان ROOT/PRED_DIR)
RF_ROOT     = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/HRF/Publications"
RF_PRED_DIR = os.path.join(RF_ROOT, "_PREDICTIONS")

# مسیر خروجی پیش‌بینی HAN (خروجی تابع run_fit_and_predict_all در کد HAN)
HAN_PRED_CSV = "Wholedata-HAN-predictions-Zika-final.csv"

# فایل برچسب‌خوردهٔ ژن‌ها (ورودی کد HAN)
FILE_GENE_LABELED = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/Zika/ZikaInputdataForDeepL500HDF1000Non39Features_WithClass.csv"

# گزینهٔ فیلتر بر اساس ویروس در نام فایل RF؛ اگر None باشد همه را در نظر می‌گیرد
VIRUS_KEYWORD_IN_RF_FILENAMES = "Zika"  # یا None

# مسیرهای خروجی
OUT_DIR         = "_ENSEMBLE_OUT"
OSUM_CV_METRICS = os.path.join(OUT_DIR, "ensemble_cv_metrics.csv")
OJSON_WEIGHTS   = os.path.join(OUT_DIR, "ensemble_selected_weights.json")
OPRED_ALL       = os.path.join(OUT_DIR, "ensemble_predictions_all_genes.csv")
OPRED_LABELED   = os.path.join(OUT_DIR, "ensemble_predictions_labeled_genes.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# کمکی‌ها
# -----------------------------

def _read_rf_pred_tsv_gz(path: str) -> pd.DataFrame:
    with gzip.open(path, 'rt') as f:
        df = pd.read_csv(f, sep='\t')
    # انتظار ستون‌ها: Ensembl_ID, Prob_HDF, Rank (و ...)
    needed = ['Ensembl_ID', 'Prob_HDF']
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise RuntimeError(f"ستون‌های {miss} در {path} پیدا نشد.")
    return df[['Ensembl_ID','Prob_HDF']].copy()


def load_rf_predictions(pred_dir: str = RF_PRED_DIR,
                        virus_keyword: Optional[str] = VIRUS_KEYWORD_IN_RF_FILENAMES) -> pd.DataFrame:
    pats = glob.glob(os.path.join(pred_dir, "*.tsv.gz"))
    if virus_keyword:
        files = [p for p in pats if re.search(virus_keyword, os.path.basename(p), flags=re.IGNORECASE)]
        if not files:
            files = pats  # fallback
    else:
        files = pats
    if not files:
        raise RuntimeError(f"هیچ فایل پیش‌بینی RF در {pred_dir} یافت نشد.")

    dfs = []
    for p in files:
        try:
            df = _read_rf_pred_tsv_gz(p)
            tag = os.path.splitext(os.path.basename(p))[0]
            df.rename(columns={'Prob_HDF': f'Prob_RF_{tag}'}, inplace=True)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ پرش فایل RF ({os.path.basename(p)}): {e}")
            continue
    if not dfs:
        raise RuntimeError("هیچ فایل RF معتبر خوانده نشد.")

    # merge تدریجی روی Ensembl_ID
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on='Ensembl_ID', how='outer')

    # میانگین روی همهٔ ستون‌های Prob_RF_*
    prob_cols = [c for c in base.columns if c.startswith('Prob_RF_')]
    base['Prob_RF'] = base[prob_cols].mean(axis=1)
    return base[['Ensembl_ID','Prob_RF']].copy()


def load_han_predictions(csv_path: str = HAN_PRED_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # انتظار: ستون "Gene_Name" و "Predicted_Probability"
    gcol = None
    for cand in ['Gene_Name','Genes','Gene','Ensembl_ID']:
        if cand in df.columns:
            gcol = cand
            break
    if gcol is None:
        raise RuntimeError(f"ستون نام ژن در {csv_path} یافت نشد.")

    pcol = None
    for cand in ['Predicted_Probability','Prob','Probability','Prob_HDF','Score']:
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise RuntimeError(f"ستون احتمال در {csv_path} یافت نشد.")

    out = df[[gcol, pcol]].copy()
    out.columns = ['Gene_Key', 'Prob_HAN']

    # اگر به نظر Ensembl ID باشد، همان را Ensembl_ID می‌گذاریم
    if out['Gene_Key'].astype(str).str.startswith(('ENSG','ENSMUSG','ENS')).mean() > 0.8:
        out.rename(columns={'Gene_Key':'Ensembl_ID'}, inplace=True)
        return out[['Ensembl_ID','Prob_HAN']]

    # در غیر اینصورت سعی در نگاشت Symbol→Ensembl (اختیاری)
    if mygene is None:
        warnings.warn("mygene موجود نیست؛ نگاشت symbol→ensembl انجام نشد. ستون Gene_Key نگه داشته می‌شود.")
        out.rename(columns={'Gene_Key':'Ensembl_ID'}, inplace=True)  # امید به این‌که همان باشد
        return out[['Ensembl_ID','Prob_HAN']]

    mg = mygene.MyGeneInfo()
    syms = out['Gene_Key'].astype(str).tolist()
    res = mg.querymany(syms, scopes='symbol,alias', fields='ensembl.gene',
                       species='human', as_dataframe=False, returnall=False)
    m: Dict[str, str] = {}
    for r in res:
        q = r.get('query'); ens = r.get('ensembl')
        if not q:
            continue
        ids = []
        if isinstance(ens, list):
            ids = [str(x.get('gene')) for x in ens if isinstance(x, dict) and x.get('gene')]
        elif isinstance(ens, dict):
            if ens.get('gene'):
                ids = [str(ens.get('gene'))]
        ids = [i for i in ids if i]
        if ids:
            m[q] = sorted(set(ids))[0]
    out['Ensembl_ID'] = out['Gene_Key'].map(m)
    out = out.dropna(subset=['Ensembl_ID']).copy()
    return out[['Ensembl_ID','Prob_HAN']]


def load_labeled_genes(file_path: str = FILE_GENE_LABELED) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # انتظار ستون‌ها: Genes, Class
    need = ['Genes','Class']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"ستون‌های {miss} در فایل برچسب‌خورده یافت نشد.")
    out = df[['Genes','Class']].copy()
    out.rename(columns={'Genes':'Ensembl_ID'}, inplace=True)
    # نرمال‌سازی برچسب‌ها: HDF=1, non-HDF=0
    y = out['Class'].astype(str).str.replace('-', '', regex=False).str.lower()
    out['y'] = np.where(y.eq('hdf'), 1, 0)
    return out[['Ensembl_ID','y']]

# -----------------------------
# Stacking & Weighted Average
# -----------------------------

def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str,float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'AUPR': average_precision_score(y_true, y_prob),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred),
    }

def cv_meta_logreg(X: np.ndarray, y: np.ndarray,
                   Cs: List[float] = [0.01,0.1,1,3,10,30],
                   n_splits: int = 5, seed: int = 42) -> Tuple[LogisticRegression, Dict[str,float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_auc, best_C = -1.0, None
    for C in Cs:
        aucs = []
        for tr, te in skf.split(X, y):
            lr = LogisticRegression(C=C, solver='liblinear')
            lr.fit(X[tr], y[tr])
            p = lr.predict_proba(X[te])[:,1]
            aucs.append(roc_auc_score(y[te], p))
        m = float(np.mean(aucs))
        if m > best_auc:
            best_auc, best_C = m, C
    # fit final on all labeled
    lr_final = LogisticRegression(C=best_C, solver='liblinear')
    lr_final.fit(X, y)
    return lr_final, {'method':'stacking_logreg','best_C':best_C,'cv_auc':best_auc}

def cv_best_weight_average(p1: np.ndarray, p2: np.ndarray, y: np.ndarray,
                           grid: int = 101, n_splits: int = 5, seed: int = 42) -> Tuple[float, Dict[str,float]]:
    """
    وزن w برای HAN: p = w*p1 + (1-w)*p2
    ✅ اصلاح: امتیازدهی روی foldهای TEST (نه train).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ws = np.linspace(0.0, 1.0, grid)
    best_auc, best_w = -1.0, 0.5
    for w in ws:
        aucs = []
        for tr, te in skf.split(p1.reshape(-1, 1), y):
            pp_te = w*p1[te] + (1.0-w)*p2[te]
            aucs.append(roc_auc_score(y[te], pp_te))
        m = float(np.mean(aucs))
        if m > best_auc:
            best_auc, best_w = m, float(w)
    return best_w, {'method':'weighted_average','best_w':best_w,'cv_auc':best_auc}

# -----------------------------
# Main
# -----------------------------

def main():
    # 1) بارگذاری خروجی‌ها
    han = load_han_predictions(HAN_PRED_CSV)
    rf  = load_rf_predictions(RF_PRED_DIR, VIRUS_KEYWORD_IN_RF_FILENAMES)

    # 2) join برای تمام ژن‌ها (پوشش مشترک و همچنین نگه‌داشتن هر کدام که موجود است)
    all_pred = pd.merge(han, rf, on='Ensembl_ID', how='outer')

    # 3) دادهٔ برچسب‌خورده برای یادگیری ensemble
    lab = load_labeled_genes(FILE_GENE_LABELED)
    L = all_pred.merge(lab, on='Ensembl_ID', how='inner').dropna(subset=['Prob_HAN','Prob_RF'])
    if L.shape[0] < 20:
        raise RuntimeError("تلاقی نمونه‌های برچسب‌خورده با هر دو مدل کم است. مسیرها/دیتا را بررسی کنید.")

    X = L[['Prob_HAN','Prob_RF']].values.astype(float)
    y = L['y'].values.astype(int)

    # 4) روش‌های Ensemble و انتخاب بهترین بر اساس CV-AUC
    lr_model, info_lr = cv_meta_logreg(X, y)
    w_best, info_w = cv_best_weight_average(X[:,0], X[:,1], y)

    # 5) انتخاب بهترین
    if info_lr['cv_auc'] >= info_w['cv_auc']:
        chosen = {'chosen':'stacking_logreg', **info_lr}
        # احتمال نهایی برای labeled
        L['Prob_Ensemble'] = lr_model.predict_proba(X)[:,1]
    else:
        chosen = {'chosen':'weighted_average', **info_w}
        L['Prob_Ensemble'] = w_best*X[:,0] + (1.0-w_best)*X[:,1]

    # 6) گزارش و ذخیرهٔ نتایج روی labeled
    met_lab = evaluate_metrics(L['y'].values, L['Prob_Ensemble'].values, thr=0.5)
    cv_rows = [
        {'Variant':'HAN_only','AUC':roc_auc_score(y, X[:,0])},
        {'Variant':'RF_only','AUC':roc_auc_score(y, X[:,1])},
        {'Variant':'Stacking(LR)','AUC':info_lr['cv_auc']},
        {'Variant':'WeightedAvg','AUC':info_w['cv_auc']},
        {'Variant':'Chosen@0.5(AUC)','AUC':met_lab['AUC']},
    ]
    pd.DataFrame(cv_rows).to_csv(OSUM_CV_METRICS, index=False)

    with open(OJSON_WEIGHTS, 'w') as f:
        json.dump(chosen, f, indent=2)

    L[['Ensembl_ID','Prob_HAN','Prob_RF','Prob_Ensemble','y']].to_csv(OPRED_LABELED, index=False)

    print("=== انتخاب نهایی Ensemble ===")
    print(json.dumps(chosen, indent=2))
    print("\n=== کارایی روی دادهٔ برچسب‌خورده (threshold=0.5) ===")
    for k,v in met_lab.items():
        print(f"{k:>10s}: {v:.4f}")

    # 7) اعمال روی تمام ژن‌ها
    A = all_pred.copy()
    # اگر یکی از احتمالات NaN بود، با دیگری جایگزین شود (fallback)
    if 'Prob_HAN' in A and 'Prob_RF' in A:
        if A['Prob_HAN'].isna().any():
            m = A['Prob_HAN'].isna() & A['Prob_RF'].notna()
            A.loc[m, 'Prob_HAN'] = A.loc[m, 'Prob_RF']
        if A['Prob_RF'].isna().any():
            m = A['Prob_RF'].isna() & A['Prob_HAN'].notna()
            A.loc[m, 'Prob_RF'] = A.loc[m, 'Prob_HAN']

    # هر دو ستون باید وجود داشته باشند
    A = A.dropna(subset=['Prob_HAN','Prob_RF']).copy()

    X_all = A[['Prob_HAN','Prob_RF']].values.astype(float)
    if chosen['chosen']=='stacking_logreg':
        A['Prob_Ensemble'] = lr_model.predict_proba(X_all)[:,1]
    else:
        A['Prob_Ensemble'] = chosen['best_w']*X_all[:,0] + (1.0-chosen['best_w'])*X_all[:,1]

    # مرتب‌سازی و ذخیره
    A = A.sort_values('Prob_Ensemble', ascending=False).reset_index(drop=True)
    A['Rank'] = np.arange(1, len(A)+1)
    A.to_csv(OPRED_ALL, index=False)

    print(f"\n✅ ذخیره شد: {OSUM_CV_METRICS}\n✅ ذخیره شد: {OJSON_WEIGHTS}\n✅ ذخیره شد: {OPRED_LABELED}\n✅ ذخیره شد: {OPRED_ALL}")

if __name__ == "__main__":
    main()
