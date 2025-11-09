# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: gpu
#     language: python
#     name: gpu
# ---

# %%
import warnings
warnings.filterwarnings('ignore')  # همهٔ هشدارها پنهان می‌شوند

# %%
# ================================================
# End-to-end HDF vs non-HDF (HRF) pipeline in Python
# Jupyter-ready, single-notebook version
# ================================================
# # !pip install pandas numpy scikit-learn imbalanced-learn mygene openpyxl xlsxwriter upsetplot matplotlib joblib tqdm

import os
import re
import json
import gzip
import math
import time
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, matthews_corrcoef, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from upsetplot import UpSet, from_contents
import matplotlib.pyplot as plt

from joblib import dump, load
from tqdm import tqdm

# --------------------------------
# پیکربندی مسیرها / تنظیمات کاربر
# --------------------------------
ROOT_RAW     = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/HRF/Emtahan_dobare_HDF/"
ROOT_PUB     = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/HRF/Publications"
FEATURES_CSV = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/Turtle/Desktop/Masterarbeit/Features/combinedFinal.csv"
VIRUSES      = ["Denv","IAV","Sars_Cov_2","Zika"]

TOP_N        = 1000
TARGET_N     = 1000
OUT_SUBDIR   = "MAIC_top_with_ensembl"
BAL_SUBDIR   = "balanced_from_bottom"

POS_CLASS    = "HDF"   # کلاس مثبت
NEG_CLASS    = "HRF"   # non-HDF (Low-MAIC)

RANDOM_SEED  = 113
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# برای cache کردن نگاشت‌ها (اختیاری اما توصیه می‌شود)
MAPPING_CACHE = os.path.join(ROOT_RAW, "_CACHE_symbol2ensembl.csv")
os.makedirs(os.path.dirname(MAPPING_CACHE), exist_ok=True)

# ============================================================
# 0) ابزارهای کمکی
# ============================================================

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_dirs(p: str) -> List[str]:
    return [d for d in sorted(Path(p).iterdir()) if d.is_dir()]

def make_safe_sheet_name(name: str) -> str:
    n = re.sub(r"[\[\]\*\?\/\\:]", "_", name).strip()
    return (n[:31] if len(n)>31 else (n if len(n)>0 else "Sheet"))

def set_global_seeds(seed: int = 113):
    np.random.seed(seed); random.seed(seed)

def write_compressed_tsv(df: pd.DataFrame, path: str):
    if path.endswith(".gz"):
        df.to_csv(path, sep="\t", index=False, compression="gzip")
    else:
        df.to_csv(path, sep="\t", index=False)

# ============================================================
# 1) نگاشت SYMBOL → ENSEMBL (اولین ENSG)
#    از mygene استفاده می‌کنیم + کش محلی
# ============================================================
def load_mapping_cache(cache_path=MAPPING_CACHE) -> Dict[str, str]:
    if os.path.isfile(cache_path):
        m = pd.read_csv(cache_path)
        m = m.dropna(subset=["symbol","ensembl"]).drop_duplicates("symbol")
        return dict(zip(m["symbol"].astype(str), m["ensembl"].astype(str)))
    return {}

def save_mapping_cache(map_dict: Dict[str,str], cache_path=MAPPING_CACHE):
    if not map_dict:
        return
    rows = [{"symbol":k, "ensembl":v} for k,v in map_dict.items() if isinstance(k,str) and isinstance(v,str)]
    df = pd.DataFrame(rows)
    if os.path.isfile(cache_path):
        old = pd.read_csv(cache_path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.dropna(subset=["symbol","ensembl"]).drop_duplicates("symbol", keep="last")
    df.to_csv(cache_path, index=False)

def query_mygene_symbols(symbols: List[str]) -> Dict[str, Optional[str]]:
    """
    تلاش برای نگاشت با mygene؛ اگر اینترنت قطع باشد یا mygene نصب نباشد، None می‌دهیم.
    """
    try:
        import mygene
    except Exception:
        return {s: None for s in symbols}
    mg = mygene.MyGeneInfo()
    out = {s: None for s in symbols}
    # batch query
    try:
        res = mg.querymany(symbols, scopes="symbol,alias,ensembl.gene", fields="ensembl.gene", species="human", as_dataframe=False, returnall=False, verbose=False)
    except Exception:
        return out
    for r in res:
        q = r.get('query')
        if 'notfound' in r and r['notfound']:
            continue
        ens = None
        # r['ensembl'] می‌تواند dict یا list باشد
        if 'ensembl' in r:
            e = r['ensembl']
            if isinstance(e, dict) and 'gene' in e:
                ens = e['gene']
            elif isinstance(e, list) and len(e)>0:
                # اولین gene که با ENSG شروع می‌شود
                for item in e:
                    g = item.get('gene') if isinstance(item, dict) else None
                    if isinstance(g, str) and g.startswith("ENSG"):
                        ens = g; break
                if ens is None:
                    # fallback به اولین
                    g = e[0].get('gene') if isinstance(e[0], dict) else None
                    ens = g
        if isinstance(ens, str) and ens.startswith("ENSG"):
            out[q] = ens
    return out

def map_symbols_to_ensembl_first(symbols: List[str]) -> List[Optional[str]]:
    """
    اول از کش، بعد از mygene استفاده می‌کنیم. فقط ENSG را قبول می‌کنیم.
    """
    symbols = [str(s).strip() if s is not None else "" for s in symbols]
    cache = load_mapping_cache()
    result: Dict[str, Optional[str]] = {}
    missing = []
    for s in symbols:
        if s in cache:
            result[s] = cache[s]
        else:
            missing.append(s)
    if missing:
        qres = query_mygene_symbols(missing)
        # به‌روز کردن کش با موارد پیدا شده
        new_pairs = {k:v for k,v in qres.items() if isinstance(v,str) and v.startswith("ENSG")}
        if new_pairs:
            cache.update(new_pairs)
            save_mapping_cache(cache)
        result.update(qres)
    # فقط ENSG، در غیر این صورت None
    out = []
    for s in symbols:
        v = result.get(s)
        out.append(v if isinstance(v,str) and v.startswith("ENSG") else None)
    return out

# ============================================================
# 2) ساخت Top/Bottom با Ensembl از فایل‌های خام
# ============================================================
def process_one_hdf_file(file_path: str, out_dir: str, top_n: int = 1000):
    print(f">>> خواندن فایل: {file_path}")
    dt = pd.read_csv(file_path, sep="\t", header=0, dtype=str)
    if dt.shape[0] == 0:
        print(f"⚠️ فایل خالی: {file_path}"); return
    # ستون‌ها
    nms = [c.lower() for c in dt.columns]
    try:
        gene_col = dt.columns[nms.index("gene")]
    except ValueError:
        raise RuntimeError(f"ستون 'gene' پیدا نشد: {file_path}")
    score_col = dt.columns[nms.index("maic_score")] if "maic_score" in nms else None

    if score_col is not None:
        dt[score_col] = pd.to_numeric(dt[score_col], errors="coerce")
        dt = dt.sort_values(by=score_col, ascending=False, kind="mergesort")
    dt["rank_global"] = np.arange(1, len(dt)+1)

    def map_and_write(sub_df: pd.DataFrame, label_suffix: str):
        syms = sub_df[gene_col].astype(str).tolist()
        ens = map_symbols_to_ensembl_first(syms)
        out = sub_df.copy()
        out.insert(1, "ensembl_id", ens)
        removed_na = int(out["ensembl_id"].isna().sum())
        out = out.dropna(subset=["ensembl_id"]).copy()

        ensure_dir(out_dir)
        base_noext = os.path.splitext(os.path.basename(file_path))[0]
        out_file = os.path.join(out_dir, f"{base_noext}_{label_suffix}{top_n}_with_ensembl.tsv")
        out.to_csv(out_file, sep="\t", index=False)
        print(f"    ↳ {label_suffix.upper()}: {len(out)} ردیف ذخیره شد (حذف {removed_na} بدون Ensembl): {out_file}")

    # Top
    top_df = dt.head(min(top_n, len(dt))).copy()
    top_df["rank_top"] = np.arange(1, len(top_df)+1)
    map_and_write(top_df, "top")
    # Bottom
    bot_df = dt.tail(min(top_n, len(dt))).copy()
    bot_df["rank_bottom"] = np.arange(1, len(bot_df)+1)
    map_and_write(bot_df, "bottom")

def step_build_top_bottom():
    for virus in VIRUSES:
        virus_dir = os.path.join(ROOT_RAW, virus)
        if not os.path.isdir(virus_dir):
            print(f"⏭️ پوشه پیدا نشد: {virus_dir}"); continue
        print(f"====== ویروس: {virus} ======")
        hdf_files = sorted([str(p) for p in Path(virus_dir).glob("filtered_H*.txt")])
        if not hdf_files:
            print(f"⏭️ filtered_H*.txt یافت نشد در: {virus_dir}"); continue
        out_dir = os.path.join(virus_dir, OUT_SUBDIR)
        for fp in hdf_files:
            process_one_hdf_file(fp, out_dir=out_dir, top_n=TOP_N)
    print("✅ ساخت فایل‌های Top/Bottom تمام شد.")

# ============================================================
# 3) برچسب‌گذاری HDF/HRF روی فیچرها (Top=HDF, Bottom=HRF)
# ============================================================
def pick_hdf_top_files(virus: str, root: str = ROOT_RAW, subdir: str = OUT_SUBDIR, prefer_all_first=True) -> List[str]:
    dir_top = os.path.join(root, virus, subdir)
    if not os.path.isdir(dir_top):
        raise RuntimeError(f"پوشه وجود ندارد: {dir_top}")
    files = sorted([str(p) for p in Path(dir_top).glob("filtered_HDF_*_top*_with_ensembl.tsv")])
    if not files:
        raise RuntimeError(f"هیچ فایل *_top*_with_ensembl.tsv در {dir_top} پیدا نشد.")
    if prefer_all_first:
        allc = [f for f in files if re.search("All_Categories", os.path.basename(f), flags=re.I)]
        others = [f for f in files if f not in allc]
        files = allc + others
    return list(dict.fromkeys(files))  # unique, keep order

def read_features(features_path: str) -> pd.DataFrame:
    ext = Path(features_path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(features_path)
    elif ext in [".tsv",".txt"]:
        df = pd.read_csv(features_path, sep="\t")
    elif ext == ".rds" or ext in [".rdata",".rda"]:
        raise RuntimeError("فرمت‌های RDS/RData در این نسخه پشتیبانی نشده—لطفاً CSV/TSV بدهید.")
    else:
        df = pd.read_csv(features_path)  # سعی می‌کنیم CSV باشد
    # نرمال‌سازی نام ستون Ensembl_ID
    cand = [c for c in df.columns if c in ["Ensembl_ID","ENSEMBL_ID","EnsemblId","ensembl_id","ENSEMBL","Ensembl","ensembl"]]
    if cand:
        df = df.rename(columns={cand[0]: "Ensembl_ID"})
    if "Ensembl_ID" not in df.columns:
        raise RuntimeError("ستون 'Ensembl_ID' در فیچرها نیست.")
    # یکتا بر اساس ID
    df = df.drop_duplicates(subset=["Ensembl_ID"]).reset_index(drop=True)
    return df

def label_top_bottom_exact_from_raw(hdf_top_file: str, features_path: str, n_each: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dir_top = os.path.dirname(hdf_top_file)
    dir_raw = os.path.dirname(dir_top)
    base = re.sub(r"_top\d+_with_ensembl\.tsv$", "", os.path.basename(hdf_top_file), flags=re.I)
    raw_guess = os.path.join(dir_raw, f"{base}.txt")
    if not os.path.isfile(raw_guess):
        pat = re.compile("^" + re.escape(base) + r"\.txt$", flags=re.I)
        cands = [str(p) for p in Path(dir_raw).glob("*.txt") if pat.search(os.path.basename(p))]
        if not cands:
            raise RuntimeError(f"فایل خام پیدا نشد: {raw_guess}")
        raw_guess = cands[0]

    # خواندن و مرتب‌سازی نزولی بر اساس MAIC
    dt = pd.read_csv(raw_guess, sep="\t", header=0, dtype=str)
    nms = [c.lower() for c in dt.columns]
    if "gene" not in nms:
        raise RuntimeError(f"ستون 'gene' در فایل خام نیست: {raw_guess}")
    gene_col = dt.columns[nms.index("gene")]
    score_col = dt.columns[nms.index("maic_score")] if "maic_score" in nms else None

    if score_col is not None:
        dt[score_col] = pd.to_numeric(dt[score_col], errors="coerce")
        dt = dt.sort_values(by=score_col, ascending=False, kind="mergesort")

    syms = dt[gene_col].astype(str).tolist()
    ens = map_symbols_to_ensembl_first(syms)
    dt["ensembl_id"] = ens
    dt = dt.dropna(subset=["ensembl_id"]).copy()
    ordered_ens = pd.unique(dt["ensembl_id"]).tolist()

    feat = read_features(features_path)
    present = set(feat["Ensembl_ID"].astype(str))

    def pick_n_in_order(vec: List[str], present_ids: set, exclude: set, n_target: int, from_tail: bool=False) -> List[str]:
        v = list(reversed(vec)) if from_tail else list(vec)
        chosen = []
        for g in v:
            if g in present_ids and g not in exclude and g not in chosen:
                chosen.append(g)
                if len(chosen) == n_target:
                    break
        return list(reversed(chosen)) if from_tail else chosen

    top_ids    = pick_n_in_order(ordered_ens, present, set(), n_each, from_tail=False)      # HDF
    bottom_ids = pick_n_in_order(ordered_ens, present, set(top_ids), n_each, from_tail=True) # HRF

    if len(top_ids)    < n_each: print(f"⚠️ فقط {len(top_ids)} از {n_each} آیتم Top در فیچرها پیدا شد.")
    if len(bottom_ids) < n_each: print(f"⚠️ فقط {len(bottom_ids)} از {n_each} آیتم Bottom در فیچرها پیدا شد.")

    feat["Class"] = np.where(feat["Ensembl_ID"].isin(top_ids), POS_CLASS,
                      np.where(feat["Ensembl_ID"].isin(bottom_ids), NEG_CLASS, np.nan))
    feat_labeled = feat.dropna(subset=["Class"]).copy()

    counts = feat_labeled.groupby("Class", dropna=False).size().reset_index(name="N").sort_values("N", ascending=False)
    print(counts)
    return feat_labeled, counts

def step_label_and_save():
    for virus in VIRUSES:
        print(f"====== Labeling → {virus} ======")
        try:
            top_files = pick_hdf_top_files(virus, root=ROOT_RAW, subdir=OUT_SUBDIR)
        except Exception as e:
            print("⏭️", e); continue
        for hdf_file in top_files:
            print(f"→ پردازش فایل: {os.path.basename(hdf_file)}")
            feat_labeled, counts = label_top_bottom_exact_from_raw(hdf_top_file=hdf_file, features_path=FEATURES_CSV, n_each=TARGET_N)
            out_dir = os.path.join(os.path.dirname(hdf_file), BAL_SUBDIR)
            ensure_dir(out_dir)
            base_stub = re.sub(r"_top\d+_with_ensembl\.tsv$", "", os.path.basename(hdf_file), flags=re.I)
            out_csv = os.path.join(out_dir, f"{base_stub}_bottom{TARGET_N}_HRF__top{TARGET_N}_HDF.csv")
            feat_labeled.to_csv(out_csv, index=False)
            print(f"✅ ذخیره شد: {out_csv}")

# ============================================================
# 4) ابزارهای ML: NZV، حذف همبستگی، آموزش و ذخیره خروجی‌ها
# ============================================================
def near_zero_var_keep_columns(X: pd.DataFrame) -> List[str]:
    """
    تقریب nearZeroVar caret:
      - حذف ستون‌های با تنوع صفر
      - حذف ستون‌هایی که (freqRatio > 19) و (percentUnique <= 10)
    """
    keep = []
    n = X.shape[0]
    for c in X.columns:
        col = X[c].values
        # حذف تمام-NA
        if np.all(pd.isna(col)):
            continue
        # zero variance
        vals, counts = np.unique(col[~pd.isna(col)], return_counts=True)
        if len(vals) <= 1:
            continue
        counts_sorted = np.sort(counts)[::-1]
        freq_ratio = counts_sorted[0] / (counts_sorted[1] if len(counts_sorted)>1 else 1)
        percent_unique = 100.0 * len(vals) / n
        if not (freq_ratio > 19 and percent_unique <= 10):
            keep.append(c)
    return keep

def find_correlation_to_drop(df: pd.DataFrame, threshold: float = 0.70) -> List[str]:
    """
    معادل تقریبی caret::findCorrelation:
    به صورت تکراری ستون‌هایی که بیشترین میانگین |corr| را دارند و از آستانه بالاترند حذف می‌کنیم.
    """
    if df.shape[1] <= 1:
        return []
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    while True:
        # بیشترین همبستگی فعلی
        max_corr = (upper.max().max())
        if np.isnan(max_corr) or max_corr < threshold:
            break
        # ستونی که میانگین همبستگی‌اش بالاتر است حذف کن
        mean_corr = upper.mean()
        col_to_drop = mean_corr.idxmax()
        to_drop.append(col_to_drop)
        # حذف ستون/سطر از upper
        upper = upper.drop(index=col_to_drop, columns=col_to_drop)
    return to_drop

def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=113) -> Tuple[float,float,float]:
    rng = check_random_state(seed)
    aucs = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        bs = rng.choice(idx, size=len(idx), replace=True)
        try:
            a = roc_auc_score(y_true[bs], y_prob[bs])
            if not np.isnan(a):
                aucs.append(a)
        except Exception:
            pass
    if not aucs:
        return (np.nan, np.nan, np.nan)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    full_auc = roc_auc_score(y_true, y_prob)
    return (lo, full_auc, hi)

def train_and_export_single_csv(df_bal: pd.DataFrame, balanced_csv_path: str, pos_class: str = POS_CLASS):
    # شناسه
    id_candidates = ["Ensembl_ID","ENSEMBL_ID","EnsemblId","ensembl_id"]
    id_col = next((c for c in id_candidates if c in df_bal.columns), None)
    if id_col is None:
        raise RuntimeError("ستون شناسه‌ی Ensembl_ID در دیتافریم یافت نشد.")
    assert "Class" in df_bal.columns

    # کلاس‌ها
    y = df_bal["Class"].astype(str)
    y = pd.Categorical(y, categories=[pos_class, (set(["HDF","HRF"]) - {pos_class}).pop()], ordered=True)
    y = y.astype(str)

    # ویژگی‌ها: بقیه ستون‌ها به جز id و Class
    pred_cols = [c for c in df_bal.columns if c not in [id_col, "Class"]]
    X_raw = df_bal[pred_cols].copy()

    # تبدیل امن به عددی
    for c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")

    # train/test split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_raw, y, df_bal[id_col].astype(str), test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # NZV
    keep_nzv = near_zero_var_keep_columns(X_train)
    if not keep_nzv:
        raise RuntimeError("[NZV] هیچ ویژگی‌ای باقی نماند.")
    X_train = X_train[keep_nzv].copy()
    X_test  = X_test[keep_nzv].copy()

    # Preprocess: impute + scale (fit روی Train)
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler(with_mean=True, with_std=True)

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test),   columns=X_test.columns,   index=X_test.index)
    X_train_scl = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_imp.columns, index=X_train_imp.index)
    X_test_scl  = pd.DataFrame(scaler.transform(X_test_imp),      columns=X_test_imp.columns,  index=X_test_imp.index)

    # RFE با RF + CV تکراری
    base_rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
    )
    cv_rfe = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)
    rfe = RFECV(
        estimator=base_rf,
        step=0.2,
        min_features_to_select=1,
        cv=cv_rfe,
        scoring="roc_auc",
        n_jobs=-1
    )
    rfe.fit(X_train_scl, y_train)
    sel_mask = rfe.support_
    sel_vars = list(X_train_scl.columns[sel_mask])
    if not sel_vars:
        raise RuntimeError("[RFE] هیچ ویژگی‌ای انتخاب نشد.")

    # حذف همبستگی روی فیچرهای انتخاب‌شده
    corr_drop = find_correlation_to_drop(X_train_scl[sel_vars], threshold=0.70)
    sel_final = [c for c in sel_vars if c not in corr_drop]
    if not sel_final:
        sel_final = sel_vars

    Xtr_sel = X_train_scl[sel_final].copy()
    Xte_sel = X_test_scl[sel_final].copy()

    # class imbalance
    cls_counts = pd.Series(y_train).value_counts()
    imb_ratio = cls_counts.max() / max(1, cls_counts.min()) if len(cls_counts)>=2 else 1.0
    use_smote = imb_ratio >= 1.5

    # RF tuning (mtry شبیه‌سازی با max_features)
    p = Xtr_sel.shape[1]
    mtries = sorted(set(int(max(1, round(math.sqrt(p)*k))) for k in [0.5,1,2,3]))
    param_grid = {"clf__max_features": mtries}

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_SEED)

    # در CV از SMOTE استفاده می‌کنیم، اما مدل نهایی را بدون SMOTE روی کل Train فیت می‌کنیم
    pipe_cv = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_SEED)) if use_smote else ("smote", "passthrough"),
        ("clf", RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1
        ))
    ])

    grid = GridSearchCV(
        estimator=pipe_cv,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        refit=True,
        verbose=0
    )
    grid.fit(Xtr_sel, y_train)
    best_mtry = grid.best_params_.get("clf__max_features", None)

    # مدل نهایی (بدون SMOTE، با best params)
    final_rf = RandomForestClassifier(
        n_estimators=500, max_features=best_mtry, random_state=RANDOM_SEED, n_jobs=-1
    )
    final_rf.fit(Xtr_sel, y_train)

    # پیش‌بینی روی Test
    pred_cls = final_rf.predict(Xte_sel)
    if POS_CLASS not in final_rf.classes_:
        raise RuntimeError("[PRED_PROB] کلاس مثبت در مدل وجود ندارد.")
    prob_pos = final_rf.predict_proba(Xte_sel)[:, list(final_rf.classes_).index(POS_CLASS)]

    # متریک‌ها
    acc = accuracy_score(y_test, pred_cls)
    bal_acc = balanced_accuracy_score(y_test, pred_cls)
    try:
        auc_val = roc_auc_score((y_test==POS_CLASS).astype(int), prob_pos)
    except Exception:
        auc_val = np.nan
    auc_lo, auc_mid, auc_hi = bootstrap_auc_ci((y_test==POS_CLASS).astype(int), prob_pos, n_boot=1000, seed=RANDOM_SEED)
    sens = recall_score(y_test, pred_cls, pos_label=POS_CLASS, zero_division=0)
    spec = recall_score(y_test, pred_cls, pos_label=NEG_CLASS, zero_division=0)
    prec = precision_score(y_test, pred_cls, pos_label=POS_CLASS, zero_division=0)
    reca = sens
    f1   = f1_score(y_test, pred_cls, pos_label=POS_CLASS, zero_division=0)
    mcc  = matthews_corrcoef(pd.Series(y_test).map({POS_CLASS:1, NEG_CLASS:0}), pd.Series(pred_cls).map({POS_CLASS:1, NEG_CLASS:0}))
    brier = brier_score_loss((y_test==POS_CLASS).astype(int), prob_pos)

    # مسیر خروجی‌ها
    base_name = os.path.basename(balanced_csv_path)
    base_stub = re.sub(r"(_balanced_top_vs_bottom|_bottom\d+_HRF__top\d+_HDF)\.csv$", "", base_name, flags=re.I)
    out_dir = os.path.join(os.path.dirname(balanced_csv_path), "ML_results", base_stub)
    ensure_dir(out_dir)

    # class counts
    train_counts = pd.Series(y_train).value_counts().reindex([POS_CLASS, NEG_CLASS]).fillna(0).astype(int)
    test_counts  = pd.Series(y_test ).value_counts().reindex([POS_CLASS, NEG_CLASS]).fillna(0).astype(int)
    cc = pd.DataFrame({
        "Split": ["Train","Train","Test","Test"],
        "Class": [POS_CLASS, NEG_CLASS, POS_CLASS, NEG_CLASS],
        "N":     [train_counts.get(POS_CLASS,0), train_counts.get(NEG_CLASS,0),
                  test_counts.get(POS_CLASS,0),  test_counts.get(NEG_CLASS,0)]
    })
    cc["Percent"] = [round(100*n/cc[cc["Split"]==sp]["N"].sum(), 2) for sp,n in zip(cc["Split"], cc["N"])]
    cc.to_csv(os.path.join(out_dir, "class_counts_train_test.csv"), index=False)

    # metrics
    metrics_df = pd.DataFrame({
        "Metric": ["N_test","Pos_in_test","Neg_in_test",
                   "Accuracy","Balanced_Accuracy","AUC","AUC_95CI_Lower","AUC_95CI_Upper",
                   "Sensitivity","Specificity","Precision","Recall","F1","MCC","Brier_Score"],
        "Value":  [Xte_sel.shape[0], int(test_counts.get(POS_CLASS,0)), int(test_counts.get(NEG_CLASS,0)),
                   acc, bal_acc, auc_mid, auc_lo, auc_hi,
                   sens, spec, prec, reca, f1, mcc, brier]
    })
    metrics_df.to_csv(os.path.join(out_dir, "metrics_test.csv"), index=False)

    # confusion matrix
    cm = confusion_matrix(y_test, pred_cls, labels=[POS_CLASS, NEG_CLASS])
    cm_df = pd.DataFrame({
        "Reference": [POS_CLASS, POS_CLASS, NEG_CLASS, NEG_CLASS],
        "Prediction":[POS_CLASS, NEG_CLASS, POS_CLASS, NEG_CLASS],
        "Count": cm.flatten()
    })
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix_test.csv"), index=False)

    # feature importance (Top 30)
    importances = final_rf.feature_importances_
    imp_df = pd.DataFrame({"Feature": sel_final, "Importance": importances}).sort_values("Importance", ascending=False)
    imp_df.head(30).to_csv(os.path.join(out_dir, "top_features.csv"), index=False)

    # test predictions
    preds_df = pd.DataFrame({
        "Ensembl_ID": ids_test.values,
        "True_Class": y_test.values,
        "Pred_Class": pred_cls,
        "Prob_Pos":   prob_pos
    })
    preds_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # model summary
    model_info = pd.DataFrame({
        "Item":  ["Model","Positive_Class","CV_Method","CV_Folds","CV_Repeats","Best_mtry",
                  "Selected_Features_RFE","Selected_Features_Final"],
        "Value": ["RandomForest", pos_class, "repeatedcv","10","3",
                  best_mtry, len(sel_vars), len(sel_final)]
    })
    model_info.to_csv(os.path.join(out_dir, "model_summary.csv"), index=False)

    # save lists
    pd.DataFrame({"Feature": sel_vars}).to_csv(os.path.join(out_dir, "selected_features_rfe.csv"), index=False)
    pd.DataFrame({"Feature": sel_final}).to_csv(os.path.join(out_dir, "selected_features_final.csv"), index=False)

    # bundle برای inference: اجزای preprocessing + selected vars + پارامترهای مدل
    bundle = {
        "imputer": imputer,           # fitted
        "scaler": scaler,             # fitted
        "sel_vars": sel_final,        # لیست ویژگی‌های نهایی
        "all_vars_train": list(X_train.columns),  # قبل از انتخاب نهایی
        "id_col": id_col,
        "pos_class": pos_class,
        "rf_params": {"n_estimators": 500, "max_features": best_mtry, "random_state": RANDOM_SEED, "n_jobs": -1},
        "created_at": time.asctime(),
        "version": {"python":"sklearn-pipeline", "seed": RANDOM_SEED}
    }
    dump((bundle, final_rf), os.path.join(out_dir, "model_bundle.joblib"))
    print(f"✅ ذخیره شد: {out_dir}")

def step_ml_over_balanced_csvs():
    for virus in VIRUSES:
        bal_dir = os.path.join(ROOT_RAW, virus, OUT_SUBDIR, BAL_SUBDIR)
        if not os.path.isdir(bal_dir):
            print(f"⏭️ مسیر نیست: {bal_dir}"); continue
        bal_files = sorted([str(p) for p in Path(bal_dir).glob("*_bottom*_HRF__top*_HDF.csv")] +
                           [str(p) for p in Path(bal_dir).glob("*_balanced_top_vs_bottom.csv")])
        if not bal_files:
            print(f"⏭️ فایل بالانس‌شده‌ای در {bal_dir} پیدا نشد."); continue
        print(f"====== ML → {virus} | files: {len(bal_files)} ======")
        for i, bal_file in enumerate(bal_files, 1):
            print(f"({i}/{len(bal_files)}) پردازش: {os.path.basename(bal_file)}")
            try:
                df_bal = pd.read_csv(bal_file)
            except Exception as e:
                print("❌ خطا در خواندن:", e); continue
            if "Class" not in df_bal.columns:
                print(f"⚠️ ستون Class یافت نشد: {os.path.basename(bal_file)}"); continue
            if df_bal["Class"].nunique() < 2:
                print(f"⚠️ کلاس کافی نیست. رد شد: {os.path.basename(bal_file)}"); continue
            try:
                train_and_export_single_csv(df_bal, balanced_csv_path=bal_file, pos_class=POS_CLASS)
            except Exception as e:
                print("❌ خطای مدل‌سازی/خروجی:", e)

    print("🎉 همهٔ فایل‌های بالانس‌شده پردازش شدند.")

# ============================================================
# 5) ادغام همهٔ metrics_test.csv به long & wide
# ============================================================
def find_all_ml_results_roots():
    roots = []
    # ابتدا ROOT_PUB
    for virus in VIRUSES:
        base = os.path.join(ROOT_PUB, virus, OUT_SUBDIR, BAL_SUBDIR, "ML_results")
        if os.path.isdir(base):
            roots.append(base)
    # سپس ROOT_RAW (اگر کسی فایل‌ها را جابه‌جا نکرده باشد)
    for virus in VIRUSES:
        base = os.path.join(ROOT_RAW, virus, OUT_SUBDIR, BAL_SUBDIR, "ML_results")
        if os.path.isdir(base):
            roots.append(base)
    # unique
    return list(dict.fromkeys(roots))

def pretty_name(base_stub: str) -> str:
    s = re.sub(r"^filtered_HDF_", "", base_stub, flags=re.I)
    s = re.sub(r"_bottom\d+_HRF__top\d+_HDF$", "", s, flags=re.I)
    s = re.sub(r"_balanced_top_vs_bottom$", "", s, flags=re.I)
    return s

def step_merge_metrics():
    SUMMARY_DIR = os.path.join(ROOT_PUB, "_ML_SUMMARY")
    ensure_dir(SUMMARY_DIR)

    rows_long = []
    rows_wide = []

    roots = find_all_ml_results_roots()
    if not roots:
        print("⚠️ هیچ پوشهٔ ML_results پیدا نشد."); return

    for base_dir in roots:
        virus_guess = Path(base_dir).parts[-4] if len(Path(base_dir).parts)>=4 else "NA"
        exp_dirs = [str(p) for p in Path(base_dir).iterdir() if p.is_dir()]
        if not exp_dirs:
            print(f"⏭️ آزمایشی در {base_dir} یافت نشد."); continue
        print(f"====== Summary → {virus_guess} | experiments: {len(exp_dirs)} ======")
        for exp_dir in exp_dirs:
            base_stub = os.path.basename(exp_dir)
            exp_name  = pretty_name(base_stub)
            f_metrics = os.path.join(exp_dir, "metrics_test.csv")
            if not os.path.isfile(f_metrics):
                print("⚠️ metrics_test.csv موجود نیست/ناقص:", exp_dir); continue
            met = pd.read_csv(f_metrics)
            if not set(["Metric","Value"]).issubset(met.columns):
                print("⚠️ metrics_test.csv ناقص:", exp_dir); continue
            long_dt = met.copy()
            # تبدیل Value به عدد
            with np.errstate(all='ignore'):
                long_dt["Value"] = pd.to_numeric(long_dt["Value"], errors="coerce")
            long_dt["Virus"] = virus_guess
            long_dt["Experiment"] = exp_name
            long_dt["Exp_Path"] = exp_dir
            long_dt = long_dt[["Virus","Experiment","Metric","Value","Exp_Path"]]
            rows_long.append(long_dt)

            pivot_vals = dict(zip(long_dt["Metric"], long_dt["Value"]))
            wide_dt = pd.DataFrame([pivot_vals])
            wide_dt["Virus"] = virus_guess
            wide_dt["Experiment"] = exp_name
            wide_dt["Exp_Path"] = exp_dir
            cols = ["Virus","Experiment","Exp_Path"] + [c for c in wide_dt.columns if c not in ["Virus","Experiment","Exp_Path"]]
            wide_dt = wide_dt[cols]
            rows_wide.append(wide_dt)

    if rows_long:
        ALL_LONG = pd.concat(rows_long, ignore_index=True)
        ALL_LONG = ALL_LONG.sort_values(["Virus","Experiment","Metric"])
        out_long = os.path.join(SUMMARY_DIR, "all_models_metrics_long.csv")
        ALL_LONG.to_csv(out_long, index=False)
        print("✅ ذخیره شد:", out_long)
    else:
        print("⚠️ هیچ metrics_test.csv برای ساخت long پیدا نشد.")

    if rows_wide:
        ALL_WIDE = pd.concat(rows_wide, ignore_index=True)
        ALL_WIDE = ALL_WIDE.sort_values(["Virus","Experiment"])
        out_wide = os.path.join(SUMMARY_DIR, "all_models_metrics_wide.csv")
        ALL_WIDE.to_csv(out_wide, index=False)
        print("✅ ذخیره شد:", out_wide)
    else:
        print("⚠️ هیچ metrics_test.csv برای ساخت wide پیدا نشد.")

# ============================================================
# 6) Prediction روی کل ژنوم + Excel (TopN) + UpSet
# ============================================================
def gather_model_bundles() -> List[str]:
    bundles = []
    for root in [ROOT_PUB, ROOT_RAW]:
        for virus in VIRUSES:
            base = os.path.join(root, virus, OUT_SUBDIR, BAL_SUBDIR, "ML_results")
            if not os.path.isdir(base): continue
            for sd in [p for p in Path(base).iterdir() if p.is_dir()]:
                bf = sd / "model_bundle.joblib"
                if bf.exists():
                    bundles.append(str(bf))
    return bundles

def predict_whole_genome_and_export():
    PRED_DIR = os.path.join(ROOT_PUB, "_PREDICTIONS")
    ensure_dir(PRED_DIR)
    XLSX_PATH = os.path.join(PRED_DIR, "all_models_predictions.xlsx")
    UPSET_PATH= os.path.join(PRED_DIR, "upset_top1000.png")

    # بارگذاری فیچرها
    feat_all = read_features(FEATURES_CSV).copy()
    id_col = "Ensembl_ID"
    # تبدیل همهٔ ستون‌های غیر ID به عدد
    num_cols = [c for c in feat_all.columns if c != id_col]
    feat_num = feat_all.copy()
    for c in num_cols:
        feat_num[c] = pd.to_numeric(feat_num[c], errors="coerce")

    bundles = gather_model_bundles()
    if not bundles:
        raise RuntimeError("هیچ model_bundle.joblib پیدا نشد.")

    # Excel writer
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
        summary_rows = []
        top_sets = {}

        # یک شیت SUMMARY از ابتدا
        pd.DataFrame({"INFO":["Predictions summary"]}).to_excel(writer, sheet_name="SUMMARY", index=False)

        for bf in bundles:
            bundle, final_rf = load(bf)
            req_keys = {"imputer","scaler","sel_vars","all_vars_train","id_col","pos_class","rf_params"}
            if not req_keys.issubset(set(bundle.keys())):
                raise RuntimeError(f"باندل ناقص است: {bf}")

            exp_dir = os.path.dirname(bf)
            exp_name0 = os.path.basename(exp_dir)
            exp_name  = exp_name0
            exp_name  = re.sub(r"^filtered_HDF_", "", exp_name, flags=re.I)
            exp_name  = re.sub(r"_bottom\d+_HRF__top\d+_HDF$", "", exp_name, flags=re.I)
            exp_name  = re.sub(r"_balanced_top_vs_bottom$", "", exp_name, flags=re.I)
            sheet = make_safe_sheet_name(exp_name)

            pre_cols = bundle["all_vars_train"]
            missing = [c for c in pre_cols if c not in feat_num.columns]
            if missing:
                raise RuntimeError(f"این فیچرها در فایل فیچرها نیستند ({len(missing)}): {', '.join(missing[:10])} ...")

            # preprocess
            X_full_raw = feat_num[pre_cols].copy()
            X_full_imp = pd.DataFrame(bundle["imputer"].transform(X_full_raw), columns=pre_cols)
            X_full_scl = pd.DataFrame(bundle["scaler"].transform(X_full_imp), columns=pre_cols)

            sel_vars = bundle["sel_vars"] if bundle.get("sel_vars") else pre_cols
            for c in sel_vars:
                if c not in X_full_scl.columns:
                    raise RuntimeError(f"selected_vars در دادهٔ پردازش‌شده نیست: {c}")

            X_pred = X_full_scl[sel_vars].copy()
            if bundle["pos_class"] not in final_rf.classes_:
                raise RuntimeError("[PRED_PROB] کلاس مثبت در مدل وجود ندارد.")
            prob_pos = final_rf.predict_proba(X_pred)[:, list(final_rf.classes_).index(bundle["pos_class"])]

            res = pd.DataFrame({
                "Ensembl_ID": feat_all[id_col].astype(str).values,
                "Prob_HDF": prob_pos
            })
            res = res.sort_values(["Prob_HDF","Ensembl_ID"], ascending=[False, True]).reset_index(drop=True)
            res["Rank"] = np.arange(1, len(res)+1)

            tsv_path = os.path.join(PRED_DIR, f"{sheet}_predictions.tsv.gz")
            res.to_csv(tsv_path, sep="\t", index=False, compression="gzip")

            # Excel: فقط TopN (به‌صورت پیش‌فرض)
            res.head(1000).to_excel(writer, sheet_name=sheet, index=False)

            summary_rows.append({
                "Experiment": exp_name,
                "Sheet": sheet,
                "ModelBundle": bf,
                "TSV_GZ": tsv_path,
                "N_all": len(res),
                "TopN_in_Excel": min(1000, len(res)),
                "Top1000_MinProb": (res["Prob_HDF"].iloc[999] if len(res)>=1000 else np.nan)
            })
            top_sets[sheet] = set(res.head(1000)["Ensembl_ID"].tolist())

        # SUMMARY sheet
        if summary_rows:
            SUM = pd.DataFrame(summary_rows)
            # بازنویسی SUMMARY
            workbook = writer.book
            if "SUMMARY" in writer.sheets:
                std = writer.sheets["SUMMARY"]
                # پاک کردن و نوشتن مجدد
                # (ساده‌تر: ایجاد شیت جدید با نام SUMMARY2)
            SUM.to_excel(writer, sheet_name="SUMMARY", index=False)

    print(f"✅ Excel saved: {XLSX_PATH}")

    # UpSet plot
    if len(top_sets) >= 2:
        contents = {k:list(v) for k,v in top_sets.items()}
        inc = from_contents(contents)
        plt.figure(figsize=(12,7))
        UpSet(inc, subset_size='count', show_counts=True).plot()
        plt.suptitle("Top-1000 Overlaps (Predicted HDFs)")
        plt.tight_layout()
        plt.savefig(UPSET_PATH, dpi=200)
        plt.close()
        print(f"✅ UpSet saved: {UPSET_PATH}")
    else:
        print("⚠️ UpSet ساخته نشد (کمتر از ۲ مدل).")

# ============================================================
# اجرای کل پایپ‌لاین (در صورت تمایل به‌صورت مرحله‌ای اجرا کنید)
# ============================================================

# مرحله 1: ساخت Top/Bottom با نگاشت Ensembl
#step_build_top_bottom()

# مرحله 2: برچسب‌گذاری HDF/HRF روی فیچرها و ساخت CSV بالانس
#step_label_and_save()

# مرحله 3: یادگیری ماشین روی CSVهای بالانس‌شده + ذخیره نتایج و مدل‌ها
# step_ml_over_balanced_csvs()

# مرحله 4: ادغام متریک‌ها در یک پوشه Summary
# step_merge_metrics()

# مرحله 5: پیش‌بینی روی کل ژنوم + Excel + UpSet
# predict_whole_genome_and_export()


# %%
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

# %%
#For Ablation Experiments
# =========================
# ===== Ablation Suite ====
# =========================

import re
from dataclasses import dataclass

# --- 1) نگاشت گروه‌های فیچر به regex (پیشوندها را با داده‌ی خودت تطبیق بده)
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
    حذف گروه‌های فیچر از DataFrame (ستون Class و Genes حفظ می‌شود).
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

# --- 2) ساخت گراف با انتخاب یال‌ها و متاپث‌های مجاز
def add_budget_relations_with_toggles(data: HeteroData,
                                      use_hh=True, use_hv=True, use_vv=True) -> HeteroData:
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

    # fallback اگر هیچ φ ایجاد نشد:
    has_any_phi = any(k[1].startswith('phi') for k in data.edge_index_dict.keys() if k[0]=='gene' and k[2]=='gene')
    if not has_any_phi:
        if ('gene','interacts','gene') in data.edge_index_dict:
            data['gene','phi2','gene'].edge_index = unique_edges(data[('gene','interacts','gene')].edge_index)
        else:
            raise RuntimeError("No meta-path edges could be constructed.")
    return data

# --- 3) مدل انعطاف‌پذیر برای ابلیشن
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
            fused, alpha = self.semantic_att(H)
        else:
            fused = H.mean(dim=1)

        if self.use_post_mha:
            # امکان حذف virus token
            hv_key = ('gene','hv','virus') if ('gene','hv','virus') in edge_index_dict else ('gene','interacts','virus')
            if self.use_virus_token and hv_key in edge_index_dict and edge_index_dict[hv_key].numel() > 0:
                zv = aggregate_virus_mean(xv, edge_index_dict[hv_key], xg.size(0))
                tokens = torch.stack([xg, fused, zv], dim=1)
            else:
                tokens = torch.stack([xg, fused], dim=1)
            fused = self.post_mha(tokens)

        logit = self.cls(fused).squeeze(-1)
        return {'gene': logit}

# --- 4) سازنده‌ی دیتای پایه با فیچرهای سفارشی (برای حذف گروه‌های فیچر)
def build_base_heterodata_custom_features(features_df: pd.DataFrame, apply_undirected: bool = True):
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

    # توجه: فرض می‌کنیم همه‌ی ستون‌ها به جز Genes/Class فیچر هستند
    feat_cols = [c for c in features_df.columns if c not in ['Genes','Class']]
    X_gene  = torch.tensor(features_df[feat_cols].values, dtype=torch.float)
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
        data = T.ToUndirected()(data)
    return data, features_df, virus_df

# --- 5) Inner-CV اختصاصی برای ابلیشن (با مدل و تنظیمات انعطاف‌پذیر)
def inner_cv_tune_ablation(data_base: HeteroData,
                           y: np.ndarray,
                           outer_train_idx: np.ndarray,
                           in_gene: int,
                           in_virus: int,
                           model_kwargs: dict,
                           allow_phi: List[int],
                           use_hh=True, use_hv=True, use_vv=True,
                           enforce_phi4=False):
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
    return best_cfg, dict(auc_mean=np.mean(aucs), auc_sd=np.std(aucs),
                          aupr_mean=np.mean(auprs), aupr_sd=np.std(auprs))

# --- 6) اجرای یک ابلیشن و ثبت خروجی outer-test
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
    drop_feature_groups: List[str] = None  # e.g., ['ppi_topology']

def run_single_ablation(cfg: AblationCfg, out_csv_path: str):
    print(f"\n===== Ablation: {cfg.name} =====")

    # 1) بارگذاری و حذف فیچرها (در صورت نیاز)
    feat_df = pd.read_csv(FILE_GENE)
    if cfg.drop_feature_groups:
        feat_df = drop_feature_groups(feat_df, cfg.drop_feature_groups)
    data_base, genes_df, virus_df = build_base_heterodata_custom_features(feat_df, apply_undirected=True)

    # 2) اعمال یال‌های مجاز + متاپث‌های مجاز روی گراف کامل (برای outer-test)
    data_full = copy.deepcopy(data_base)
    data_full = add_budget_relations_with_toggles(data_full, use_hh=cfg.use_hh, use_hv=cfg.use_hv, use_vv=cfg.use_vv)
    data_full = add_metapath_phi_edges_allowed(data_full, cfg.allow_phi, enforce_phi4=cfg.require_phi4)

    in_gene  = data_base['gene'].x.size(1)
    in_virus = data_base['virus'].x.size(1)
    y = data_base['gene'].y.cpu().numpy()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for fold, (outer_train_idx, outer_test_idx) in enumerate(skf.split(np.zeros_like(y), y), 1):
        # --- tuning با inner-CV روی outer-train
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

        # گراف‌های آموزش/ولید + بودجه‌ها/متاپث‌ها
        data_train = induce_graph_by_genes(data_base, inner_train_idx)
        data_tv    = induce_graph_by_genes(data_base, np.concatenate([inner_train_idx, outer_valid_idx]))
        data_train = add_budget_relations_with_toggles(data_train, cfg.use_hh, cfg.use_hv, cfg.use_vv)
        data_tv    = add_budget_relations_with_toggles(data_tv,    cfg.use_hh, cfg.use_hv, cfg.use_vv)
        data_train = add_metapath_phi_edges_allowed(data_train, cfg.allow_phi, enforce_phi4=cfg.require_phi4)
        data_tv    = add_metapath_phi_edges_allowed(data_tv,    cfg.allow_phi, enforce_phi4=cfg.require_phi4)

        data_train, scg, scv = scale_features_in_fold(copy.deepcopy(data_train), inner_train_idx)
        data_tv    = apply_scalers_to_data(scg, scv, data_tv)
        data_full_ = apply_scalers_to_data(scg, scv, copy.deepcopy(data_full))

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

        # آستانه بر اساس outer-valid
        model.eval()
        with torch.no_grad():
            v_logits = model(data_tv.x_dict, data_tv.edge_index_dict)['gene'][torch.as_tensor(outer_valid_idx, device=DEVICE)]
            v_prob   = torch.sigmoid(v_logits).cpu().numpy()
            v_true   = y[outer_valid_idx]
        thr = find_best_threshold(v_true, v_prob, grid=201, metric='f1')

        # ارزیابی روی outer-test (گراف کاملِ مقیاس‌شده)
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
            WD=best_cfg['WEIGHT_DECAY']
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    print(f"[Ablation:{cfg.name}] saved -> {out_csv_path}")
    # چاپ خلاصه‌ی جدول (میانگین±SD برای مقاله)
    for m in ['AUROC','AUPRC','F1','Acc','Prec','Rec','Brier']:
        print(f"{m}: {df[m].mean():.3f} ± {df[m].std():.3f}")
    return df

# --- 7) فهرست ابلیشن‌ها و لانچر جدول 3
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
        # Feature ablations
        AblationCfg(name="−Topology", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['ppi_topology']),
        AblationCfg(name="−Node2Vec/DeepLoc", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['emb_loc']),
        AblationCfg(name="−GO/Pathway", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['go_pathway']),
        AblationCfg(name="−Sequence",  allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['sequence']),
        AblationCfg(name="−Conservation", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['conservation']),
        AblationCfg(name="−Domains", allow_phi=[1,2,3,4], require_phi4=True,
                    drop_feature_groups=['domains_motifs']),
    ]

    all_rows = []
    for cfg in ablations:
        df = run_single_ablation(cfg, out_csv_path=f"ablt_{cfg.name}_{VIRUS_NAME}.csv")
        all_rows.append(df)
    big = pd.concat(all_rows, axis=0).reset_index(drop=True)
    big.to_csv(out_path, index=False)
    print(f"\n=== TABLE 3 (outer-test, all ablations) -> {out_path} ===")

# --- فراخوانی اختیاری
# run_ablation_table3()


# %%
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

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
RF_ROOT   = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/HRF/Publications"
RF_PRED_DIR = os.path.join(RF_ROOT, "_PREDICTIONS")

# مسیر خروجی پیش‌بینی HAN (خروجی تابع run_fit_and_predict_all در کد HAN)
HAN_PRED_CSV = "Wholedata-HAN-predictions-Zika-final.csv"

# فایل برچسب‌خوردهٔ ژن‌ها (ورودی کد HAN)
FILE_GENE_LABELED = "/media/mohadeseh/d2987156-83a1-4537-b507-30f08b63b454/Naseri/FinalFolder/Zika/ZikaInputdataForDeepL500HDF1000Non39Features_WithClass.csv"

# گزینهٔ فیلتر بر اساس ویروس در نام فایل RF؛ اگر None باشد همه را در نظر می‌گیرد
VIRUS_KEYWORD_IN_RF_FILENAMES = "Zika"  # یا None

# مسیرهای خروجی
OUT_DIR = "_ENSEMBLE_OUT"
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
            # اگر در نام نبود، همه را بگیر
            files = pats
    else:
        files = pats
    if not files:
        raise RuntimeError(f"هیچ فایل پیش‌بینی RF در {pred_dir} یافت نشد.")

    dfs = []
    for p in files:
        try:
            df = _read_rf_pred_tsv_gz(p)
            df.rename(columns={'Prob_HDF': f'Prob_RF_{os.path.splitext(os.path.basename(p))[0]}'}, inplace=True)
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
    # در برخی خروحی‌ها ممکن است نام ستون‌ها کمی متفاوت باشد
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
    res = mg.querymany(syms, scopes='symbol,alias', fields='ensembl.gene', species='human', as_dataframe=False, returnall=False)
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
    y = out['Class'].astype(str).str.replace('-', '').str.lower()
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


def cv_meta_logreg(X: np.ndarray, y: np.ndarray, Cs: List[float] = [0.01,0.1,1,3,10,30], n_splits: int = 5, seed: int = 42) -> Tuple[LogisticRegression, Dict[str,float]]:
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
    # fit final
    lr_final = LogisticRegression(C=best_C, solver='liblinear')
    lr_final.fit(X, y)
    return lr_final, {'method':'stacking_logreg','best_C':best_C,'cv_auc':best_auc}


def cv_best_weight_average(p1: np.ndarray, p2: np.ndarray, y: np.ndarray, grid: int = 101, n_splits: int = 5, seed: int = 42) -> Tuple[float, Dict[str,float]]:
    # وزن w برای HAN: p = w*p1 + (1-w)*p2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ws = np.linspace(0.0, 1.0, grid)
    best_auc, best_w = -1.0, 0.5
    for w in ws:
        aucs = []
        for tr, te in skf.split(p1, y):
            pp = w*p1[tr] + (1-w)*p2[tr]
            aucs.append(roc_auc_score(y[tr], pp))
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
        {'Variant':'Chosen@0.5(AUC)', 'AUC':met_lab['AUC']},
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
    if A['Prob_HAN'].isna().any() and 'Prob_RF' in A:
        A.loc[A['Prob_HAN'].isna() & A['Prob_RF'].notna(), 'Prob_HAN'] = A.loc[A['Prob_HAN'].isna() & A['Prob_RF'].notna(), 'Prob_RF']
    if A['Prob_RF'].isna().any() and 'Prob_HAN' in A:
        A.loc[A['Prob_RF'].isna() & A['Prob_HAN'].notna(), 'Prob_RF'] = A.loc[A['Prob_RF'].isna() & A['Prob_HAN'].notna(), 'Prob_HAN']

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

# %%
"""
Ensemble (Stacking) without leakage for RF + HAN.

- Builds the same 10-fold stratified splits once.
- 8:1:1 train/val/test per fold.

- RF is trained inline in this script (no leakage; preprocessors fitted on train only).
- HAN is called via your han_hdf_han_final_cv.py utilities in inductive mode (no transductive leakage).

Outputs:
  ./_ENSEMBLE_OUT/
    oof_val_table.csv          # OOF predictions (val) per fold for RF/HAN + y
    perfold_test_table.csv     # test predictions per fold for RF/HAN + y
    stacking_meta_info.json    # chosen C, splits, scheme, seeds
    stacking_metrics_test.csv  # aggregated test metrics (AUC/AUPR/... for base & ensemble)
    ensemble_predictions_all_genes.csv  # final ensemble probs for ALL genes (using final HAN+RF preds)

"""

from __future__ import annotations
import os, json, math, warnings
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

# Use your han utilities
from han_hdf_han_final_cv import (
    build_base_heterodata, add_budget_relations, add_metapath_phi_edges,
    induce_graph_by_genes, scale_features_in_fold, apply_scalers_to_data,
    SimpleHAN, train_one_fold_inductive, evaluate_split, find_best_threshold,
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

def load_labeled_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Genes' not in df.columns or 'Class' not in df.columns:
        raise RuntimeError("Expected columns 'Genes' and 'Class' in labeled file.")
    # y: 1=HDF, 0=nonHDF
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
    if X.shape[1] <= 1: return list(X.columns)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    keep = [c for c in X.columns if c not in to_drop]
    return keep or list(X.columns)

def train_rf_fold_return_probs(X: pd.DataFrame, y: np.ndarray,
                               train_idx, val_idx, test_idx,
                               seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    # Preprocess fit on train only
    X_train = X.iloc[train_idx].copy()
    X_val   = X.iloc[val_idx].copy()
    X_test  = X.iloc[test_idx].copy()

    # NZV
    nzv = near_zero_var_mask(X_train)
    cols = [c for c in X_train.columns if not nzv[c]]
    X_train, X_val, X_test = X_train[cols], X_val[cols], X_test[cols]

    # Impute + Scale
    pp = Pipeline([('imp', SimpleImputer(strategy='median')),
                   ('sc', StandardScaler())])
    Xtr = pd.DataFrame(pp.fit_transform(X_train), index=X_train.index, columns=cols)
    Xva = pd.DataFrame(pp.transform(X_val),   index=X_val.index,   columns=cols)
    Xte = pd.DataFrame(pp.transform(X_test),  index=X_test.index,  columns=cols)

    # Drop highly correlated (on train only)
    sel = drop_high_correlation(Xtr, cutoff=0.70)
    Xtr, Xva, Xte = Xtr[sel], Xva[sel], Xte[sel]

    # Simple RF (strong, stable)
    p = Xtr.shape[1]
    mtry = max(1, int(round(math.sqrt(p))))
    rf = RandomForestClassifier(
        n_estimators=800, max_features=mtry, random_state=seed, n_jobs=-1, class_weight=None
    )
    rf.fit(Xtr, y[train_idx])

    # Prob for positive class (HDF=1)
    prob_val  = rf.predict_proba(Xva)[:, 1]
    prob_test = rf.predict_proba(Xte)[:, 1]
    return prob_val, prob_test


# -------------------------
# HAN — per-fold OOF (val) and test probs via your inductive pipeline
# -------------------------
def han_fold_return_probs(data_base,
                          in_gene: int, in_virus: int, y_all: np.ndarray,
                          train_idx, val_idx, test_idx,
                          seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    # Build graphs
    data_train    = induce_graph_by_genes(data_base, np.asarray(train_idx))
    data_trainval = induce_graph_by_genes(data_base, np.asarray(list(train_idx) + list(val_idx)))

    data_full = build_full_graph_scaled_later(data_base)

    # scale (fit on train_idx genes of data_train)
    data_train, sc_gene, sc_v = scale_features_in_fold(data_train, np.asarray(train_idx))
    data_trainval = apply_scalers_to_data(sc_gene, sc_v, data_trainval)
    data_full     = apply_scalers_to_data(sc_gene, sc_v, data_full)

    # Check φ4 exists on train graph
    phi_names_train = [k[1] for k in data_train.edge_index_dict.keys()
                       if (k[0]=='gene' and k[2]=='gene' and k[1].startswith('phi') and data_train[k].edge_index.numel()>0)]
    if 'phi4' not in phi_names_train:
        raise RuntimeError("φ4 required but missing in TRAIN graph.")

    # Model
    model = SimpleHAN(in_dim_gene=in_gene, in_dim_virus=in_virus,
                      phi_names=phi_names_train,
                      hidden=HIDDEN, heads=HEADS, dropout=DROPOUT,
                      use_post_mha=True, use_virus_token=True).to(DEVICE)

    model = train_one_fold_inductive(model, data_train, data_trainval,
                                     np.asarray(train_idx), np.asarray(val_idx),
                                     lr=LR, max_epochs=MAX_EPOCHS, weight_decay=WEIGHT_DECAY,
                                     patience=PATIENCE)

    # Prob on validation (train∪val graph)
    model.eval()
    with torch.no_grad():
        logits_val = model(data_trainval.x_dict, data_trainval.edge_index_dict)['gene'][torch.as_tensor(val_idx, device=DEVICE)]
        prob_val   = torch.sigmoid(logits_val).detach().cpu().numpy()

    # Prob on test (full graph)
    test_res, prob_test, _ = evaluate_split(model, data_full, np.asarray(test_idx), thr=None)
    # Here we need probs, not thresholding
    model.eval()
    with torch.no_grad():
        logits_test = model(data_full.x_dict, data_full.edge_index_dict)['gene'][torch.as_tensor(test_idx, device=DEVICE)]
        prob_test   = torch.sigmoid(logits_test).detach().cpu().numpy()

    return prob_val, prob_test


def build_full_graph_scaled_later(data_base):
    data_full = add_budget_relations(copy_like(data_base))
    data_full = add_metapath_phi_edges(data_full, include_vv=INCLUDE_VV_METAPATH, min_vv_edges=MIN_VV_EDGES_FOR_USE)
    return data_full

def copy_like(data):
    import copy as _copy
    return _copy.deepcopy(data)


# -------------------------
# RF + HAN → OOF VAL + TEST per fold
# -------------------------
def make_splits(y: np.ndarray, scheme: str = SPLIT_SCHEME, seed: int = SEED):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (train_pool, test_idx) in enumerate(skf.split(np.zeros_like(y), y), 1):
        if scheme == "han_5x5":
            # split held-out test_idx 50/50 into val/test
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
            te_labels = y[test_idx]
            val_rel, test_rel = next(sss.split(np.zeros_like(te_labels), te_labels))
            val_idx = test_idx[val_rel]
            test2_idx = test_idx[test_rel]
            train_idx = train_pool
        else:
            # 8:1:1 → test = held-out fold (10%), val = 10% of ALL chosen from the 90% train pool
            train_full = train_pool
            # val fraction inside the 90% pool to reach 10% of ALL
            val_frac_in_pool = 0.1 / 0.9
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
import re, glob, gzip

def load_han_all_preds(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect: Gene_Name | Predicted_Probability
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
        if not set(need).issubset(d.columns): continue
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

    # ----- 0) Labeled data (for RF features + labels)
    lab_short, lab_full = load_labeled_df(FILE_GENE)
    gene_ids = lab_short['Ensembl_ID'].values.astype(str)
    y = lab_short['y'].values.astype(int)

    # RF feature matrix from FILE_GENE (all non-id/non-class columns)
    X_rf = lab_full.drop(columns=['Genes','Class'])
    X_rf.columns = [str(c) for c in X_rf.columns]
    X_rf = X_rf.apply(pd.to_numeric, errors='coerce')
    if X_rf.isna().any().any():
        X_rf = X_rf.fillna(X_rf.median())

    # ----- 1) Build base hetero graph for HAN (once)
    data_base, genes_df, virus_df = build_base_heterodata(apply_undirected=True)
    in_gene  = data_base['gene'].x.size(1)
    in_virus = data_base['virus'].x.size(1)
    # Sanity: ensure same gene order between data_base and lab file
    # here we assume FILE_GENE order matches data_base['gene'] order.
    # If not, you may map indices by name; for this dataset they are aligned.

    # ----- 2) OOF containers
    n = len(y)
    oof_val_rf  = np.full(n, np.nan, dtype=float)
    oof_val_han = np.full(n, np.nan, dtype=float)
    oof_val_y   = np.full(n, np.nan, dtype=float)

    test_rows = []  # will collect rows for all folds: {'idx':..,'fold':..,'prob_rf':..,'prob_han':..,'y':..}

    # ----- 3) Iterate folds (shared splits)
    for fold, train_idx, val_idx, test_idx in make_splits(y, scheme=SPLIT_SCHEME, seed=SEED):
        print(f"[Fold {fold}] sizes -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # RF per fold
        prob_rf_val, prob_rf_test = train_rf_fold_return_probs(X_rf, y, train_idx, val_idx, test_idx)

        # HAN per fold
        prob_han_val, prob_han_test = han_fold_return_probs(
            data_base, in_gene, in_virus, y,
            train_idx, val_idx, test_idx, seed=SEED
        )

        # Fill OOF val
        oof_val_rf[val_idx]  = prob_rf_val
        oof_val_han[val_idx] = prob_han_val
        oof_val_y[val_idx]   = y[val_idx]

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

    # ----- 4) Save OOF val table & test table
    oof_val_df = pd.DataFrame({
        'Index': np.arange(n, dtype=int),
        'Ensembl_ID': gene_ids,
        'y': oof_val_y,
        'Prob_RF': oof_val_rf,
        'Prob_HAN': oof_val_han
    }).dropna(subset=['y','Prob_RF','Prob_HAN'])
    oof_val_df.to_csv(os.path.join(OUT_DIR, "oof_val_table.csv"), index=False)

    test_df = pd.DataFrame(test_rows)
    test_df = test_df.sort_values(['Fold','Index']).reset_index(drop=True)
    test_df.to_csv(os.path.join(OUT_DIR, "perfold_test_table.csv"), index=False)

    # ----- 5) Fit stacking meta-learner on OOF(val), evaluate on test
    X_val = oof_val_df[['Prob_RF','Prob_HAN']].values
    y_val = oof_val_df['y'].values.astype(int)

    # choose C by inner CV
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
        'meta_best_C': best_C,
        'meta_cv_auc_on_OOF_val': best_auc
    }
    with open(os.path.join(OUT_DIR, "stacking_meta_info.json"), 'w') as f:
        json.dump(info, f, indent=2)

    print("\n=== Test metrics (aggregated across folds) ===")
    print(met_tbl.to_string(index=False))

    # ----- 6) Apply ensemble to ALL genes (final HAN+RF global preds)
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
