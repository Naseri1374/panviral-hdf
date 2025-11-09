# panviral-hdf
Code for a pan-viral map of Host Dependency Factors (HDFs) across **IAV, SARS-CoV-2, ZIKV, DENV** using **MAIC**, a **Heterogeneous Graph Neural Network (HAN)** with meta-paths, and a **Random Forest (RF) + XGBoost + ensemble**. Includes leakage-safe splits, **OOF stacking**, and **ablations**.

---

## What’s inside
- **HAN** 
- **RF and XGBoost** on tabular features
- **Ensembling** via **OOF stacking** (no leakage) + simple weighted average
- **Ablations** for components & feature groups

---

## Project structure

├─ han_hdf_han_final_cv.py # HAN: nested CV + tuning + final predict-all (ZIKV example)
├─ han_ablation_table3.py # Ablations (optional)
├─ ensemble_oof_stacking.py # Main ensemble (OOF stacking; no leakage)
├─ ensemble_merge_rf_han_all.py # Simple HAN+RF merge (weighted avg / meta-LR)
└─ _ENSEMBLE_OUT/ # Ensemble outputs (created at runtime)

---

## Data requirements

**1) Gene features + labels**
- Columns: `Genes` (Ensembl ID), many feature columns, `Class` (`HDF`/`nonHDF` or 1/0)  

**2) Gene–Virus interactions**
- Columns: `Genes` (Ensembl), `GenesVirus` (matches virus feature table)  

**3) Human PPI**
- Columns: `Ensembl_ID_A`, `Ensembl_ID_B`  

**4) Virus features**
- Columns: `GenesSymbolVirus`, plus virus feature columns

**5) Virus–Virus interactions**
- Columns depend on virus.  
  - Zika example: `Official_Symbol_A_Zika`, `Official_Symbol_B_Zika`

**6) Whole-genome gene features (no `Class`)**
- Columns: `Genes`, feature columns

---

## Environment
Python **3.10** recommended.

```bash
conda create -n hdf_env python=3.10 -y
conda activate hdf_env

# Core
pip install numpy pandas scikit-learn openpyxl mygene

# PyTorch (pick the build that matches your CUDA)
pip install torch==2.*

# PyG (+ scatter/sparse/cluster): match your Torch/CUDA
# Example wheel index (adjust to your versions):
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
