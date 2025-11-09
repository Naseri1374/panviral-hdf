# panviral-hdf
Code for a pan-viral map of host dependency factors (IAV, SARS-CoV-2, ZIKV, DENV) using MAIC, HAN, and RF/ensemble; includes ablations and stacking.

This repository contains a leakage-safe pipeline to predict host dependency factors (HDFs) for viruses using:

Heterogeneous graph neural network (HAN) with meta-paths

Random Forest (RF) on tabular features

Ensembling via OOF stacking (no leakage) and a simple weighted average baseline

Ablation runs for model components and feature groups

1) Project Structure
.
├─ han_hdf_han_final_cv.py          
├─ han_ablation_table3.py           
├─ ensemble_oof_stacking.py         
├─ ensemble_merge_rf_han_all.py     
├─ _ENSEMBLE_OUT/                   
└─ data/                            

2) Data Requirements

Gene features + labels:
Columns: Genes (Ensembl ID), many feature columns, Class (HDF/nonHDF or 1/0)

Gene–Virus interactions:
Columns: Genes (Ensembl), GenesVirus (matches virus feature table)

Human PPI:
Columns: Ensembl_ID_A, Ensembl_ID_B

Virus features:
Columns: GenesSymbolVirus, plus virus feature columns

Virus–Virus interactions:

Columns: Official_Symbol_A, Official_Symbol_B

Whole-genome gene features (no Class):

Columns: Genes, feature columns

3) Environment Setup

Python 3.10 is recommended.

conda create -n hdf_env python=3.10 -y
conda activate hdf_env

# Core
pip install numpy pandas scikit-learn openpyxl mygene

# PyTorch (choose a build that matches your CUDA)
pip install torch==2.*  # or use the official install command from pytorch.org

# PyG (+ scatter/sparse/cluster) – pick wheels compatible with your Torch/CUDA
# Example URL shown; adjust for your torch/cuda versions:
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

GPU strongly recommended. If PyG installation fails, follow the instructions on pyg.org

4) Quickstart (Zika example)
4.1 Configure paths

Open the scripts and set the file paths at the top to your local locations. Key ones:

In han_hdf_han_final_cv.py:

FILE_GV, FILE_GENE, FILE_PPI_H, FILE_VIRUS, FILE_PPI_VV, FILE_GENE_ALL

VIRUS_NAME = "ZIKV"

φ4 is required; make sure your VV file has edges. If not, see Troubleshooting.


4.2 Run HAN
python han_hdf_han_final_cv.py

Outputs:

tuning_summary.csv – per-fold tuning results

consensus_hparams.csv – mode of best configs (table-ready)

Wholedata-HAN-predictions-final.csv ✅ (probabilities for all genes)

4.3 Run Ensemble (OOF stacking; no leakage)
python ensemble_oof_stacking.py

Outputs (in _ENSEMBLE_OUT/):

oof_val_table.csv – OOF val predictions for RF/HAN + y

perfold_test_table.csv – test predictions per fold

stacking_metrics_test.csv – RF vs HAN vs Stacking metrics on test

stacking_meta_info.json – meta-learner C, seeds, scheme

ensemble_predictions_all_genes.csv ✅ – final ensemble for all genes

4.4 (Optional) Ablations (Table 3)

Make sure the script calls run_ablation_table3() (uncomment if needed), then:

python han_ablation_table3.py

Outputs:

Several ablt_<name>.csv files

ablation_table3.csv – summary table

5) Reproducibility

Global seed: SEED = 42 (changeable)

Stratified K-Fold: N_SPLITS = 10

HAN nested CV: inner folds for tuning (leakage-safe)

Ensemble: OOF stacking (meta-learner trains only on OOF val)

6) Key Notes & Gotchas

φ4 (H→V→V→H) is enforced in HAN: you must have VV edges and consistent H↔V mappings.

If your VV file is sparse, keep MIN_VV_EDGES_FOR_USE = 1 or lower budgets.

If VV is unavailable: set INCLUDE_VV_METAPATH=False and remove φ4 requirements in the model. Model performance may change.

Column names must match (case-sensitive):

HV file: Genes, GenesVirus

Human PPI: Ensembl_ID_A, Ensembl_ID_B

Virus feats: GenesSymbolVirus

VV (Zika): Official_Symbol_A_Zika, Official_Symbol_B_Zika (rename in code for other viruses)

Gene order assumption: By default, we assume FILE_GENE order matches the constructed graph’s gene order. If you see mismatches, add an index mapping by Ensembl ID before training.

GPU memory: If you hit OOM, try HIDDEN=128, HEADS=2, reduce MAX_EPOCHS, or increase DROPOUT.

RF all-genes predictions: The ensemble needs a folder of *.tsv.gz with Ensembl_ID, Prob_HDF. If you don’t have them yet, run your RF pipeline first (outside this repo) to generate them.

7) Running on a Different Virus

Replace the data files with the new virus’ files.

Update VV column names in code (e.g., Official_Symbol_A_<Virus>).

Set VIRUS_NAME and (optionally) RF_FILENAME_FILTER_KEYWORD.

Re-run HAN and then ensemble (steps 4.2 and 4.3).
