# SMIM vs. MIM under stochastic multiple imputation

Experiment for the research question:

**Does the Selective Missing Indicator Method (SMIM) outperform the standard
Missing Indicator Method (MIM) when both are combined with stochastic multiple
imputation, on tabular datasets with missing values?**

It reuses the methodology of Van Ness et al., *The Missing Indicator Method:
From Low to High Dimensions* (KDD 2023, [repo](https://github.com/mvanness354/missing_indicator_method)):
the self-masking MNAR mechanism and the χ²/Welch-t-test + Benjamini–Hochberg
indicator selection used by SMIM. It adds the new axis the paper did not study:
**stochastic multiple imputation** and a **no-indicator** baseline.

## Environment

```powershell
python -m venv .venv
pip install -r requirements.txt
```
## Run
```powershell
python experiment.py
python analysis.py
```

Outputs land in `outputs/`:
- `results.csv` — one row per experimental cell.
- `table_mi_single_vs_multiple.csv` — single vs. multiple imputation.
- `table_indicator_help.csv` — does any indicator beat imputation-only?
- `table_smim_vs_mim.csv` — overall SMIM vs. MIM gap.
- `table_smim_vs_mim_by_task.csv` — SMIM vs. MIM by task type.
- `table_smim_vs_mim_by_model.csv` — SMIM vs. MIM by model family.
- `table_smim_vs_mim_by_dataset.csv` — per-dataset SMIM vs. MIM gaps.
- `table_infprob_trend.csv` — trend across missingness/informativeness conditions.
- `table_interaction.csv` — does multiple imputation change the SMIM–MIM gap?
