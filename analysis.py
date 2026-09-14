"""
Analysis of results.csv -> paired comparisons, tests, CSV tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests

import config as C

KEYS = ["dataset", "task", "n", "d", "missing_rate", "mechanism", "inf_prob", "model"]

N_BOOT = 4000
BOOT_SEED = 12345

def _cluster_bootstrap_ci(values, clusters, n_boot=N_BOOT, alpha=0.05, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.size < 2:
        return (np.nan, np.nan)
    uniq = np.unique(clusters)
    if uniq.size < 2:
        return (np.nan, np.nan)
    blocks = [values[clusters == c] for c in uniq]
    rng = np.random.default_rng(seed)
    k = len(blocks)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        means[b] = np.concatenate([blocks[i] for i in pick]).mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

def _summarize(values, clusters) -> dict:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    keep = ~np.isnan(values)
    values, clusters = values[keep], clusters[keep]
    n = values.size

    out = dict(n=n, n_datasets=int(np.unique(clusters).size) if n else 0,
               mean=np.nan, median=np.nan, wins=0, losses=0, ties=0,
               win_rate=np.nan, ci_low=np.nan, ci_high=np.nan,
               n_eff=0, sign_p=np.nan, wilcoxon_p=np.nan, wilcoxon_p_1s=np.nan)
    if n == 0:
        return out

    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    ties = int(np.sum(values == 0))
    out.update(mean=float(np.mean(values)), median=float(np.median(values)),
               wins=wins, losses=losses, ties=ties, n_eff=wins + losses)
    if wins + losses > 0:
        out["win_rate"] = wins / (wins + losses)
        out["sign_p"] = float(binomtest(wins, wins + losses, 0.5).pvalue)
    out["ci_low"], out["ci_high"] = _cluster_bootstrap_ci(values, clusters)
    if wins + losses > 0:
        try:
            out["wilcoxon_p"] = float(wilcoxon(values).pvalue)
            out["wilcoxon_p_1s"] = float(
                wilcoxon(values, alternative="greater").pvalue)
        except Exception:
            pass
    return out

def _grouped(df, value_col, groupcols, unit="dataset"):
    rows = []
    for keys, g in df.groupby(groupcols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(groupcols, keys))
        g = g.dropna(subset=[value_col])
        if unit == "dataset":
            s = g.groupby("dataset")[value_col].mean()
            rec.update(_summarize(s.values, s.index.values))
        else:
            rec.update(_summarize(g[value_col].values, g["dataset"].values))
        rec["unit"] = unit
        rows.append(rec)
    return pd.DataFrame(rows)

def _correct_family(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    df["wilcoxon_p_adj"] = np.nan
    df["wilcoxon_reject"] = False
    if "wilcoxon_p" not in df.columns or df.empty:
        return df
    mask = df["wilcoxon_p"].notna()
    if mask.any():
        reject, p_adj, _, _ = multipletests(
            df.loc[mask, "wilcoxon_p"].values, alpha=0.05, method=method)
        df.loc[mask, "wilcoxon_p_adj"] = p_adj
        df.loc[mask, "wilcoxon_reject"] = reject
    return df

def build_pivot(df):
    ok = df[df["status"] == "ok"].copy()
    ok["score"] = pd.to_numeric(ok["score"], errors="coerce")
    ok["inf_prob"] = ok["inf_prob"].fillna("NA").astype(str)
    wide = ok.pivot_table(index=KEYS + ["imputation"], columns="indicator",
                          values="score", aggfunc="mean").reset_index()
    return wide

def main():
    df = pd.read_csv(C.RESULTS_CSV)

    wide = build_pivot(df)
    for col in ["none", "MIM", "SMIM"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["help_MIM"] = wide["none"] - wide["MIM"]
    wide["help_SMIM"] = wide["none"] - wide["SMIM"]
    wide["gap_SMIM_MIM"] = wide["MIM"] - wide["SMIM"]

    mi_rows = []
    for ind in ["none", "MIM", "SMIM"]:
        p = wide.pivot_table(index=KEYS, columns="imputation",
                             values=ind).reset_index()
        if not {"single", "multiple"}.issubset(p.columns):
            continue
        p["mi"] = p["single"] - p["multiple"]
        for unit in ("dataset", "cell"):
            r = _grouped(p.dropna(subset=["mi"]).assign(_g=1), "mi", ["_g"],
                         unit=unit).drop(columns="_g")
            r.insert(0, "indicator", ind)
            mi_rows.append(r)
    table_mi = pd.concat(mi_rows, ignore_index=True) if mi_rows else pd.DataFrame()
    if not table_mi.empty:
        table_mi = pd.concat(
            [_correct_family(table_mi[table_mi.unit == "dataset"], "holm"),
             table_mi[table_mi.unit == "cell"]], ignore_index=True)

    grp = ["mechanism", "inf_prob", "imputation"]
    help_tables = []
    for unit in ("dataset", "cell"):
        a = _grouped(wide.dropna(subset=["help_MIM"]), "help_MIM", grp, unit=unit)
        a.insert(0, "comparison", "none - MIM")
        b = _grouped(wide.dropna(subset=["help_SMIM"]), "help_SMIM", grp, unit=unit)
        b.insert(0, "comparison", "none - SMIM")
        t = pd.concat([a, b], ignore_index=True)
        help_tables.append(_correct_family(t, "holm") if unit == "dataset" else t)
    table_help = pd.concat(help_tables, ignore_index=True)

    gap_tables = []
    for unit in ("dataset", "cell"):
        t = _grouped(wide.dropna(subset=["gap_SMIM_MIM"]), "gap_SMIM_MIM", grp,
                     unit=unit)
        gap_tables.append(_correct_family(t, "holm") if unit == "dataset" else t)
    table_gap = pd.concat(gap_tables, ignore_index=True)

    table_gap_task = _correct_family(
        _grouped(wide.dropna(subset=["gap_SMIM_MIM"]), "gap_SMIM_MIM",
                 grp + ["task"], unit="dataset"), "fdr_bh")
    table_gap_model = _correct_family(
        _grouped(wide.dropna(subset=["gap_SMIM_MIM"]), "gap_SMIM_MIM",
                 grp + ["model"], unit="dataset"), "fdr_bh")
    wide_multiple = wide[wide["imputation"] == "multiple"].copy()

    table_gap_ds = (
        wide_multiple
        .pivot_table(
            index=["dataset", "task", "d"],
            columns=["mechanism", "inf_prob"],
            values="gap_SMIM_MIM",
            aggfunc="mean",
        )
        .reset_index()
    )

    trend = (
        wide.groupby(["dataset", "mechanism", "inf_prob", "imputation"])
        ["gap_SMIM_MIM"]
        .mean()
        .reset_index()
    )

    trend_tab = (
        trend.assign(cond=trend["mechanism"] + "/" + trend["inf_prob"])
        .pivot_table(
            index=["dataset", "imputation"],
            columns="cond",
            values="gap_SMIM_MIM",
        )
        .reset_index()
    )

    trend_rows = []
    for a, b in [
        ("MCAR/NA", "MNAR/0.1"),
        ("MNAR/0.1", "MNAR/0.5"),
        ("MCAR/NA", "MNAR/0.5"),
    ]:
        if a in trend_tab.columns and b in trend_tab.columns:
            for imp in sorted(trend_tab["imputation"].unique()):
                s = trend_tab[trend_tab.imputation == imp].dropna(subset=[a, b])
                d = (s[a] - s[b]).values
                rec = dict(contrast=f"gap({a}) - gap({b})", imputation=imp)
                rec.update(_summarize(d, s["dataset"].values))
                trend_rows.append(rec)

    table_trend = _correct_family(pd.DataFrame(trend_rows), "holm")


    gap = wide.pivot_table(
        index=KEYS,
        columns="imputation",
        values="gap_SMIM_MIM",
        aggfunc="mean",
    ).reset_index()

    if {"multiple", "single"}.issubset(gap.columns):
        gap["interaction"] = gap["multiple"] - gap["single"]
        inter_tables = []
        for unit in ("dataset", "cell"):
            t = _grouped(
                gap.dropna(subset=["interaction"]),
                "interaction",
                ["mechanism", "inf_prob"],
                unit=unit,
            )
            inter_tables.append(_correct_family(t, "holm") if unit == "dataset" else t)
        table_inter = pd.concat(inter_tables, ignore_index=True)
    else:
        table_inter = pd.DataFrame()

    out = {
        "table_mi_single_vs_multiple": table_mi,
        "table_indicator_help": table_help,
        "table_smim_vs_mim": table_gap,
        "table_smim_vs_mim_by_task": table_gap_task,
        "table_smim_vs_mim_by_model": table_gap_model,
        "table_smim_vs_mim_by_dataset": table_gap_ds,
        "table_infprob_trend": table_trend,
        "table_interaction": table_inter,
    }

    for name, t in out.items():
        t.to_csv(f"{C.OUTPUT_DIR}/{name}.csv", index=False)

    print(f"Analysis complete. Tables written to {C.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
