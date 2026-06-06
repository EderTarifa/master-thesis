"""
Regenera TODAS las tablas por-variante del capitulo 4 desde una UNICA fuente
(results/full/runs.parquet y results/full_optimal/runs.parquet), de modo que
las cifras de la memoria sean reproducibles y mutuamente consistentes.

Convencion canonica de la comparacion entre variantes:
    - Mercados de renta variable: DJIA, SP50, IBEX
    - Semillas 0-4 (las 5 comunes a TODAS las variantes; V0/V4 tienen 5 mas
      reservadas para el estudio de convergencia, que NO se mezclan aqui)
    => n = 3 mercados x 13 folds x 5 semillas = 195 episodios por variante.

Uso (desde la raiz de master-thesis/):
    python scripts/regenerate_chapter4.py

Salidas en results/_canonical_tables/:
    tabla_4_1_original.tex        (por-variante, experimento original)
    tabla_4_2_optimal.tex         (por-variante, full optimal)
    tabla_4_3_comparacion.tex     (MDD original vs full optimal)
    tabla_4_4_contrastes_original.tex
    tabla_4_5_contrastes_optimal.tex
    canonical_summary.csv         (todas las cifras, para verificacion)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

EQ = ["DJIA", "SP50", "IBEX"]
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["V0", "V1", "V2", "V3", "V4"]
CONTRASTS = [("V0", "V1"), ("V0", "V2"), ("V0", "V3"), ("V0", "V4"),
             ("V1", "V2"), ("V1", "V3"), ("V1", "V4"), ("V2", "V4"), ("V3", "V4")]
METRICS = ["mdd", "cdar_95", "sharpe", "calmar", "turnover_mean"]
MLABEL = {"mdd": "MDD", "cdar_95": "CDaR$_{95}$", "sharpe": "Sharpe",
          "calmar": "Calmar", "turnover_mean": "Turnover"}

OUT = Path("results/_canonical_tables")


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(f"results/{name}/runs.parquet")
    return df[df.market.isin(EQ) & df.seed.isin(SEEDS)].copy()


def per_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variant")[METRICS].mean().reindex(VARIANTS)
    g["n"] = df.groupby("variant").size().reindex(VARIANTS).astype(int)
    return g


def paired_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Contrastes pareados sobre el MDD, emparejando por (market, fold, seed)."""
    key = ["market", "fold", "seed"]
    wide = df.pivot_table(index=key, columns="variant", values="mdd")
    rows = []
    for base, treat in CONTRASTS:
        pair = wide[[base, treat]].dropna()
        d = (pair[treat] - pair[base]).values          # < 0 => treat mejora
        # t-test pareado unilateral H1: treat < base
        t_stat, t_p = stats.ttest_rel(pair[treat], pair[base], alternative="less")
        # Wilcoxon unilateral
        try:
            w_stat, w_p = stats.wilcoxon(pair[treat], pair[base], alternative="less")
        except ValueError:
            w_p = np.nan
        rows.append({
            "base": base, "treat": treat, "n": len(d),
            "delta_pp": 100 * d.mean(),
            "median_pp": 100 * np.median(d),
            "pct_better": 100 * (d < 0).mean(),
            "p_t": float(t_p), "p_w": float(w_p),
        })
    return pd.DataFrame(rows)


# --------------------------- LaTeX writers --------------------------------

def _p(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x < 1e-4:
        return f"${x:.1e}".replace("e-0", "e{-").replace("e-", "e{-") + "}$"
    return f"{x:.3g}"


def write_per_variant(tab: pd.DataFrame, path: Path, caption: str, label: str):
    L = ["% Generado por scripts/regenerate_chapter4.py -- NO editar a mano",
         "\\begin{table}[ht]", "  \\centering", f"  \\caption{{{caption}}}",
         f"  \\label{{{label}}}", "  \\begin{tabular}{lccccc}", "    \\toprule",
         "    Variante & " + " & ".join(MLABEL[m] for m in METRICS) + " \\\\",
         "    \\midrule"]
    for v in VARIANTS:
        r = tab.loc[v]
        cells = [f"{r['mdd']:.4f}", f"{r['cdar_95']:.4f}", f"{r['sharpe']:.3f}",
                 f"{r['calmar']:.3f}", f"{r['turnover_mean']:.4f}"]
        L.append(f"    {v} & " + " & ".join(cells) + " \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_comparison(orig, opt, path: Path):
    L = ["% Generado por scripts/regenerate_chapter4.py -- NO editar a mano",
         "\\begin{table}[ht]", "  \\centering",
         "  \\caption{Comparacion del MDD medio entre experimentos "
         "(renta variable, $n=195$ por variante). Diferencia absoluta en puntos "
         "porcentuales.}", "  \\label{tab:mdd_comparison}",
         "  \\begin{tabular}{lccc}", "    \\toprule",
         "    Variante & Original & Full optimal & $\\Delta$ (pp) \\\\", "    \\midrule"]
    for v in VARIANTS:
        o, p = orig.loc[v, "mdd"], opt.loc[v, "mdd"]
        L.append(f"    {v} & {o:.4f} & {p:.4f} & {100*(p-o):+.2f} \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_contrasts(ct: pd.DataFrame, path: Path, caption: str, label: str):
    L = ["% Generado por scripts/regenerate_chapter4.py -- NO editar a mano",
         "\\begin{table}[ht]", "  \\centering", f"  \\caption{{{caption}}}",
         f"  \\label{{{label}}}", "  \\begin{tabular}{llccccc}", "    \\toprule",
         "    Base & Trat. & $\\Delta$ (pp) & $\\tilde{\\Delta}$ (pp) & "
         "$P_{<}$ & $p_t$ & $p_W$ \\\\", "    \\midrule"]
    for _, r in ct.iterrows():
        L.append(f"    {r['base']} & {r['treat']} & {r['delta_pp']:.2f} & "
                 f"{r['median_pp']:.2f} & {r['pct_better']:.1f}\\% & "
                 f"{_p(r['p_t'])} & {_p(r['p_w'])} \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    orig_df, opt_df = load("full"), load("full_optimal")
    orig_t, opt_t = per_variant_table(orig_df), per_variant_table(opt_df)
    orig_c, opt_c = paired_contrasts(orig_df), paired_contrasts(opt_df)

    write_per_variant(orig_t, OUT / "tabla_4_1_original.tex",
                      "Metricas medias por variante, experimento original "
                      "($\\lambda_1=1{,}0$, $\\lambda_2=0{,}5$, 200k pasos, $n=195$).",
                      "tab:per_variant_original")
    write_per_variant(opt_t, OUT / "tabla_4_2_optimal.tex",
                      "Metricas medias por variante, experimento full optimal "
                      "($\\lambda_1=2{,}0$, $\\lambda_2=2{,}0$, 100k pasos, $n=195$).",
                      "tab:per_variant_optimal")
    write_comparison(orig_t, opt_t, OUT / "tabla_4_3_comparacion.tex")
    write_contrasts(orig_c, OUT / "tabla_4_4_contrastes_original.tex",
                    "Contrastes pareados sobre el MDD, experimento original "
                    "($n=195$). $\\Delta$ media en pp; $\\tilde{\\Delta}$ mediana; "
                    "$P_{<}$ \\% de pares en que la variante de tratamiento mejora.",
                    "tab:contrasts_original")
    write_contrasts(opt_c, OUT / "tabla_4_5_contrastes_optimal.tex",
                    "Contrastes pareados sobre el MDD, experimento full optimal "
                    "($n=195$).", "tab:contrasts_optimal")

    summary = pd.concat([orig_t.add_prefix("orig_"), opt_t.add_prefix("opt_")], axis=1)
    summary.to_csv(OUT / "canonical_summary.csv")

    print("=" * 78)
    print("TABLA POR VARIANTE -- FULL OPTIMAL (canonica, n=195)")
    print("=" * 78)
    print(opt_t.round(4).to_string())
    print("\nContrastes clave (full optimal):")
    for _, r in opt_c[opt_c.base.isin(["V0", "V1"])].iterrows():
        print(f"  {r['base']}->{r['treat']}: dMDD={r['delta_pp']:+.2f}pp  "
              f"%mejor={r['pct_better']:.1f}%  p_t={r['p_t']:.2e}  p_W={r['p_w']:.2e}")
    print(f"\nLaTeX + canonical_summary.csv escritos en: {OUT}/")


if __name__ == "__main__":
    main()