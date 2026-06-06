"""
Genera la tabla limpia de validacion EVT (Kupiec + Christoffersen) en LaTeX
a partir de results/sidequest_evt/tail_backtest_all.csv.

Convencion de signos (debe coincidir con sidequest_evt_analysis.py):
    L_t = -r_t  (perdida; L_t > 0 es perdida)
    VaR_alpha(L) y CVaR_alpha(L) son POSITIVOS
    violacion  <=>  L_t > VaR_alpha(L)
    tasa de violacion esperada = 1 - alpha  (0.05 a 95 %, 0.01 a 99 %)

Uso:
    python scripts/make_kupiec_table.py

Salida:
    results/sidequest_evt/tabla_kupiec.tex   (\\input desde el capitulo 3)
    + resumen por consola
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CSV = Path("results/sidequest_evt/tail_backtest_all.csv")
OUT = Path("results/sidequest_evt/tabla_kupiec.tex")

MODEL_ORDER = ["Empirical", "Normal", "EVT_POT"]
MODEL_LABEL = {"Empirical": "Empirico", "Normal": "Normal", "EVT_POT": "EVT-POT"}


def _fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 1e-4:
        return "$<\\!10^{-4}$"
    return f"{p:.3f}"


def _verdict(kp: float) -> str:
    """Veredicto de cobertura incondicional (Kupiec) al 5 %."""
    if kp is None or (isinstance(kp, float) and np.isnan(kp)):
        return "--"
    return "\\checkmark" if kp > 0.05 else "$\\times$"


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"No existe {CSV}. Ejecuta antes sidequest_evt_analysis.py")
    df = pd.read_csv(CSV)

    # Orden estable: mercado, alpha, modelo
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["market", "alpha", "model"]).reset_index(drop=True)

    lines = []
    lines.append("% Generado por scripts/make_kupiec_table.py -- NO editar a mano")
    lines.append("\\begin{table}[ht]")
    lines.append("  \\centering")
    lines.append("  \\caption{Backtest de cobertura del VaR sobre la cartera 1/N "
                 "(ventana movil de 500 dias). Convencion: $L_t=-r_t$; una violacion "
                 "ocurre cuando $L_t>\\mathrm{VaR}_\\alpha(L)$, con tasa esperada "
                 "$1-\\alpha$. $p_{\\mathrm{Kup}}$ es el $p$-valor del test de Kupiec "
                 "(cobertura incondicional) y $p_{\\mathrm{Chr}}$ el de independencia de "
                 "Christoffersen. \\checkmark{} indica no rechazo de calibracion "
                 "correcta al 5\\,\\%.}")
    lines.append("  \\label{tab:kupiec_validation}")
    lines.append("  \\begin{tabular}{llccccc}")
    lines.append("    \\toprule")
    lines.append("    Mercado & Modelo & $\\alpha$ & Viol./$n$ & Tasa & "
                 "$p_{\\mathrm{Kup}}$ & $p_{\\mathrm{Chr}}$ \\\\")
    lines.append("    \\midrule")

    for market in df["market"].unique():
        sub = df[df["market"] == market]
        first = True
        for _, r in sub.iterrows():
            mkt = market if first else ""
            first = False
            rate = 100 * r["kupiec_rate"]
            exp = 100 * r["kupiec_expected_rate"]
            viol = int(r["kupiec_violations"])
            n = int(r["kupiec_n"])
            lines.append(
                f"    {mkt} & {MODEL_LABEL[r['model']]} & "
                f"{int(round(100*r['alpha']))}\\% & "
                f"{viol}/{n} & {rate:.2f}\\% (esp.\\ {exp:.0f}\\%) & "
                f"{_fmt_p(r['kupiec_pvalue'])}\\,{_verdict(r['kupiec_pvalue'])} & "
                f"{_fmt_p(r['chr_pvalue'])} \\\\"
            )
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Resumen por consola ----
    print("=" * 72)
    print("RESUMEN DE VALIDACION EVT (Kupiec, cobertura incondicional al 5 %)")
    print("=" * 72)
    for _, r in df.iterrows():
        kp = r["kupiec_pvalue"]
        ok = "PASA " if (not np.isnan(kp) and kp > 0.05) else "FALLA"
        print(f"  {r['market']:5s} {MODEL_LABEL[r['model']]:9s} a={r['alpha']:.2f}  "
              f"tasa={100*r['kupiec_rate']:6.2f}%  esp={100*r['kupiec_expected_rate']:4.0f}%  "
              f"p_Kup={kp:8.4f}  [{ok}]")
    print(f"\nTabla LaTeX escrita en: {OUT}")


if __name__ == "__main__":
    main()