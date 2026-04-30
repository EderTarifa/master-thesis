# EVT-PPO: Drawdown-Aware Portfolio Management with Extreme Value Theory

Implementación del TFM "Extreme Value Theory in Reinforcement Learning
for financial portfolio management to minimize drawdown's impact".

## Estructura

```
master-thesis/
├── src/evt_ppo/
│   ├── data.py            # Descarga y limpieza de datos (Yahoo Finance)
│   ├── evt.py             # Estimadores EVT (POT-GPD, Block Maxima-GEV)
│   ├── drawdown.py        # Cálculo de drawdown y métricas relacionadas
│   ├── environment.py     # Entorno Gymnasium para portfolio management
│   ├── features.py        # Construcción del estado (mercado, portfolio, EVT)
│   ├── reward.py          # Funciones de recompensa (V0..V4)
│   ├── agent.py           # Wrapper PPO sobre Stable-Baselines3
│   ├── benchmarks.py      # Min-variance, 1/N, Markowitz
│   ├── metrics.py         # CAGR, Sharpe, MDD, Calmar, CVaR, etc.
│   ├── statistics.py      # Tests de hipótesis (t, Wilcoxon, block bootstrap)
│   ├── walkforward.py     # Protocolo walk-forward
│   ├── plots.py           # Visualizaciones
│   └── experiment.py      # Orquestación experimental
├── src/configs/           # Configuraciones YAML por experimento
├── scripts/               # Scripts de ejecución
├── tests/                 # Tests unitarios
└── results/               # Outputs (creado en runtime)
```

## Instalación

```bash
pip install -r requirements.txt
```

## Reproducir resultados

```bash
# 1. Descargar y preparar datos
python scripts/01_download_data.py

# 2. Test rápido (1 mercado, 1 ventana, 1 semilla, 1 variante)
python scripts/02_smoke_test.py

# 3. Experimento completo (3 mercados, 12 ventanas, 5 semillas, 5 variantes)
python scripts/03_run_full_experiment.py --config src/configs/full.yaml

# 4. Analizar y graficar
python scripts/04_analyze_results.py --results-dir results/full/
```

## Variantes implementadas

| ID | Estado incluye EVT | Reward incluye penalización drawdown | Reward incluye CVaR-EVT |
|----|--------------------|--------------------------------------|--------------------------|
| V0 | No  | No  | No  |
| V1 | No  | Sí  | No  |
| V2 | Sí  | Sí  | No  |
| V3 | No  | Sí  | Sí  |
| V4 | Sí  | Sí  | Sí  |

V0 = vanilla PPO. V1 = drawdown-aware sin EVT. V4 = propuesta del TFM.
La hipótesis principal contrasta V1 vs V4.
