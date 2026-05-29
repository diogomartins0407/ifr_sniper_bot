# Sniper IFR2 Bot

Scanner institucional de sinais de **reversão à média via IFR2 (Stormer)** para o mercado brasileiro, com integração de dados oficiais de fluxo da B3 por tipo de investidor.

🌐 **App ao vivo:** [ifrsniperbot.streamlit.app](https://ifrsniperbot.streamlit.app/)

---

## O que faz

1. **Monitoramento de sinais** — Scan diário de uma lista curada de ativos B3 com cálculo do IFR2 (Wilder), Score Sniper composto, WR histórico no nível atual + setup, e indicadores de liquidez.
2. **Backtest por ativo** — Simulador com métricas institucionais (Sharpe, Sortino, Calmar, Max DD, Profit Factor) e opção de custos B3 reais.
3. **Fluxo B3 oficial** — Métricas de fluxo por tipo de investidor (Estrangeiro, Institucional, PF, Inst. Financeira), alertas de Sell/Buy Climax e heatmap calendário do ano corrente.

## Setup canônico operacional

```
Indicador:   IFR de Wilder, período 2
Entrada:     IFR2 ≤ 25, no fechamento do candle diário
Alvo:        Máxima dos 2 candles anteriores (alvo móvel)
Stop:        Time stop de 7 candles (canônico Stormer)
Filtro MM:   Não usado por default (Stormer puro)
Custos:      Teórico por default; opção de B3 reais (0,0825%) disponível
```

## Score Sniper (0–100)

Sistema composto de convicção que sintetiza 4 vetores:

| Componente | Peso |
|---|---:|
| IFR Score (intensidade sobrevenda) | 45 pts |
| Δ WR Score (prêmio de sobrevenda) | 25 pts |
| Fluxo B3 Score (sell climax estrangeiro) | 20 pts |
| Liquidez Score (ADTV) | 10 pts |

**Kill switch:** se IFR > limite operacional → Score = 0. Tabela ordenada por Score descendente — maior convicção no topo.

📐 Documentação matemática: [docs/SCORE_SNIPER.md](docs/SCORE_SNIPER.md)

## Stack técnico

- Python 3.11 (fixado via `runtime.txt`)
- Streamlit (UI), Plotly (gráficos), pandas/numpy (dados)
- yfinance (cotações), pandas-ta (indicadores legados)
- Scraping de fluxo B3 via `dadosdemercado.com.br/fluxo`

## Estrutura do repositório

```
.
├── ifr_sniper.py           # App Streamlit principal (3 abas)
├── b3_flow.py              # Módulo de fluxo B3 + heatmap calendário
├── requirements.txt        # Dependências Python
├── runtime.txt             # Python 3.11 (Streamlit Cloud)
├── .streamlit/config.toml  # Tema Deep Quant institucional
└── docs/SCORE_SNIPER.md    # Documentação do Score Sniper
```

## Limitações conhecidas

- **Survivorship bias** — yfinance retorna apenas ativos vivos hoje
- **Modo teórico** — Score e baseline não consideram custos transacionais por default
- **Setup mean-reversion** — sofre estruturalmente em bear markets persistentes

---

**Disclaimer:** Esta é uma ferramenta de pesquisa quantitativa. Operações reais devem ser validadas por análise fundamentalista antes de execução. Risco de mercado, liquidez e crédito não estão totalmente modelados.

Inspirado na estratégia IFR2 do Alexandre Wolwacz (Stormer).
