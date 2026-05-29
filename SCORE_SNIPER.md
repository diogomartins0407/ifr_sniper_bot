# Score Sniper — Documentação Institucional

> Sistema composto de convicção (0–100) que sintetiza os múltiplos sinais técnicos e de fluxo da mesa Sniper Quant em uma única métrica ordenável.

**Versão:** 1.0 — 2026-05-27
**Localização do código:** `ifr_sniper.py`, função `calcular_score_sniper()`

---

## 1. Motivação institucional

Em um pregão típico, o scanner Sniper Quant pode acender múltiplos sinais simultâneos. Sem uma métrica única de priorização, o trader fica refém da intuição para decidir **qual ativo operar primeiro** com capital limitado.

O **Score Sniper** resolve isso compondo os 4 vetores de informação que a mesa já produz:

| Vetor | Origem | Peso |
|---|---|---:|
| Intensidade da sobrevenda | IFR2 do dia | **45 pts** |
| Prêmio estatístico atual | Δ WR (WR @ Nível − WR Setup) | **25 pts** |
| Contexto de fluxo institucional | Z-score do Estrangeiro (B3) | **20 pts** |
| Executabilidade | ADTV (volume médio em R$ MM) | **10 pts** |
| **Total** | | **100 pts** |

Tabela ordenada por Score descrescente — a **maior convicção aparece no topo**.

---

## 2. Pré-condição operacional (kill switch)

Se `IFR2 > ifr_limite` (limite operacional definido no sidebar, default = 25), o **Score é forçado a 0**, independentemente dos outros componentes. Justificativa: sem sinal técnico válido, não há trade — não importa quão bom esteja o resto do contexto.

```python
if ifr > ifr_limite:
    return 0.0
```

---

## 3. Componentes detalhados

### 3.1. IFR Score (até 45 pts)

Mede a **intensidade da sobrevenda** no momento atual. Quanto menor o IFR2, mais extremo o pânico, maior o edge teórico de mean reversion.

**Fórmula:**
```
ifr_score = 45 × (1 − IFR / ifr_limite)
```

**Pontuação de referência (limite=25):**

| IFR2 | IFR Score |
|---:|---:|
| 0 (extremo absoluto) | 45,0 |
| 5 | 36,0 |
| 10 | 27,0 |
| 15 | 18,0 |
| 20 | 9,0 |
| 25 (limite) | 0,0 |
| > 25 | **Score total = 0** (kill switch) |

### 3.2. Δ WR Score (até 25 pts)

Mede o **prêmio estatístico do nível atual** em relação ao baseline operacional. Δ WR é a diferença entre `WR @ Nível (3y)` (WR histórico no nível atual do IFR) e `WR Setup (3y)` (WR histórico do setup operacional fixo).

**Fórmula:**
```
delta_wr_score = clip(10 + Δ_WR × 1.0, 0, 25)
```

**Pontuação de referência:**

| Δ WR (pp) | Score | Interpretação |
|---:|---:|---|
| ≥ +15 | 25 | Prêmio máximo (estado histórico privilegiado) |
| +10 | 20 | Prêmio alto |
| +5 | 15 | Prêmio médio |
| 0 | 10 | Neutro (nível atual ≈ baseline) |
| −5 | 5 | Estado adverso leve |
| ≤ −10 | 0 | Estado adverso (sobrevenda atual não é confiável neste ativo) |

### 3.3. Fluxo B3 Score (até 20 pts)

Mede o **contexto macro institucional** via Z-score do fluxo do Investidor Estrangeiro nos últimos 30 dias. A lógica é **contrarian**: estrangeiro vendendo extremo (Z < −2σ) é **sinal de capitulação**, historicamente seguido de recuperação. Estrangeiro comprando extremo (Z > +2σ) é **sinal de exaustão de compra**.

**Fórmula:**
```
fluxo_score = clip(10 − Z_estrangeiro × 5, 0, 20)
```

**Pontuação de referência:**

| Z-score Estrangeiro | Score | Interpretação |
|---:|---:|---|
| ≤ −2,0 (Sell Climax) | 20 | Pânico institucional — oportunidade máxima |
| −1,0 | 15 | Pressão vendedora moderada |
| 0,0 (neutro) | 10 | Fluxo dentro da média |
| +1,0 | 5 | Compra acima da média |
| ≥ +2,0 (Buy Climax) | 0 | Estrangeiro já comprou tudo, sem espaço |

**Nota:** O Z-score é calculado em `b3_flow.calcular_metricas_fluxo()` usando média e desvio padrão dos últimos 30 dias do fluxo estrangeiro.

### 3.4. Liquidez Score (até 10 pts)

Mede a **executabilidade** do trade. ADTV (Average Daily Trading Volume) é calculado como `Vol_Médio × Preço`, em R$ milhões. Ativos com baixa liquidez sofrem slippage relevante mesmo em ordens institucionais médias.

**Pontuação por faixa:**

| ADTV (R$ MM/dia) | Score | Comentário |
|---:|---:|---|
| ≥ 100 | 10 | Blue chip, slippage desprezível |
| 50–100 | 8 | Mid cap líquida |
| 20–50 | 6 | Mid cap aceitável |
| 10–20 | 4 | Limite institucional |
| 5–10 | 2 | Cuidado com slippage |
| < 5 | 0 | Não-operacional para tamanho relevante |

---

## 4. Exemplo numérico

**Cenário:** PETR4 dispara sinal com IFR=8, Δ WR=+12pp, Estrangeiro em Sell Climax Z=−2,2, ADTV ≈ R$ 200 MM/dia. Limite operacional = 25.

```
1. IFR Score      = 45 × (1 − 8/25)    = 45 × 0,68  = 30,6 pts
2. Δ WR Score     = clip(10 + 12, 0, 25)              = 22,0 pts
3. Fluxo Score    = clip(10 − (−2,2)×5, 0, 20) = clip(21, 0, 20) = 20,0 pts
4. Liquidez Score = (ADTV >= 100)                     = 10,0 pts
─────────────────────────────────────────────────────────────
TOTAL                                                  = 82,6 pts
```

→ **Score 82,6 / 100** → faixa **CONVICÇÃO FORTE**, alocar peso pleno.

---

## 5. Interpretação por faixa (uso operacional)

| Faixa | Cor na tabela | Interpretação | Ação institucional |
|---:|---|---|---|
| **85–100** | 🟢 fundo verde | Convicção máxima — confluência total | **Alocar peso cheio (1.0×)** |
| **70–84** | 🟢 verde neon | Convicção forte | Alocar peso pleno (0.8×) |
| **50–69** | 🟡 amarelo dourado | Setup razoável | Alocar peso reduzido (0.5×) |
| **30–49** | branco | Setup fraco — sem edge claro | Operar só se for parte de cesta |
| **< 30** | 🔴 vermelho | Sem edge | **Não operar** |
| **0** | 🔴 vermelho | IFR acima do limite (kill switch) | **Sem sinal de compra** |

---

## 6. Validação por testes unitários

Os seguintes cenários foram validados (ver bloco de teste no histórico de implementação):

| Cenário | Score esperado | Score obtido | Status |
|---|---:|---:|:---:|
| Pânico máximo (IFR=2, Δ+15pp, Sell Climax forte, alta liquidez) | 90–100 | 96,4 | ✅ |
| Setup canônico forte (IFR=8, Δ+12pp, ADTV=200MM, Sell Climax) | 80–100 | 82,6 | ✅ |
| Setup ideal (IFR=5, Δ+10pp, ADTV=300MM) | 85–100 | 86,0 | ✅ |
| Setup razoável (IFR=20, Δ+5pp, fluxo neutro) | 35–55 | 42,0 | ✅ |
| Setup fraco (IFR=22, Δ=0pp, estrangeiro comprou) | 15–35 | 21,9 | ✅ |
| Sem sinal (IFR=35, acima limite) | 0 | 0,0 | ✅ |
| Sem sinal marginal (IFR=26, 1pt acima limite) | 0 | 0,0 | ✅ |

---

## 7. Limitações conhecidas e roadmap

### Limitações atuais
- **Pesos são opinativos**, não fitados estatisticamente. Os 45/25/20/10 refletem prioridade intuitiva da mesa, não otimização.
- **Z-score do Estrangeiro é macro**, igual para todos os ativos do dia. Não captura fluxo setor-específico.
- **ADTV é estimativa simples** (Vol_Médio × Preço atual). Não considera profundidade do book.
- **Score não é predítivo** — é uma síntese ordenadora. Validação requer track record real comparando Score declarado vs P&L realizado.

### Roadmap (próximas versões)
1. **v1.1** — Calibração dos pesos via grid search no baseline histórico (maximizar Sharpe agregado do top decile do Score)
2. **v1.2** — Fluxo setor-específico (separar fluxo por setor IBOV)
3. **v1.3** — Componente fundamental (penalizar Score em ativos com Dívida Líq/EBITDA > 4 ou ROIC < 0)
4. **v1.4** — Track record automático: logar cada decisão e medir hit rate por faixa de Score

---

## 8. Histórico de alterações

| Versão | Data | Mudança |
|---|---|---|
| 1.0 | 2026-05-27 | Implementação inicial com 4 componentes + kill switch |

---

## 9. Referências

- **Setup IFR2:** Alexandre Wolwacz (Stormer), `Stormer de A a Z`
- **Fluxo B3:** Dados oficiais via `dadosdemercado.com.br/fluxo` (B3 T+1)
- **Wilder RSI:** J. Welles Wilder, *New Concepts in Technical Trading Systems* (1978)
- **Backtest histórico baseline:** ver `reports/baseline_report.md`
