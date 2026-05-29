"""
================================================================================
B3 FLOW - Fluxo real de investidores na B3
================================================================================
Modulo standalone que captura dados OFICIAIS do fluxo de investimento na B3,
por tipo de investidor (Estrangeiro, Institucional, Pessoa Fisica, etc.).

Fonte primaria: dadosdemercado.com.br/fluxo (HTML estruturado, atualizacao
diaria, baseado em dados oficiais B3 - Resumo de Operacoes por Tipo de
Investidor, disponivel apos fechamento do pregao T+1).

Diferenca para indicadores tecnicos (OBV, MFI):
  - OBV mede volume com sinal de preco (proxy de momentum)
  - Fluxo B3 mede DINHEIRO REAL movimentado por categoria de investidor

Cache: 1 dia (B3 publica dados apenas T+1, nao adianta refazer scraping)
================================================================================
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "b3_flow.csv")

# URL primaria (HTML scraping)
URL_DADOSMERCADO = "https://www.dadosdemercado.com.br/fluxo"


def _parse_valor_br(valor: str) -> float:
    """
    Converte string brasileira de valor monetario em float (em milhoes).
    Aceita formatos:
        '1.234,56 mi'  -> 1234.56
        '-999,96 mi'   -> -999.96
        '1.234,56'     -> 1234.56
        ''             -> NaN
    """
    if not isinstance(valor, str) or not valor.strip():
        return np.nan
    s = valor.strip().replace("mi", "").replace("MI", "").strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _cache_valido(max_age_hours: int = 18) -> bool:
    """Retorna True se cache existe e foi atualizado nas ultimas N horas."""
    if not os.path.exists(CACHE_FILE):
        return False
    age_seconds = datetime.now().timestamp() - os.path.getmtime(CACHE_FILE)
    return age_seconds < (max_age_hours * 3600)


def _fetch_dadosmercado(days: int = 180) -> Optional[pd.DataFrame]:
    """
    Scraping da tabela HTML do dadosdemercado.com.br/fluxo.
    Usa pandas.read_html que e robusto a mudancas leves de layout.
    """
    try:
        # User-agent realistic para evitar bloqueio
        import urllib.request
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        req = urllib.request.Request(URL_DADOSMERCADO, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        tables = pd.read_html(html)
        if not tables:
            return None

        # Identifica a tabela com colunas de fluxo
        target_cols = {"Estrangeiro", "Institucional", "Pessoa fisica",
                       "Pessoa física", "Inst. Financeira", "Outros"}
        df = None
        for t in tables:
            if "Data" in t.columns and any(c in t.columns for c in target_cols):
                df = t.copy()
                break

        if df is None:
            return None

        # Normaliza nomes de colunas
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {
            "Pessoa fisica": "Pessoa_Fisica",
            "Pessoa física": "Pessoa_Fisica",
            "Inst. Financeira": "Inst_Financeira",
        }
        df = df.rename(columns=rename_map)

        # Parse valores
        for col in ["Estrangeiro", "Institucional", "Pessoa_Fisica",
                    "Inst_Financeira", "Outros"]:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(_parse_valor_br)

        # Parse data
        df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)

        # Limita a janela
        if days and len(df) > days:
            df = df.tail(days).reset_index(drop=True)

        return df
    except Exception as e:
        print(f"[b3_flow] Falha no fetch: {e}")
        return None


def fetch_fluxo_b3(days: int = 180, force_refresh: bool = False) -> pd.DataFrame:
    """
    Retorna DataFrame com fluxo B3 por tipo de investidor.
    Colunas: Data, Estrangeiro, Institucional, Pessoa_Fisica, Inst_Financeira, Outros
    Valores em R$ milhoes (positivo = compra liquida).

    Usa cache local se valido (atualizacao max 1x por dia).
    """
    if not force_refresh and _cache_valido():
        try:
            df = pd.read_csv(CACHE_FILE, parse_dates=["Data"])
            if not df.empty:
                return df
        except Exception:
            pass

    # Fetch fresh
    df = _fetch_dadosmercado(days=days)
    if df is None or df.empty:
        # Tenta cache mesmo expirado como fallback
        if os.path.exists(CACHE_FILE):
            try:
                return pd.read_csv(CACHE_FILE, parse_dates=["Data"])
            except Exception:
                pass
        return pd.DataFrame()

    # Salva cache
    try:
        df.to_csv(CACHE_FILE, index=False)
    except Exception:
        pass

    return df


def calcular_metricas_fluxo(df: pd.DataFrame) -> dict:
    """
    Calcula metricas institucionais sobre o fluxo:
      - fluxo_hoje: ultimo dia disponivel
      - mm_7d, mm_30d: medias moveis
      - z_score: anomalia estatistica (vs 30 dias)
      - sell_climax: True se Z < -2 sigma (oportunidade de mean reversion)
      - buy_climax: True se Z > +2 sigma (sinal de exaustao de compra)
      - regime: 'NET_VENDEDOR' ou 'NET_COMPRADOR' (acumulado 30d)
    """
    if df.empty:
        return {"erro": "Sem dados de fluxo disponivel"}

    out = {}
    for cat in ["Estrangeiro", "Institucional", "Pessoa_Fisica",
                "Inst_Financeira", "Outros"]:
        if cat not in df.columns:
            continue
        s = df[cat].dropna()
        if len(s) < 2:
            continue

        hoje = float(s.iloc[-1])
        mm7 = float(s.tail(7).mean()) if len(s) >= 7 else hoje
        mm30 = float(s.tail(30).mean()) if len(s) >= 30 else hoje
        acum30 = float(s.tail(30).sum()) if len(s) >= 30 else float(s.sum())

        # Z-score do dia vs media 30d
        if len(s) >= 30:
            mu = s.tail(30).mean()
            sigma = s.tail(30).std()
            z = (hoje - mu) / sigma if sigma > 0 else 0.0
        else:
            z = 0.0

        out[cat] = {
            "hoje": hoje,
            "mm_7d": mm7,
            "mm_30d": mm30,
            "acum_30d": acum30,
            "z_score": float(z),
            "sell_climax": bool(z < -2.0),   # venda extrema
            "buy_climax": bool(z > 2.0),     # compra extrema (exaustao)
            "regime_30d": "COMPRADOR" if acum30 > 0 else "VENDEDOR",
        }

    out["data_referencia"] = df["Data"].max().strftime("%Y-%m-%d") if not df.empty else None
    return out


def gerar_alertas_institucionais(metricas: dict) -> list:
    """
    Gera lista de alertas baseado nas metricas.
    Para Sniper Quant: foco em sell climax do estrangeiro (mean reversion edge).
    """
    alertas = []
    if "erro" in metricas:
        return [{"tipo": "ERRO", "msg": metricas["erro"]}]

    est = metricas.get("Estrangeiro", {})
    if est.get("sell_climax"):
        alertas.append({
            "tipo": "🎯 OPORTUNIDADE MEAN REVERSION",
            "msg": (f"Estrangeiro em SELL CLIMAX hoje "
                    f"(R$ {est['hoje']:+,.0f}mi, Z={est['z_score']:+.2f}). "
                    f"Historicamente associado a recuperacao 3-5 dias a frente. "
                    f"Sinergico com IFR2 baixo - sobreponderar entradas.")
        })
    if est.get("buy_climax"):
        alertas.append({
            "tipo": "⚠️ EXAUSTAO DE COMPRA",
            "msg": (f"Estrangeiro em BUY CLIMAX hoje "
                    f"(R$ {est['hoje']:+,.0f}mi, Z={est['z_score']:+.2f}). "
                    f"Cuidado com entradas - rali pode estar no fim.")
        })
    if est.get("regime_30d") == "VENDEDOR" and abs(est.get("acum_30d", 0)) > 10000:
        alertas.append({
            "tipo": "📉 REGIME VENDEDOR PERSISTENTE",
            "msg": (f"Estrangeiro vendeu liquido R$ {est['acum_30d']:+,.0f}mi "
                    f"nos ultimos 30d. Bear market estrutural - operar com size reduzido.")
        })
    elif est.get("regime_30d") == "COMPRADOR" and abs(est.get("acum_30d", 0)) > 10000:
        alertas.append({
            "tipo": "📈 REGIME COMPRADOR PERSISTENTE",
            "msg": (f"Estrangeiro comprou liquido R$ {est['acum_30d']:+,.0f}mi "
                    f"nos ultimos 30d. Bull market - aproveitar IFR2 com convicao.")
        })

    if not alertas:
        alertas.append({
            "tipo": "🟡 REGIME NEUTRO",
            "msg": "Fluxo dentro de bandas normais. Operar setup IFR2 padrao."
        })
    return alertas



# ================================================================================
# HEATMAP CALENDARIO (estilo GitHub) - Visualizacao de regime de fluxo
# ================================================================================

def construir_matriz_calendario(df: pd.DataFrame, categoria: str = "Estrangeiro",
                                 ano: int = None) -> dict:
    """
    Constroi matriz 5xN para heatmap calendario.
    Linhas: dias uteis (Seg=0 ... Sex=4)
    Colunas: semanas do ano

    Retorna dict com:
      - z_matrix: matriz numpy de valores (em milhoes R$, NaN se sem pregao)
      - z_norm: mesma matriz normalizada para escala de cor [-1, +1]
      - x_dates: data representativa de cada coluna (segunda-feira da semana)
      - y_labels: ["Seg", "Ter", "Qua", "Qui", "Sex"]
      - z_text: texto de hover (valor formatado por celula)
      - month_ticks: dict {data_inicio_mes: nome_mes} para xticks
    """
    import numpy as np
    if df.empty or categoria not in df.columns:
        return {}

    if ano is None:
        ano = df["Data"].dt.year.max()

    df_ano = df[df["Data"].dt.year == ano].copy()
    if df_ano.empty:
        return {}

    df_ano["dow"] = df_ano["Data"].dt.dayofweek  # 0=Seg, 4=Sex
    df_ano = df_ano[df_ano["dow"] <= 4]  # so dias uteis
    # Semana relativa ao primeiro pregao do ano (0-indexed)
    primeiro_dia = df_ano["Data"].min()
    df_ano["semana_rel"] = ((df_ano["Data"] - primeiro_dia).dt.days // 7).astype(int)

    n_semanas = int(df_ano["semana_rel"].max()) + 1
    z = np.full((5, n_semanas), np.nan)
    z_text = np.full((5, n_semanas), "", dtype=object)
    x_dates = []
    for s in range(n_semanas):
        # Data representativa: primeira data daquela semana
        rows_sem = df_ano[df_ano["semana_rel"] == s]
        if not rows_sem.empty:
            x_dates.append(rows_sem["Data"].min())
        else:
            # placeholder caso semana vazia
            x_dates.append(primeiro_dia + pd.Timedelta(days=s*7))

    for _, row in df_ano.iterrows():
        valor = row[categoria]
        if pd.isna(valor):
            continue
        dow = int(row["dow"])
        sem = int(row["semana_rel"])
        z[dow, sem] = valor
        z_text[dow, sem] = (f"{row['Data'].strftime('%d/%m')} "
                            f"({row['Data'].strftime('%a')})<br>"
                            f"R$ {valor:+,.0f} mi")

    # Normalizacao para escala de cor: usa percentis para robustez vs outliers
    valores_validos = z[~np.isnan(z)]
    if len(valores_validos) > 0:
        p95 = np.percentile(np.abs(valores_validos), 95)
        z_norm = np.clip(z / p95, -1.0, 1.0) if p95 > 0 else np.zeros_like(z)
    else:
        z_norm = np.zeros_like(z)

    # Ticks de mes: pega 1a data de cada mes presente
    month_ticks = {}
    df_ano_sorted = df_ano.sort_values("Data")
    for mes in range(1, 13):
        df_mes = df_ano_sorted[df_ano_sorted["Data"].dt.month == mes]
        if not df_mes.empty:
            d = df_mes["Data"].iloc[0]
            month_ticks[d] = d.strftime("%b").capitalize()

    return {
        "z_matrix": z,
        "z_norm": z_norm,
        "x_dates": x_dates,
        "y_labels": ["Seg", "Ter", "Qua", "Qui", "Sex"],
        "z_text": z_text,
        "month_ticks": month_ticks,
        "ano": ano,
    }


def estatisticas_calendario(df: pd.DataFrame, categoria: str = "Estrangeiro",
                             ano: int = None) -> dict:
    """
    Estatisticas para painel lateral do heatmap:
      - total_pregoes
      - pregoes_compra (e %)
      - pregoes_venda (e %)
      - maior_sequencia_compra (dias consecutivos + mes em que ocorreu)
      - maior_sequencia_venda (dias consecutivos + mes em que ocorreu)
      - mes_mais_verde (mes com maior soma positiva)
      - mes_mais_vermelho (mes com maior soma negativa)
    """
    if df.empty or categoria not in df.columns:
        return {}

    if ano is None:
        ano = df["Data"].dt.year.max()

    df_ano = df[df["Data"].dt.year == ano].copy().sort_values("Data")
    if df_ano.empty:
        return {}

    serie = df_ano[categoria].dropna()
    total = len(serie)
    compra = int((serie > 0).sum())
    venda = int((serie < 0).sum())

    # Maior sequencia de compra/venda
    def maior_run(serie_bool, datas):
        max_run = 0
        run_atual = 0
        mes_max = None
        meses_run = []
        for v, d in zip(serie_bool.values, datas.values):
            if v:
                run_atual += 1
                meses_run.append(pd.Timestamp(d).month)
                if run_atual > max_run:
                    max_run = run_atual
                    # Mes mais frequente no run
                    from collections import Counter
                    mes_max = Counter(meses_run).most_common(1)[0][0]
            else:
                run_atual = 0
                meses_run = []
        return max_run, mes_max

    run_compra, mes_run_compra = maior_run(df_ano[categoria] > 0, df_ano["Data"])
    run_venda, mes_run_venda = maior_run(df_ano[categoria] < 0, df_ano["Data"])

    # Soma por mes
    df_ano["mes_num"] = df_ano["Data"].dt.month
    soma_por_mes = df_ano.groupby("mes_num")[categoria].sum()

    meses_nomes = {1:"Jan", 2:"Fev", 3:"Mar", 4:"Abr", 5:"Mai", 6:"Jun",
                   7:"Jul", 8:"Ago", 9:"Set", 10:"Out", 11:"Nov", 12:"Dez"}

    if not soma_por_mes.empty:
        mes_verde_num = int(soma_por_mes.idxmax())
        mes_vermelho_num = int(soma_por_mes.idxmin())
        mes_verde = meses_nomes.get(mes_verde_num, "-")
        mes_vermelho = meses_nomes.get(mes_vermelho_num, "-")
        valor_mes_verde = float(soma_por_mes.max())
        valor_mes_vermelho = float(soma_por_mes.min())
    else:
        mes_verde, mes_vermelho = "-", "-"
        valor_mes_verde, valor_mes_vermelho = 0.0, 0.0

    return {
        "ano": ano,
        "total_pregoes": total,
        "pregoes_compra": compra,
        "pct_compra": (compra/total*100) if total else 0,
        "pregoes_venda": venda,
        "pct_venda": (venda/total*100) if total else 0,
        "maior_sequencia_compra": run_compra,
        "mes_run_compra": meses_nomes.get(mes_run_compra, "-") if mes_run_compra else "-",
        "maior_sequencia_venda": run_venda,
        "mes_run_venda": meses_nomes.get(mes_run_venda, "-") if mes_run_venda else "-",
        "mes_mais_verde": mes_verde,
        "valor_mes_verde": valor_mes_verde,
        "mes_mais_vermelho": mes_vermelho,
        "valor_mes_vermelho": valor_mes_vermelho,
    }
